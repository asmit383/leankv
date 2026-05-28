"""
Throughput and VRAM benchmark harness.

Usage:
    # Baseline
    python scripts/benchmark.py --model mistralai/Mistral-7B-v0.3

    # TEAL only
    python scripts/benchmark.py --model mistralai/Mistral-7B-v0.3 --teal 0.4

    # TurboQuant only
    python scripts/benchmark.py --model mistralai/Mistral-7B-v0.3 --tq-bits 3

    # Combined
    python scripts/benchmark.py --model mistralai/Mistral-7B-v0.3 --teal 0.4 --tq-bits 3

    # With config file
    python scripts/benchmark.py --config configs/mistral7b.yaml --teal 0.4 --tq-bits 3
"""

import argparse
import csv
import os
import sys
import time
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leankv.combined import load_model, optimize_model, reset_optimizations
from leankv.teal.patching import print_sparsity_report


def benchmark_config(
    model, tokenizer, prompt, batch_sizes, max_new_tokens=128, num_runs=5,
    cache_factory=None,
):
    """Run benchmarks across batch sizes."""
    results = []

    for bs in batch_sizes:
        prompts = [prompt] * bs
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Create a fresh cache per benchmark run
        cache = cache_factory() if cache_factory else None

        # Warmup
        with torch.no_grad():
            try:
                model.generate(
                    **inputs, max_new_tokens=10,
                    past_key_values=cache,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at batch_size={bs} (warmup), skipping.")
                results.append({
                    "batch_size": bs, "tok_s": 0, "vram_gb": -1, "status": "OOM",
                })
                torch.cuda.empty_cache()
                continue

        torch.cuda.reset_peak_memory_stats()
        times = []

        for run in range(num_runs):
            cache = cache_factory() if cache_factory else None
            torch.cuda.synchronize()
            start = time.perf_counter()

            try:
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        past_key_values=cache,
                    )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at batch_size={bs} (run {run+1}), skipping.")
                times = []
                break

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            num_generated = out.shape[1] - inputs["input_ids"].shape[1]
            total_tokens = num_generated * bs
            tok_s = total_tokens / elapsed
            times.append(tok_s)

        vram_gb = torch.cuda.max_memory_allocated() / 1e9

        if times:
            avg_tok_s = sum(times) / len(times)
            results.append({
                "batch_size": bs,
                "tok_s": round(avg_tok_s, 1),
                "vram_gb": round(vram_gb, 2),
                "status": "OK",
            })
            print(f"  B={bs}: {avg_tok_s:.1f} tok/s, {vram_gb:.2f} GB VRAM")
        else:
            results.append({
                "batch_size": bs, "tok_s": 0, "vram_gb": -1, "status": "OOM",
            })

        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="LeanKV benchmark")
    parser.add_argument("--model", type=str, help="HuggingFace model name")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--teal", type=float, default=None, help="TEAL sparsity (0.0-1.0)")
    parser.add_argument("--teal-thresholds", type=str, default=None, help="Calibrated thresholds path")
    parser.add_argument("--tq-bits", type=int, default=None, help="TurboQuant bits (2-4)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1], help="Batch sizes")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--num-runs", type=int, default=5, help="Runs per measurement")
    parser.add_argument("--prompt", type=str, default="Explain the theory of relativity in simple terms.")
    parser.add_argument("--output", type=str, default=None, help="CSV output path")
    args = parser.parse_args()

    # Resolve model
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg["model_name"]
        if args.batch_sizes == [1]:
            args.batch_sizes = cfg["benchmark"]["batch_sizes"]
    elif args.model:
        model_name = args.model
    else:
        parser.error("Either --model or --config is required")

    # Load model
    model, tokenizer = load_model(model_name)

    # Apply optimizations
    model, cache = optimize_model(
        model,
        teal_sparsity=args.teal,
        teal_thresholds_path=args.teal_thresholds,
        turboquant_bits=args.tq_bits,
    )

    # Build config label
    parts = ["baseline"]
    if args.teal or args.teal_thresholds:
        parts = [f"teal_{int((args.teal or 0)*100)}pct"]
    if args.tq_bits:
        parts.append(f"tq_{args.tq_bits}bit")
    config_label = "+".join(parts)

    print(f"\n{'='*60}")
    print(f"Benchmark: {model_name}")
    print(f"Config: {config_label}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"{'='*60}\n")

    # Create cache factory if TurboQuant enabled
    cache_factory = None
    if args.tq_bits:
        config = model.config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        def cache_factory():
            return TurboQuantCache(
                bits=args.tq_bits,
                head_dim=head_dim,
                num_layers=config.num_hidden_layers,
                device=model.device,
            )
        from leankv.turboquant.cache import TurboQuantCache

    results = benchmark_config(
        model, tokenizer, args.prompt, args.batch_sizes,
        max_new_tokens=args.max_tokens, num_runs=args.num_runs,
        cache_factory=cache_factory,
    )

    # Print sparsity report if TEAL enabled
    if args.teal or args.teal_thresholds:
        print("\nTEAL Sparsity Report:")
        print_sparsity_report(model)

    # Save CSV
    if args.output:
        os.makedirs(os.path.dirname(args.output) or "results", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "config", "batch_size", "tok_s", "vram_gb", "status",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "model": model_name,
                    "config": config_label,
                    **r,
                })
        print(f"\nResults saved to {args.output}")

    # Summary table
    print(f"\n{'Batch':<8}{'Tok/s':<12}{'VRAM (GB)':<12}{'Status':<8}")
    print("-" * 40)
    for r in results:
        print(f"{r['batch_size']:<8}{r['tok_s']:<12}{r['vram_gb']:<12}{r['status']:<8}")


if __name__ == "__main__":
    main()
