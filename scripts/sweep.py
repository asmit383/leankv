"""
Grid sweep: batch_size x sparsity x bits.

Runs the full benchmark grid and outputs a CSV for analysis.

Usage:
    python scripts/sweep.py --config configs/mistral7b.yaml
    python scripts/sweep.py --model mistralai/Mistral-7B-v0.3 --output results/sweep.csv
"""

import argparse
import csv
import itertools
import os
import sys
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leankv.combined import load_model, optimize_model, reset_optimizations
from leankv.turboquant.cache import TurboQuantCache


def run_single_benchmark(model, tokenizer, prompt, batch_size, max_tokens, num_runs, cache_factory):
    """Run a single benchmark configuration and return results."""
    import time

    prompts = [prompt] * batch_size
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    cache = cache_factory() if cache_factory else None

    # Warmup
    try:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10, past_key_values=cache)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None  # OOM

    torch.cuda.reset_peak_memory_stats()
    times = []

    for _ in range(num_runs):
        cache = cache_factory() if cache_factory else None
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_tokens, past_key_values=cache)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        num_generated = out.shape[1] - inputs["input_ids"].shape[1]
        times.append(num_generated * batch_size / elapsed)

    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.empty_cache()

    return {
        "tok_s": round(sum(times) / len(times), 1),
        "vram_gb": round(vram_gb, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="LeanKV grid sweep")
    parser.add_argument("--model", type=str, help="HuggingFace model name")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--output", type=str, default="results/sweep.csv")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg["model_name"]
        sparsity_levels = [0.0] + cfg["teal"]["sparsity_levels"]
        bit_widths = [None] + [int(b) for b in cfg["turboquant"]["bit_widths"]]
        batch_sizes = cfg["benchmark"]["batch_sizes"]
        prompt = cfg["benchmark"]["prompt"]
    else:
        model_name = args.model
        sparsity_levels = [0.0, 0.3, 0.4, 0.5]
        bit_widths = [None, 4, 3]
        batch_sizes = [1, 4, 8, 16, 32]
        prompt = "Explain the theory of relativity in simple terms."

    model, tokenizer = load_model(model_name)
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    # Generate all configurations
    configs = list(itertools.product(sparsity_levels, bit_widths, batch_sizes))
    total = len(configs)

    os.makedirs(os.path.dirname(args.output) or "results", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "sparsity", "tq_bits", "batch_size", "tok_s", "vram_gb", "status",
        ])
        writer.writeheader()

        for i, (sparsity, tq_bits, bs) in enumerate(configs):
            label = f"s={sparsity:.0%}, tq={tq_bits or 'none'}, B={bs}"
            print(f"\n[{i+1}/{total}] {label}")

            # Reset and reapply optimizations
            reset_optimizations(model)
            teal_s = sparsity if sparsity > 0 else None
            model, _ = optimize_model(
                model, teal_sparsity=teal_s, turboquant_bits=tq_bits,
            )

            cache_factory = None
            if tq_bits:
                def cache_factory(bits=tq_bits):
                    return TurboQuantCache(
                        bits=bits, head_dim=head_dim,
                        num_layers=config.num_hidden_layers,
                        device=model.device,
                    )

            result = run_single_benchmark(
                model, tokenizer, prompt, bs, args.max_tokens, args.num_runs,
                cache_factory,
            )

            row = {
                "model": model_name,
                "sparsity": sparsity,
                "tq_bits": tq_bits or "none",
                "batch_size": bs,
            }
            if result:
                row.update(result)
                row["status"] = "OK"
                print(f"  → {result['tok_s']} tok/s, {result['vram_gb']} GB")
            else:
                row["tok_s"] = 0
                row["vram_gb"] = -1
                row["status"] = "OOM"
                print(f"  → OOM")

            writer.writerow(row)
            f.flush()

    print(f"\nSweep complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
