"""
Raw baseline benchmark — no LeanKV dependencies.
Tests Mistral 7B and Llama 3 8B throughput with and without torch.compile.

Usage:
    python3 scripts/raw_baseline.py
    python3 scripts/raw_baseline.py --model mistralai/Mistral-7B-v0.3
    python3 scripts/raw_baseline.py --model meta-llama/Meta-Llama-3-8B
    python3 scripts/raw_baseline.py --compile           # enable torch.compile
    python3 scripts/raw_baseline.py --batch-sizes 1 4 8 16 32
    python3 scripts/raw_baseline.py --max-tokens 2048   # longer sequences
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_system_info():
    print("=" * 60)
    print("System Info")
    print("=" * 60)
    print(f"PyTorch:       {torch.__version__}")
    import transformers
    print(f"Transformers:  {transformers.__version__}")
    print(f"CUDA:          {torch.version.cuda}")
    print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)


def benchmark(model, tokenizer, prompt, batch_size, max_new_tokens, num_runs):
    prompts = [prompt] * batch_size
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    torch.cuda.reset_peak_memory_stats()
    times = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated = out.shape[1] - inputs["input_ids"].shape[1]
        total_tokens = generated * batch_size
        tok_s = total_tokens / elapsed
        per_seq = tok_s / batch_size
        times.append(tok_s)
        print(f"  Run {i+1}/{num_runs}: {tok_s:.1f} tok/s total, {per_seq:.1f} tok/s/seq, {elapsed:.2f}s")

    vram = torch.cuda.max_memory_allocated() / 1e9
    avg = sum(times) / len(times)
    return avg, vram


def main():
    parser = argparse.ArgumentParser(description="Raw baseline benchmark")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--prompt", type=str,
                        default="Explain the theory of relativity in simple terms.")
    args = parser.parse_args()

    print_system_info()

    print(f"\nLoading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    print("Model loaded.")

    if args.compile:
        print("Running torch.compile (first inference will be slow)...")
        model = torch.compile(model)

    compile_label = "torch.compile" if args.compile else "no compile"
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {args.model}")
    print(f"Mode: {compile_label}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"{'=' * 60}")

    results = []
    for bs in args.batch_sizes:
        print(f"\n--- Batch size {bs} ---")
        try:
            avg, vram = benchmark(
                model, tokenizer, args.prompt, bs,
                args.max_tokens, args.num_runs,
            )
            results.append((bs, avg, avg / bs, vram, "OK"))
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch_size={bs}")
            results.append((bs, 0, 0, -1, "OOM"))
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"Results: {args.model} | {compile_label} | {args.max_tokens} tokens")
    print(f"{'=' * 70}")
    print(f"{'Batch':<8}{'Total tok/s':<15}{'Per-seq tok/s':<16}{'VRAM (GB)':<12}{'Status':<8}")
    print("-" * 59)
    for bs, total, per_seq, vram, status in results:
        if status == "OK":
            print(f"{bs:<8}{total:<15.1f}{per_seq:<16.1f}{vram:<12.2f}{status:<8}")
        else:
            print(f"{bs:<8}{'—':<15}{'—':<16}{'—':<12}{status:<8}")

    # KV cache size estimate
    print(f"\n--- KV Cache Size Estimate ---")
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    bytes_per_token = num_layers * num_kv_heads * head_dim * 2 * 2  # 2 for K+V, 2 for FP16
    print(f"  Per token: {bytes_per_token / 1024:.1f} KB")
    print(f"  Per sequence ({args.max_tokens} tokens): {bytes_per_token * args.max_tokens / 1e6:.1f} MB")
    for bs in args.batch_sizes:
        total_mb = bytes_per_token * args.max_tokens * bs / 1e6
        print(f"  B={bs} x {args.max_tokens} tokens: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
