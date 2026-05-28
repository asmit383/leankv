"""Utility functions for VRAM tracking, timing, and model loading."""

import time
import contextlib
import torch


@contextlib.contextmanager
def track_time(label=""):
    """Context manager that measures wall-clock time."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if label:
        print(f"[{label}] {elapsed:.3f}s")


def get_vram_usage_gb():
    """Return current peak GPU memory allocated in GB."""
    return torch.cuda.max_memory_allocated() / 1e9


def reset_vram_tracking():
    """Reset peak memory stats."""
    torch.cuda.reset_peak_memory_stats()


def get_vram_current_gb():
    """Return currently allocated GPU memory in GB."""
    return torch.cuda.memory_allocated() / 1e9


def measure_throughput(model, tokenizer, prompts, max_new_tokens=128, num_runs=5,
                       past_key_values=None):
    """Measure generation throughput in tokens/second.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompts: Single string or list of strings (for batched inference).
        max_new_tokens: Number of tokens to generate.
        num_runs: Number of benchmark runs (after 1 warmup).
        past_key_values: Optional cache object (e.g. TurboQuantCache).

    Returns:
        dict with avg_tok_s, all_tok_s, peak_vram_gb.
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, past_key_values=past_key_values)

    reset_vram_tracking()
    times = []

    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                past_key_values=past_key_values,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        num_generated = out.shape[1] - inputs["input_ids"].shape[1]
        total_tokens = num_generated * len(prompts)
        times.append(total_tokens / elapsed)

    return {
        "avg_tok_s": sum(times) / len(times),
        "all_tok_s": times,
        "peak_vram_gb": get_vram_usage_gb(),
        "batch_size": len(prompts),
    }
