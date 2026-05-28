"""
TEAL calibration: collect activation histograms from calibration data.

Runs the model on a small dataset (C4 or Alpaca) and records the magnitude
distribution of activations at each projection's input. These histograms
are used by greedy_opt.py to find optimal per-layer sparsity thresholds.

Usage:
    python scripts/calibrate.py --model mistralai/Mistral-7B-v0.3 --samples 300
"""

import os
import json
import torch
import torch.nn as nn
from typing import Optional

from .patching import _get_layers, _get_projection, PROJ_NAMES


def collect_activation_histograms(
    model,
    tokenizer,
    num_samples: int = 300,
    seq_len: int = 512,
    num_bins: int = 2000,
    dataset_name: str = "allenai/c4",
    dataset_split: str = "train",
    dataset_config: str = "en",
    save_dir: Optional[str] = None,
) -> dict:
    """
    Collect activation magnitude histograms for all projections.

    Args:
        model: HuggingFace CausalLM.
        tokenizer: corresponding tokenizer.
        num_samples: number of calibration samples.
        seq_len: sequence length per sample.
        num_bins: histogram resolution.
        dataset_name: HuggingFace dataset to use.
        dataset_split: dataset split.
        dataset_config: dataset config name.
        save_dir: if provided, save histograms to this directory.

    Returns:
        dict mapping (layer_idx, proj_name) to {counts, edges} histogram.
    """
    from datasets import load_dataset

    print(f"[TEAL Calibration] Loading {num_samples} samples from {dataset_name}...")
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True)

    # Tokenize calibration samples
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        text = example.get("text", "")
        tokens = tokenizer(
            text, return_tensors="pt", max_length=seq_len,
            truncation=True, padding=False,
        )
        if tokens["input_ids"].shape[1] >= 32:  # skip very short samples
            samples.append(tokens["input_ids"])

    print(f"[TEAL Calibration] Collected {len(samples)} valid samples.")

    layers = _get_layers(model)
    num_layers = len(layers)

    # Set up histogram accumulators
    # We track magnitude (abs value) histograms for each projection input
    histograms = {}
    _running_min = {}
    _running_max = {}

    # First pass: find magnitude ranges
    print("[TEAL Calibration] Pass 1/2: Finding activation ranges...")
    hooks = []
    activation_stats = {}

    for layer_idx in range(num_layers):
        for proj_name in PROJ_NAMES:
            key = (layer_idx, proj_name)
            activation_stats[key] = {"min": float("inf"), "max": 0.0}

    def make_range_hook(key):
        def hook_fn(module, args):
            x = args[0]
            mag = x.abs()
            stats = activation_stats[key]
            stats["min"] = min(stats["min"], mag.min().item())
            stats["max"] = max(stats["max"], mag.max().item())
        return hook_fn

    for layer_idx, layer in enumerate(layers):
        for proj_name in PROJ_NAMES:
            proj = _get_projection(layer, proj_name)
            key = (layer_idx, proj_name)
            handle = proj.register_forward_pre_hook(make_range_hook(key))
            hooks.append(handle)

    # Run a subset of samples for range estimation
    range_samples = min(20, len(samples))
    for i in range(range_samples):
        with torch.no_grad():
            model(samples[i].to(model.device))

    for h in hooks:
        h.remove()

    # Second pass: build histograms
    print("[TEAL Calibration] Pass 2/2: Building histograms...")
    hist_data = {}
    hooks = []

    for layer_idx, layer in enumerate(layers):
        for proj_name in PROJ_NAMES:
            key = (layer_idx, proj_name)
            stats = activation_stats[key]
            max_val = stats["max"] * 1.1  # 10% margin

            hist_data[key] = {
                "counts": torch.zeros(num_bins, dtype=torch.long),
                "max_val": max_val,
            }

    def make_hist_hook(key):
        def hook_fn(module, args):
            x = args[0]
            mag = x.abs().flatten()
            max_val = hist_data[key]["max_val"]
            # Bin magnitudes into [0, max_val] range
            bins = (mag / (max_val + 1e-10) * num_bins).long().clamp(0, num_bins - 1)
            hist_data[key]["counts"].scatter_add_(
                0, bins.cpu(), torch.ones_like(bins, dtype=torch.long, device="cpu")
            )
        return hook_fn

    for layer_idx, layer in enumerate(layers):
        for proj_name in PROJ_NAMES:
            proj = _get_projection(layer, proj_name)
            key = (layer_idx, proj_name)
            handle = proj.register_forward_pre_hook(make_hist_hook(key))
            hooks.append(handle)

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"  Sample {i + 1}/{len(samples)}")
        with torch.no_grad():
            model(sample.to(model.device))

    for h in hooks:
        h.remove()

    # Package results
    results = {}
    for key, data in hist_data.items():
        layer_idx, proj_name = key
        max_val = data["max_val"]
        edges = torch.linspace(0, max_val, num_bins + 1)
        results[key] = {
            "counts": data["counts"],
            "edges": edges,
            "max_val": max_val,
            "total": data["counts"].sum().item(),
        }

    # Save to disk
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for (layer_idx, proj_name), hist in results.items():
            path = os.path.join(save_dir, f"layer{layer_idx}_{proj_name}.pt")
            torch.save({
                "counts": hist["counts"],
                "edges": hist["edges"],
                "max_val": hist["max_val"],
            }, path)
        print(f"[TEAL Calibration] Saved histograms to {save_dir}")

    return results


def load_histograms(hist_dir: str) -> dict:
    """Load precomputed histograms from disk."""
    results = {}
    for fname in os.listdir(hist_dir):
        if not fname.endswith(".pt"):
            continue
        # Parse layer{idx}_{proj_name}.pt
        base = fname.replace(".pt", "")
        parts = base.split("_", 1)
        layer_idx = int(parts[0].replace("layer", ""))
        proj_name = parts[1]
        data = torch.load(os.path.join(hist_dir, fname), weights_only=True)
        results[(layer_idx, proj_name)] = data
    return results
