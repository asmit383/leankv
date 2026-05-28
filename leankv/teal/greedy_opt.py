"""
Block-wise greedy sparsity allocation for TEAL.

Given a global sparsity budget (e.g., 40%), finds the optimal per-projection
sparsity distribution that minimizes L2 output error. This is Algorithm 1
from the TEAL paper.

The optimizer works on precomputed activation histograms (from calibration.py)
and does NOT require GPU access — it's a CPU-only optimization.

Usage:
    python scripts/calibrate.py --model mistral7b --sparsity 0.4
"""

import json
import os
import torch
import numpy as np
from typing import Optional

from .sparse_fns import compute_threshold_from_histogram
from .patching import PROJ_NAMES


# Relative parameter counts for each projection type (Llama/Mistral architecture)
# These are used to weight the sparsity budget allocation
# q_proj: hidden_size x hidden_size
# k_proj: hidden_size x (hidden_size / num_heads * num_kv_heads)
# v_proj: same as k_proj
# o_proj: hidden_size x hidden_size
# gate_proj: hidden_size x intermediate_size
# up_proj: hidden_size x intermediate_size
# down_proj: intermediate_size x hidden_size
_DEFAULT_PARAM_WEIGHTS = {
    "q_proj": 1.0,
    "k_proj": 0.25,    # GQA: typically 1/4 of q_proj for 8 KV heads / 32 heads
    "v_proj": 0.25,
    "o_proj": 1.0,
    "gate_proj": 3.5,  # intermediate_size / hidden_size ≈ 3.5
    "up_proj": 3.5,
    "down_proj": 3.5,
}


def greedy_optimize(
    histograms: dict,
    target_sparsity: float,
    num_layers: int,
    param_weights: Optional[dict] = None,
    step_size: float = 0.05,
) -> dict:
    """
    Block-wise greedy sparsity allocation.

    For each increment step, tries increasing sparsity on each projection
    and picks the one that would cause the least additional error (estimated
    from the histogram — higher magnitude activations being pruned = more error).

    Args:
        histograms: dict from calibration.py mapping (layer_idx, proj_name) to
            {counts, edges} histogram data.
        target_sparsity: global sparsity target (0.0 to 1.0).
        num_layers: number of transformer layers.
        param_weights: relative parameter counts per projection type.
        step_size: sparsity increment per step (default 0.05 = 5%).

    Returns:
        dict mapping {layer_idx: {proj_name: threshold_value}}.
    """
    if param_weights is None:
        param_weights = _DEFAULT_PARAM_WEIGHTS

    total_weight = sum(param_weights[p] for p in PROJ_NAMES) * num_layers

    # Initialize per-projection sparsity levels to 0
    sparsity_levels = {}
    for layer_idx in range(num_layers):
        sparsity_levels[layer_idx] = {p: 0.0 for p in PROJ_NAMES}

    # Weighted sparsity increment per step
    # delta_i = step_size * total_weight / weight_i
    # This ensures each step contributes equally to the global sparsity budget
    current_global_sparsity = 0.0

    while current_global_sparsity < target_sparsity:
        best_key = None
        best_error = float("inf")
        best_new_sparsity = 0.0

        for layer_idx in range(num_layers):
            for proj_name in PROJ_NAMES:
                key = (layer_idx, proj_name)
                if key not in histograms:
                    continue

                current_s = sparsity_levels[layer_idx][proj_name]
                new_s = min(current_s + step_size, 0.95)  # cap at 95%

                if new_s <= current_s:
                    continue

                # Estimate error: sum of magnitudes in the newly pruned band
                # (activations between old threshold and new threshold)
                error = _estimate_pruning_error(
                    histograms[key], current_s, new_s
                )

                # Weight by parameter count — pruning a large projection
                # saves more compute, so we normalize error by weight
                weighted_error = error / (param_weights[proj_name] + 1e-10)

                if weighted_error < best_error:
                    best_error = weighted_error
                    best_key = (layer_idx, proj_name)
                    best_new_sparsity = new_s

        if best_key is None:
            break

        layer_idx, proj_name = best_key
        sparsity_levels[layer_idx][proj_name] = best_new_sparsity

        # Recompute global sparsity
        weighted_sum = 0.0
        for li in range(num_layers):
            for p in PROJ_NAMES:
                weighted_sum += sparsity_levels[li][p] * param_weights[p]
        current_global_sparsity = weighted_sum / total_weight

    # Convert sparsity levels to thresholds
    thresholds = {}
    for layer_idx in range(num_layers):
        thresholds[layer_idx] = {}
        for proj_name in PROJ_NAMES:
            key = (layer_idx, proj_name)
            s = sparsity_levels[layer_idx][proj_name]
            if s > 0 and key in histograms:
                hist = histograms[key]
                t = compute_threshold_from_histogram(
                    hist["counts"], hist["edges"], s
                )
                thresholds[layer_idx][proj_name] = t
            else:
                thresholds[layer_idx][proj_name] = 0.0

    return thresholds


def _estimate_pruning_error(hist_data: dict, old_sparsity: float, new_sparsity: float) -> float:
    """
    Estimate the L2 error from pruning activations between old and new sparsity levels.

    Uses the histogram to approximate: sum of squared magnitudes in the
    [old_threshold, new_threshold] band.
    """
    counts = hist_data["counts"].float()
    edges = hist_data["edges"]
    total = counts.sum()

    if total == 0:
        return 0.0

    # CDF
    cdf = counts.cumsum(0) / total

    # Find bins corresponding to old and new thresholds
    old_bin = torch.searchsorted(cdf, old_sparsity).item()
    new_bin = torch.searchsorted(cdf, new_sparsity).item()
    old_bin = min(old_bin, len(counts) - 1)
    new_bin = min(new_bin, len(counts) - 1)

    # Sum squared magnitudes in the pruned band
    # Each bin center represents the typical magnitude of activations in that bin
    error = 0.0
    for b in range(old_bin, new_bin + 1):
        bin_center = (edges[b] + edges[b + 1]) / 2.0
        bin_count = counts[b].item()
        error += (bin_center ** 2) * bin_count

    return error


def save_thresholds(thresholds: dict, path: str):
    """Save thresholds dict to JSON."""
    # Convert int keys to strings for JSON
    serializable = {
        str(k): v for k, v in thresholds.items()
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[TEAL] Saved thresholds to {path}")


def load_thresholds(path: str) -> dict:
    """Load thresholds dict from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}
