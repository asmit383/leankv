"""
Monkey-patch HuggingFace LLM models with TEAL activation sparsity.

Registers forward pre-hooks on all 7 nn.Linear projections per transformer
block (W_q, W_k, W_v, W_o, W_gate, W_up, W_down) that zero out sub-threshold
activations before the matmul.

Usage:
    from leankv.teal.patching import apply_teal, remove_teal

    apply_teal(model, sparsity=0.4)           # uniform 40% sparsity
    apply_teal(model, thresholds=thresholds)  # calibrated per-layer thresholds
    remove_teal(model)                        # cleanly remove all hooks
"""

import json
import os
import torch
import torch.nn as nn
from typing import Optional

from .sparse_fns import SparsifyFn, compute_threshold_for_sparsity

# Projection names in the order TEAL processes them
PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Maps projection names to their parent module attribute
ATTN_PROJS = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_PROJS = {"gate_proj", "up_proj", "down_proj"}


def _get_layers(model):
    """Extract transformer layers from a HuggingFace model."""
    # Works for Llama, Mistral, and other HF models with this structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(
        f"Cannot find transformer layers in model of type {type(model)}. "
        "Expected model.model.layers attribute."
    )


def _get_projection(layer, proj_name: str) -> nn.Linear:
    """Get a specific projection module from a transformer layer."""
    if proj_name in ATTN_PROJS:
        return getattr(layer.self_attn, proj_name)
    else:
        return getattr(layer.mlp, proj_name)


def apply_teal(
    model,
    sparsity: Optional[float] = None,
    thresholds: Optional[dict] = None,
    thresholds_path: Optional[str] = None,
    calibration_data: Optional[list[torch.Tensor]] = None,
):
    """
    Apply TEAL activation sparsity to a HuggingFace model.

    Exactly one of sparsity, thresholds, or thresholds_path must be provided.

    Args:
        model: HuggingFace CausalLM (LlamaForCausalLM, MistralForCausalLM, etc.)
        sparsity: uniform sparsity level (0.0 to 1.0) applied to all projections.
            Uses a fixed threshold estimated from activation statistics.
        thresholds: dict mapping {layer_idx: {proj_name: threshold_value}}.
            Produced by greedy_opt.py calibration.
        thresholds_path: path to JSON file containing thresholds dict.
        calibration_data: optional list of input tensors for estimating thresholds
            when using uniform sparsity. If None, uses a heuristic threshold.

    Returns:
        dict of {(layer_idx, proj_name): SparsifyFn} for diagnostics.
    """
    if sum(x is not None for x in [sparsity, thresholds, thresholds_path]) != 1:
        raise ValueError(
            "Exactly one of sparsity, thresholds, or thresholds_path must be provided."
        )

    if thresholds_path is not None:
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)
        # Keys in JSON are strings, convert to int
        thresholds = {int(k): v for k, v in thresholds.items()}

    layers = _get_layers(model)
    sparsifiers = {}
    hooks = []

    for layer_idx, layer in enumerate(layers):
        for proj_name in PROJ_NAMES:
            proj = _get_projection(layer, proj_name)

            # Determine threshold for this projection
            if thresholds is not None:
                t = thresholds.get(layer_idx, {}).get(proj_name, 0.0)
            elif sparsity is not None and sparsity > 0:
                if calibration_data is not None:
                    # Estimate threshold from calibration activations
                    with torch.no_grad():
                        acts = _collect_input_activations(
                            model, proj, calibration_data
                        )
                        t = compute_threshold_for_sparsity(acts, sparsity)
                else:
                    # Heuristic: use sparsity as a rough threshold scale
                    # This will be replaced by proper calibration in Step 4
                    t = sparsity * 0.1
            else:
                t = 0.0

            sparsifier = SparsifyFn(threshold=t, enabled=(t > 0))
            sparsifiers[(layer_idx, proj_name)] = sparsifier

            # Register the pre-hook: sparsify input before the Linear's forward
            def make_hook(sf):
                def hook_fn(module, args):
                    x = args[0]
                    x_sparse = sf(x)
                    return (x_sparse,) + args[1:]
                return hook_fn

            handle = proj.register_forward_pre_hook(make_hook(sparsifier))
            hooks.append(handle)

    # Store hooks on the model for later removal
    if not hasattr(model, "_teal_hooks"):
        model._teal_hooks = []
    model._teal_hooks.extend(hooks)
    model._teal_sparsifiers = sparsifiers

    return sparsifiers


def remove_teal(model):
    """Remove all TEAL hooks from a model."""
    if hasattr(model, "_teal_hooks"):
        for handle in model._teal_hooks:
            handle.remove()
        model._teal_hooks.clear()
    if hasattr(model, "_teal_sparsifiers"):
        del model._teal_sparsifiers


def get_sparsity_report(model) -> dict:
    """
    Get the last-observed sparsity for each projection in the model.

    Returns:
        dict mapping (layer_idx, proj_name) to sparsity fraction.
    """
    if not hasattr(model, "_teal_sparsifiers"):
        return {}
    return {
        key: sf.last_sparsity
        for key, sf in model._teal_sparsifiers.items()
    }


def print_sparsity_report(model):
    """Print a summary of per-layer sparsity levels."""
    report = get_sparsity_report(model)
    if not report:
        print("No TEAL sparsifiers found on model.")
        return

    # Group by layer
    layers = {}
    for (layer_idx, proj_name), sparsity in report.items():
        if layer_idx not in layers:
            layers[layer_idx] = {}
        layers[layer_idx][proj_name] = sparsity

    print(f"{'Layer':<8}", end="")
    for proj in PROJ_NAMES:
        print(f"{proj:<12}", end="")
    print(f"{'Avg':<8}")
    print("-" * (8 + 12 * 7 + 8))

    for layer_idx in sorted(layers.keys()):
        print(f"{layer_idx:<8}", end="")
        vals = []
        for proj in PROJ_NAMES:
            s = layers[layer_idx].get(proj, 0.0)
            vals.append(s)
            print(f"{s:>10.1%}  ", end="")
        print(f"{sum(vals)/len(vals):>6.1%}")


def _collect_input_activations(model, target_module, data_samples):
    """Collect input activations to a specific module from data samples."""
    activations = []

    def hook_fn(module, args):
        activations.append(args[0].detach().cpu())

    handle = target_module.register_forward_pre_hook(hook_fn)
    try:
        for sample in data_samples[:10]:  # limit samples for speed
            with torch.no_grad():
                model(sample.to(model.device))
    finally:
        handle.remove()

    return torch.cat(activations, dim=0)
