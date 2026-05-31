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

try:
    from .kernels.sparse_gemv import sparse_gemv, prepare_weight_for_sparse_gemv
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

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
    use_triton: bool = False,
):
    """
    Apply TEAL activation sparsity to a HuggingFace model.

    Exactly one of sparsity, thresholds, or thresholds_path must be provided.

    Args:
        model: HuggingFace CausalLM (LlamaForCausalLM, MistralForCausalLM, etc.)
        sparsity: uniform sparsity level (0.0 to 1.0) applied to all projections.
        thresholds: dict mapping {layer_idx: {proj_name: threshold_value}}.
        thresholds_path: path to JSON file containing thresholds dict.
        calibration_data: optional list of input tensors for threshold estimation.
        use_triton: if True, replace nn.Linear forward with Triton sparse GEMV
            kernel for actual wall-clock speedup. Requires triton>=2.2, CUDA 12+,
            and FP16 inputs. Falls back to hooks if Triton not available.

    Returns:
        dict of {(layer_idx, proj_name): SparsifyFn} for diagnostics.
    """
    if sum(x is not None for x in [sparsity, thresholds, thresholds_path]) != 1:
        raise ValueError(
            "Exactly one of sparsity, thresholds, or thresholds_path must be provided."
        )

    if use_triton and not HAS_TRITON:
        print("[TEAL] Warning: Triton not available, falling back to hook-based sparsification.")
        use_triton = False

    if thresholds_path is not None:
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)
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
                    with torch.no_grad():
                        acts = _collect_input_activations(
                            model, proj, calibration_data
                        )
                        t = compute_threshold_for_sparsity(acts, sparsity)
                else:
                    t = sparsity * 0.1
            else:
                t = 0.0

            sparsifier = SparsifyFn(threshold=t, enabled=(t > 0))
            sparsifiers[(layer_idx, proj_name)] = sparsifier

            if use_triton and t > 0:
                # Replace the Linear's forward with Triton sparse GEMV
                _patch_linear_with_triton(proj, t, sparsity or 0.4)
            else:
                # Hook-based: sparsify input before the Linear's forward
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

    if use_triton:
        print(f"[TEAL] Using Triton sparse GEMV kernels (actual speedup)")
    else:
        print(f"[TEAL] Using hook-based sparsification (correctness only)")

    return sparsifiers


def _patch_linear_with_triton(linear: nn.Linear, threshold: float, sparsity: float):
    """
    Replace a nn.Linear's forward method with Triton sparse GEMV.

    Converts weight to column-major layout and replaces forward()
    to use the fused sparse kernel during decode (seq_len=1).
    Falls back to dense matmul during prefill (seq_len>1).
    """
    # Convert weight to column-major (one-time cost)
    with torch.no_grad():
        linear.weight.data = prepare_weight_for_sparse_gemv(linear.weight.data)

    # Compute sparsity bin for autotuning (0-10 scale)
    sparsity_bin = int(sparsity * 10)

    original_forward = linear.forward

    def sparse_forward(x):
        if x.shape[-2] == 1:
            # Decode mode
            batch_size = x.shape[0] if x.dim() == 3 else 1
            if batch_size == 1:
                # B=1: use Triton sparse GEMV (fastest path)
                x_flat = x.reshape(1, 1, x.shape[-1])
                out = sparse_gemv(x_flat, linear.weight, threshold, sparsity_bin)
                out = out.reshape(x.shape[:-1] + (linear.weight.shape[0],))
            else:
                # B>1: shared column mask + smaller dense matmul
                # Find columns where ALL batch items are near-zero → skip those
                x_2d = x.reshape(batch_size, x.shape[-1])  # (B, D)
                col_max = x_2d.abs().max(dim=0).values  # (D,) max magnitude per column
                keep_mask = col_max > threshold  # columns to keep
                n_keep = keep_mask.sum().item()

                if n_keep < x_2d.shape[-1] * 0.95:
                    # Enough sparsity to be worth it — do smaller dense matmul
                    x_sparse = x_2d[:, keep_mask]  # (B, n_keep)
                    w_sparse = linear.weight[:, keep_mask]  # (out_dim, n_keep)
                    out = torch.matmul(x_sparse, w_sparse.t())  # (B, out_dim)
                    out = out.reshape(x.shape[:-1] + (linear.weight.shape[0],))
                else:
                    # Not enough sparsity — full dense matmul
                    out = torch.nn.functional.linear(x, linear.weight, None)
            if linear.bias is not None:
                out = out + linear.bias
            return out
        else:
            # Prefill: dense matmul
            return torch.nn.functional.linear(x, linear.weight, linear.bias)

    linear.forward = sparse_forward
    linear._teal_patched = True


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
