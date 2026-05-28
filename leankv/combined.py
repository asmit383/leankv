"""
Unified API for applying TEAL + TurboQuant to a HuggingFace model.

Usage:
    from leankv.combined import optimize_model, load_model

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.3")
    model, cache = optimize_model(
        model,
        teal_sparsity=0.4,            # or teal_thresholds_path="thresholds/..."
        turboquant_bits=3,
    )
    output = model.generate(**inputs, past_key_values=cache)
"""

import torch
from dataclasses import dataclass
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .teal.patching import apply_teal, remove_teal
from .turboquant.cache import TurboQuantCache


@dataclass
class TEALConfig:
    sparsity: Optional[float] = None
    thresholds_path: Optional[str] = None


@dataclass
class TurboQuantConfig:
    bits: int = 3
    head_dim: int = 128
    num_layers: int = 32


def load_model(model_name: str, dtype=torch.float16, device_map="cuda"):
    """Load a HuggingFace model and tokenizer."""
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    print(f"Model loaded. Device: {model.device}")
    return model, tokenizer


def optimize_model(
    model,
    teal_sparsity: Optional[float] = None,
    teal_thresholds_path: Optional[str] = None,
    turboquant_bits: Optional[int] = None,
):
    """
    Apply TEAL and/or TurboQuant to a HuggingFace model.

    Args:
        model: HuggingFace CausalLM.
        teal_sparsity: uniform TEAL sparsity (0.0-1.0). Mutually exclusive
            with teal_thresholds_path.
        teal_thresholds_path: path to calibrated TEAL thresholds JSON.
        turboquant_bits: KV cache quantization bits (2-4). None = no TQ.

    Returns:
        (model, cache) tuple. cache is None if TurboQuant not enabled.
    """
    # Apply TEAL
    if teal_sparsity is not None or teal_thresholds_path is not None:
        apply_teal(
            model,
            sparsity=teal_sparsity,
            thresholds_path=teal_thresholds_path,
        )
        label = teal_thresholds_path or f"{teal_sparsity:.0%}"
        print(f"[LeanKV] TEAL enabled (sparsity={label})")

    # Create TurboQuant cache
    cache = None
    if turboquant_bits is not None:
        config = model.config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        num_layers = config.num_hidden_layers

        cache = TurboQuantCache(
            bits=turboquant_bits,
            head_dim=head_dim,
            num_layers=num_layers,
            device=model.device,
            dtype=torch.float16,
        )
        print(f"[LeanKV] TurboQuant enabled (bits={turboquant_bits}, head_dim={head_dim})")

    return model, cache


def reset_optimizations(model):
    """Remove all optimizations from a model."""
    remove_teal(model)
    print("[LeanKV] All optimizations removed.")
