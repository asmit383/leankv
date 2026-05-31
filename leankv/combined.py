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

try:
    from .turboquant.paged_cache import PagedKVCache
    HAS_PAGED = True
except ImportError:
    HAS_PAGED = False


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
    use_paged_cache: bool = False,
    page_size: int = 16,
    max_pages: int = 2048,
):
    """
    Apply TEAL and/or TurboQuant to a HuggingFace model.

    Args:
        model: HuggingFace CausalLM.
        teal_sparsity: uniform TEAL sparsity (0.0-1.0).
        teal_thresholds_path: path to calibrated TEAL thresholds JSON.
        turboquant_bits: KV cache quantization bits (2-4). None = no TQ.
        use_paged_cache: use FlashInfer paged attention (requires flashinfer).
        page_size: tokens per page (default 16).
        max_pages: maximum number of pages in the pool.

    Returns:
        (model, cache) tuple. cache is None if no cache optimization enabled.
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

    cache = None
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    num_q_heads = config.num_attention_heads

    if use_paged_cache:
        if not HAS_PAGED:
            raise RuntimeError(
                "FlashInfer not installed. Run: pip install flashinfer"
            )
        cache = PagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
            head_dim=head_dim,
            page_size=page_size,
            max_pages=max_pages,
            device=model.device,
            dtype=torch.float16,
        )
        pool_mb = cache.get_total_pool_mb()
        print(f"[LeanKV] PagedKVCache enabled (pages={max_pages}, "
              f"page_size={page_size}, pool={pool_mb:.0f} MB)")
    elif turboquant_bits is not None:
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
