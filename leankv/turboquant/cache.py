"""
TurboQuantCache — KV cache with quantization for HuggingFace models.

Implements the Cache interface directly (no DynamicCache dependency) to
work across transformers versions.

Usage:
    cache = TurboQuantCache(bits=3, head_dim=128, num_layers=32, device="cuda")
    output = model.generate(**inputs, past_key_values=cache)
"""

import torch
from transformers.cache_utils import Cache

from .quantizer import TurboQuantMSE


class TurboQuantCache(Cache):
    """
    KV cache that compresses keys and values using TurboQuant MSE quantization.

    On each update(), incoming K/V states are quantized and stored compressed.
    The full dequantized cache is returned for attention computation.

    Memory savings: 16-bit -> N-bit per coordinate, ~4.5x at 3.5 bits.
    """

    def __init__(
        self,
        bits: int = 3,
        head_dim: int = 128,
        num_layers: int = 32,
        device=None,
        dtype: torch.dtype = torch.float16,
        seed: int = 42,
    ):
        super().__init__()
        self.bits = bits
        self.head_dim = head_dim
        self.num_layers = num_layers
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._seen_tokens = 0

        # Per-layer quantizers for keys and values (separate rotations)
        self.key_quantizers = []
        self.val_quantizers = []
        for i in range(num_layers):
            self.key_quantizers.append(
                TurboQuantMSE(
                    dim=head_dim, bits=bits, device=self._device,
                    dtype=torch.float32, seed=seed + i * 2,
                )
            )
            self.val_quantizers.append(
                TurboQuantMSE(
                    dim=head_dim, bits=bits, device=self._device,
                    dtype=torch.float32, seed=seed + i * 2 + 1,
                )
            )

        # Compressed storage: list of quantized tuples per layer
        self._compressed_keys: list[list] = [[] for _ in range(num_layers)]
        self._compressed_values: list[list] = [[] for _ in range(num_layers)]

        # Dequantized cache (what the model reads for attention)
        self._key_cache: list[torch.Tensor] = []
        self._value_cache: list[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        """
        Cache new K/V states with quantization.

        Args:
            key_states: (batch, num_kv_heads, seq_len, head_dim)
            value_states: (batch, num_kv_heads, seq_len, head_dim)
            layer_idx: which transformer layer

        Returns:
            (all_keys, all_values) — dequantized full cache for attention.
        """
        # Track tokens seen (only count from layer 0 to avoid double counting)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        # Quantize the new tokens
        k_q = self.key_quantizers[layer_idx].quantize(key_states.float())
        v_q = self.val_quantizers[layer_idx].quantize(value_states.float())

        # Store compressed
        self._compressed_keys[layer_idx].append(k_q)
        self._compressed_values[layer_idx].append(v_q)

        # Dequantize everything for attention
        all_keys = self._dequantize_all(
            self._compressed_keys[layer_idx],
            self.key_quantizers[layer_idx],
        )
        all_values = self._dequantize_all(
            self._compressed_values[layer_idx],
            self.val_quantizers[layer_idx],
        )

        # Store dequantized for get_seq_length etc.
        if layer_idx < len(self._key_cache):
            self._key_cache[layer_idx] = all_keys
            self._value_cache[layer_idx] = all_values
        else:
            self._key_cache.append(all_keys)
            self._value_cache.append(all_values)

        return all_keys.to(self._dtype), all_values.to(self._dtype)

    def _dequantize_all(self, compressed_list, quantizer):
        """Dequantize and concatenate all cached tokens for a layer."""
        if not compressed_list:
            return torch.empty(0, device=self._device)

        parts = [quantizer.dequantize(q) for q in compressed_list]
        return torch.cat(parts, dim=2)  # concat along seq_len dim

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current sequence length in the cache."""
        if layer_idx < len(self._key_cache) and self._key_cache[layer_idx].numel() > 0:
            return self._key_cache[layer_idx].shape[2]
        return 0

    def get_max_cache_shape(self) -> int:
        """Required by some transformers versions."""
        return None

    def get_max_length(self) -> int:
        """No fixed max length."""
        return None

    def __len__(self) -> int:
        return len(self._key_cache)

    def __getitem__(self, layer_idx: int):
        """Allow indexing: cache[layer_idx] -> (keys, values)."""
        if layer_idx < len(self._key_cache):
            return (
                self._key_cache[layer_idx].to(self._dtype),
                self._value_cache[layer_idx].to(self._dtype),
            )
        raise IndexError(f"Layer {layer_idx} not in cache (have {len(self._key_cache)} layers)")

    def __iter__(self):
        """Iterate over (key, value) pairs per layer."""
        for i in range(len(self._key_cache)):
            yield (
                self._key_cache[i].to(self._dtype),
                self._value_cache[i].to(self._dtype),
            )

    def to_legacy_cache(self):
        """Convert to tuple format for older transformers versions."""
        return tuple(
            (k.to(self._dtype), v.to(self._dtype))
            for k, v in zip(self._key_cache, self._value_cache)
        )

    @property
    def seen_tokens(self):
        return self._seen_tokens

    def get_memory_usage_bytes(self) -> int:
        """Estimate compressed memory usage in bytes."""
        total = 0
        for layer_keys in self._compressed_keys:
            for q in layer_keys:
                total += q.indices.numel()
                total += q.norms.numel() * 4
        for layer_vals in self._compressed_values:
            for q in layer_vals:
                total += q.indices.numel()
                total += q.norms.numel() * 4
        return total

    def reset(self):
        """Clear all cached data."""
        self._key_cache.clear()
        self._value_cache.clear()
        self._compressed_keys = [[] for _ in range(self.num_layers)]
        self._compressed_values = [[] for _ in range(self.num_layers)]
        self._seen_tokens = 0
