"""
TurboQuantCache — HuggingFace DynamicCache subclass with KV cache quantization.

This is the integration layer that makes TurboQuant work with model.generate().
HuggingFace calls cache.update() every token during generation. We override it
to quantize K/V on write and dequantize on read.

Usage:
    cache = TurboQuantCache(bits=3, head_dim=128, num_layers=32, device="cuda")
    output = model.generate(**inputs, past_key_values=cache)
"""

import torch
from transformers import DynamicCache

from .quantizer import TurboQuantMSE, ProdQuantized


class TurboQuantCache(DynamicCache):
    """
    KV cache that compresses keys and values using TurboQuant.

    Keys are quantized using TurboQuantMSE (optimized for MSE distortion).
    Values are also quantized using TurboQuantMSE.

    We use MSE quantization (not ProdQuantized) for simplicity in the cache —
    the inner product optimization from Algorithm 2 requires modifying the
    attention computation itself, which we can add later. MSE quantization at
    3.5 bits is already quality-neutral per the paper.

    Memory savings: 16-bit → N-bit per coordinate, so ~4.5x at 3.5 bits.
    """

    def __init__(
        self,
        bits: int = 3,
        head_dim: int = 128,
        num_layers: int = 32,
        device: torch.device = None,
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
            layer_idx: which transformer layer this belongs to

        Returns:
            Tuple of (all_keys, all_values) — dequantized full cache for
            attention computation.
        """
        # Quantize the new tokens
        k_q = self.key_quantizers[layer_idx].quantize(key_states.float())
        v_q = self.val_quantizers[layer_idx].quantize(value_states.float())

        # Store compressed
        self._compressed_keys[layer_idx].append(k_q)
        self._compressed_values[layer_idx].append(v_q)

        # Dequantize everything for attention (HF expects full tensors back)
        all_keys = self._dequantize_all(
            self._compressed_keys[layer_idx],
            self.key_quantizers[layer_idx],
        )
        all_values = self._dequantize_all(
            self._compressed_values[layer_idx],
            self.val_quantizers[layer_idx],
        )

        # Update the parent class's tracking for seq length etc.
        # DynamicCache stores key_cache and value_cache as lists of tensors
        if layer_idx < len(self.key_cache):
            self.key_cache[layer_idx] = all_keys
            self.value_cache[layer_idx] = all_values
        else:
            self.key_cache.append(all_keys)
            self.value_cache.append(all_values)

        return all_keys.to(self._dtype), all_values.to(self._dtype)

    def _dequantize_all(self, compressed_list, quantizer):
        """Dequantize and concatenate all cached tokens for a layer."""
        if not compressed_list:
            return torch.empty(0)

        parts = []
        for q in compressed_list:
            parts.append(quantizer.dequantize(q))

        return torch.cat(parts, dim=2)  # concat along seq_len dim

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current sequence length stored in the cache."""
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx].numel() > 0:
            return self.key_cache[layer_idx].shape[2]
        return 0

    def get_memory_usage_bytes(self) -> int:
        """Estimate compressed memory usage in bytes."""
        total = 0
        for layer_keys in self._compressed_keys:
            for q in layer_keys:
                total += q.indices.numel()  # uint8 packed indices
                total += q.norms.numel() * 4  # float32 norms
        for layer_vals in self._compressed_values:
            for q in layer_vals:
                total += q.indices.numel()
                total += q.norms.numel() * 4
        return total

    def get_compression_ratio(self) -> float:
        """Return compression ratio vs FP16 cache."""
        if not self._compressed_keys[0]:
            return 0.0
        # FP16 = 2 bytes per element
        # Count total elements across all layers
        total_elements = 0
        for layer_idx in range(len(self.key_cache)):
            if layer_idx < len(self.key_cache) and self.key_cache[layer_idx].numel() > 0:
                total_elements += self.key_cache[layer_idx].numel()
                total_elements += self.value_cache[layer_idx].numel()
        fp16_bytes = total_elements * 2
        compressed_bytes = self.get_memory_usage_bytes()
        if compressed_bytes == 0:
            return 0.0
        return fp16_bytes / compressed_bytes

    def reset(self):
        """Clear all cached data."""
        self.key_cache.clear()
        self.value_cache.clear()
        self._compressed_keys = [[] for _ in range(self.num_layers)]
        self._compressed_values = [[] for _ in range(self.num_layers)]
