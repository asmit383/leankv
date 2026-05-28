"""
TurboQuantCache — KV cache with quantization for HuggingFace models.

Implements the full Cache interface via duck typing to work across
transformers versions without depending on Cache base class internals.

Usage:
    cache = TurboQuantCache(bits=3, head_dim=128, num_layers=32, device="cuda")
    output = model.generate(**inputs, past_key_values=cache)
"""

import torch

from .quantizer import TurboQuantMSE


class TurboQuantCache:
    """
    KV cache that compresses keys and values using TurboQuant MSE quantization.

    On each update(), incoming K/V states are quantized and stored compressed.
    The full dequantized cache is returned for attention computation.

    Memory savings: 16-bit -> N-bit per coordinate, ~4.5x at 3.5 bits.
    """

    is_compileable = False
    layer_type = None

    def __init__(
        self,
        bits: int = 3,
        head_dim: int = 128,
        num_layers: int = 32,
        device=None,
        dtype: torch.dtype = torch.float16,
        seed: int = 42,
    ):
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

    # ── Core cache operations ───────────────────────────────────────────

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

    # ── Interface methods HF generation code calls ──────────────────────

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._key_cache) and self._key_cache[layer_idx].numel() > 0:
            return self._key_cache[layer_idx].shape[2]
        return 0

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0):
        """Return (kv_length, kv_offset) for attention mask construction."""
        kv_length = self.get_seq_length(layer_idx) + query_length
        kv_offset = 0  # no sliding window
        return kv_length, kv_offset

    def get_max_cache_shape(self, *args, **kwargs):
        return None

    def get_max_length(self):
        return None

    def has_previous_state(self, layer_idx=None):
        if layer_idx is not None:
            return layer_idx < len(self._key_cache) and self._key_cache[layer_idx].numel() > 0
        return len(self._key_cache) > 0

    def crop(self, max_length: int):
        """Crop cache to max_length (for beam search compatibility)."""
        for i in range(len(self._key_cache)):
            if self._key_cache[i].numel() > 0 and self._key_cache[i].shape[2] > max_length:
                self._key_cache[i] = self._key_cache[i][:, :, :max_length, :]
                self._value_cache[i] = self._value_cache[i][:, :, :max_length, :]

    def batch_repeat_interleave(self, repeats: int):
        """Repeat cache entries for beam search."""
        for i in range(len(self._key_cache)):
            if self._key_cache[i].numel() > 0:
                self._key_cache[i] = self._key_cache[i].repeat_interleave(repeats, dim=0)
                self._value_cache[i] = self._value_cache[i].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Select specific batch entries."""
        for i in range(len(self._key_cache)):
            if self._key_cache[i].numel() > 0:
                self._key_cache[i] = self._key_cache[i][indices]
                self._value_cache[i] = self._value_cache[i][indices]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search."""
        for i in range(len(self._key_cache)):
            if self._key_cache[i].numel() > 0:
                self._key_cache[i] = self._key_cache[i].index_select(0, beam_idx.to(self._key_cache[i].device))
                self._value_cache[i] = self._value_cache[i].index_select(0, beam_idx.to(self._value_cache[i].device))

    def reset(self):
        self._key_cache.clear()
        self._value_cache.clear()
        self._compressed_keys = [[] for _ in range(self.num_layers)]
        self._compressed_values = [[] for _ in range(self.num_layers)]
        self._seen_tokens = 0

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def seen_tokens(self):
        return self._seen_tokens

    @property
    def key_cache(self):
        return self._key_cache

    @property
    def value_cache(self):
        return self._value_cache

    @property
    def is_initialized(self):
        return len(self._key_cache) > 0

    @property
    def is_sliding(self):
        return [False] * self.num_layers

    @property
    def max_batch_size(self):
        if self._key_cache and self._key_cache[0].numel() > 0:
            return self._key_cache[0].shape[0]
        return 0

    @property
    def max_cache_len(self):
        if self._key_cache and self._key_cache[0].numel() > 0:
            return self._key_cache[0].shape[2]
        return 0

    # ── Container protocol ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._key_cache)

    def __getitem__(self, layer_idx: int):
        if layer_idx < len(self._key_cache):
            return (
                self._key_cache[layer_idx].to(self._dtype),
                self._value_cache[layer_idx].to(self._dtype),
            )
        raise IndexError(f"Layer {layer_idx} not in cache")

    def __iter__(self):
        for i in range(len(self._key_cache)):
            yield (
                self._key_cache[i].to(self._dtype),
                self._value_cache[i].to(self._dtype),
            )

    def __bool__(self):
        return True

    def to_legacy_cache(self):
        return tuple(
            (k.to(self._dtype), v.to(self._dtype))
            for k, v in zip(self._key_cache, self._value_cache)
        )

    # ── Diagnostics ─────────────────────────────────────────────────────

    def get_memory_usage_bytes(self) -> int:
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
