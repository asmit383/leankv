"""
TurboQuantCache — KV cache with real compressed storage.

Stores KV cache as quantized indices (uint8) + norms (float32) in GPU memory.
Only dequantizes transiently when attention needs the full tensors.

Memory layout per token per head:
  FP16:       128 dims × 2 bytes = 256 bytes
  3-bit TQ:   128 dims × 3 bits / 8 = 48 bytes (indices) + 4 bytes (norm) = 52 bytes
  Savings:    ~5x

Peak VRAM = model weights + compressed KV (all layers) + 1 layer dequantized (transient)
vs baseline = model weights + full FP16 KV (all layers)

Usage:
    cache = TurboQuantCache(bits=3, head_dim=128, num_layers=32, device="cuda")
    output = model.generate(**inputs, past_key_values=cache)
"""

import torch

from .quantizer import TurboQuantMSE, MSEQuantized


class TurboQuantCache:
    """
    KV cache with real compressed storage using TurboQuant MSE quantization.

    Stores only quantized indices (uint8) and norms (float32) persistently.
    Dequantizes on-the-fly when update() is called, returning full tensors
    for attention. These transient tensors are freed after each layer's
    forward pass completes.
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
        self._seq_lengths = [0] * num_layers

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

        # Compressed storage: single MSEQuantized per layer (concatenated indices + norms)
        # We merge chunks on arrival to avoid O(n) dequantization of separate chunks
        self._compressed_keys: list[MSEQuantized | None] = [None] * num_layers
        self._compressed_values: list[MSEQuantized | None] = [None] * num_layers

    # ── Core cache operations ───────────────────────────────────────────

    def _merge_quantized(self, existing: MSEQuantized | None, new: MSEQuantized) -> MSEQuantized:
        """Merge two quantized representations by concatenating along seq dim."""
        if existing is None:
            return new
        # indices: (..., packed_len) — concat along seq dimension
        # For packed indices, seq is embedded in the batch dims before packed_len
        # Shape is (batch, heads, seq, packed_len_per_token)
        merged_indices = torch.cat([existing.indices, new.indices], dim=2)
        merged_norms = torch.cat([existing.norms, new.norms], dim=2)
        return MSEQuantized(indices=merged_indices, norms=merged_norms, bits=new.bits)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        """
        Cache new K/V states with real compressed storage.

        1. Quantize new tokens → merge into compressed storage
        2. Dequantize full compressed cache → return for attention
        3. Returned tensors are transient — freed after attention uses them

        Args:
            key_states: (batch, num_kv_heads, seq_len, head_dim)
            value_states: (batch, num_kv_heads, seq_len, head_dim)
            layer_idx: which transformer layer

        Returns:
            (all_keys, all_values) — transient dequantized tensors for attention
        """
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        # Quantize new tokens
        k_q = self.key_quantizers[layer_idx].quantize(key_states.float())
        v_q = self.val_quantizers[layer_idx].quantize(value_states.float())

        # Merge into single compressed representation per layer
        self._compressed_keys[layer_idx] = self._merge_quantized(
            self._compressed_keys[layer_idx], k_q
        )
        self._compressed_values[layer_idx] = self._merge_quantized(
            self._compressed_values[layer_idx], v_q
        )
        self._seq_lengths[layer_idx] += key_states.shape[2]

        # Dequantize full cache for attention (transient)
        all_keys = self.key_quantizers[layer_idx].dequantize(
            self._compressed_keys[layer_idx]
        )
        all_values = self.val_quantizers[layer_idx].dequantize(
            self._compressed_values[layer_idx]
        )

        return all_keys.to(self._dtype), all_values.to(self._dtype)

    # ── Interface methods HF generation code calls ──────────────────────

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._seq_lengths):
            return self._seq_lengths[layer_idx]
        return 0

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0):
        kv_length = self.get_seq_length(layer_idx) + query_length
        kv_offset = 0
        return kv_length, kv_offset

    def get_max_cache_shape(self, *args, **kwargs):
        return None

    def get_max_length(self):
        return None

    def has_previous_state(self, layer_idx=None):
        if layer_idx is not None:
            return self._seq_lengths[layer_idx] > 0 if layer_idx < len(self._seq_lengths) else False
        return any(s > 0 for s in self._seq_lengths)

    def crop(self, max_length: int):
        pass  # not supported with compressed storage

    def batch_repeat_interleave(self, repeats: int):
        pass  # beam search not supported yet

    def batch_select_indices(self, indices: torch.Tensor):
        pass

    def reorder_cache(self, beam_idx: torch.LongTensor):
        pass

    def reset(self):
        self._compressed_keys = [None] * self.num_layers
        self._compressed_values = [None] * self.num_layers
        self._seq_lengths = [0] * self.num_layers
        self._seen_tokens = 0

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def seen_tokens(self):
        return self._seen_tokens

    @property
    def key_cache(self):
        # Return list of empty tensors — real data is in compressed storage
        # HF checks len(key_cache) to determine number of cached layers
        return [torch.empty(0, device=self._device)] * len([s for s in self._seq_lengths if s > 0])

    @property
    def value_cache(self):
        return [torch.empty(0, device=self._device)] * len([s for s in self._seq_lengths if s > 0])

    @property
    def is_initialized(self):
        return any(s > 0 for s in self._seq_lengths)

    @property
    def is_sliding(self):
        return [False] * self.num_layers

    @property
    def max_batch_size(self):
        for q in self._compressed_keys:
            if q is not None:
                return q.norms.shape[0]
        return 0

    @property
    def max_cache_len(self):
        return max(self._seq_lengths) if self._seq_lengths else 0

    # ── Container protocol ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len([s for s in self._seq_lengths if s > 0])

    def __getitem__(self, layer_idx: int):
        if self._seq_lengths[layer_idx] > 0 and self._compressed_keys[layer_idx] is not None:
            keys = self.key_quantizers[layer_idx].dequantize(
                self._compressed_keys[layer_idx]
            )
            values = self.val_quantizers[layer_idx].dequantize(
                self._compressed_values[layer_idx]
            )
            return keys.to(self._dtype), values.to(self._dtype)
        raise IndexError(f"Layer {layer_idx} not in cache")

    def __iter__(self):
        for i in range(self.num_layers):
            if self._seq_lengths[i] > 0:
                yield self[i]

    def __bool__(self):
        return True

    def to_legacy_cache(self):
        result = []
        for i in range(self.num_layers):
            if self._seq_lengths[i] > 0:
                k, v = self[i]
                result.append((k, v))
        return tuple(result)

    # ── Diagnostics ─────────────────────────────────────────────────────

    def get_compressed_memory_bytes(self) -> int:
        """Actual compressed memory usage on GPU in bytes."""
        total = 0
        for q in self._compressed_keys:
            if q is not None:
                total += q.indices.numel() * q.indices.element_size()
                total += q.norms.numel() * q.norms.element_size()
        for q in self._compressed_values:
            if q is not None:
                total += q.indices.numel() * q.indices.element_size()
                total += q.norms.numel() * q.norms.element_size()
        return total

    def get_fp16_equivalent_bytes(self) -> int:
        """What the same cache would cost in FP16."""
        total_elements = 0
        for i, s in enumerate(self._seq_lengths):
            if s > 0 and self._compressed_keys[i] is not None:
                norms = self._compressed_keys[i].norms
                batch = norms.shape[0]
                heads = norms.shape[1] if norms.dim() > 1 else 1
                total_elements += batch * heads * s * self.head_dim * 2  # K + V
        return total_elements * 2  # FP16 = 2 bytes

    def get_compression_ratio(self) -> float:
        compressed = self.get_compressed_memory_bytes()
        if compressed == 0:
            return 0.0
        fp16 = self.get_fp16_equivalent_bytes()
        return fp16 / compressed

    def print_memory_report(self):
        compressed_mb = self.get_compressed_memory_bytes() / 1e6
        fp16_mb = self.get_fp16_equivalent_bytes() / 1e6
        ratio = self.get_compression_ratio()
        seq_len = max(self._seq_lengths) if self._seq_lengths else 0
        print(f"[TurboQuantCache] seq_len={seq_len}, "
              f"compressed={compressed_mb:.1f} MB, "
              f"FP16 equivalent={fp16_mb:.1f} MB, "
              f"ratio={ratio:.1f}x")
