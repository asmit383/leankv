"""
TurboQuantCache — Hybrid KV cache with compressed cold storage + FP16 hot buffer.

Memory layout:
  - Hot buffer: last N tokens in FP16 (fast, no dequant needed)
  - Cold storage: older tokens as quantized indices (uint8) + norms (float32)
  - On attention: dequantize cold + concat hot = full KV for attention

This gives both VRAM savings AND near-baseline speed:
  - New tokens: quantize→dequantize (1 token, cheap) → append to hot buffer
  - When hot buffer exceeds N: flush oldest to cold compressed storage
  - Attention: dequantize cold (done once per layer) + hot buffer

Peak VRAM = model + compressed cold KV + hot buffer (N tokens FP16) + 1 layer dequantized cold
vs baseline = model + full FP16 KV (all layers, all tokens)

Usage:
    cache = TurboQuantCache(bits=3, head_dim=128, num_layers=32, device="cuda")
    output = model.generate(**inputs, past_key_values=cache)
"""

import torch

from .quantizer import TurboQuantMSE, MSEQuantized


class TurboQuantCache:
    """
    Hybrid KV cache: compressed cold storage + FP16 hot buffer.
    """

    is_compileable = False
    layer_type = None

    def __init__(
        self,
        bits: int = 3,
        head_dim: int = 128,
        num_layers: int = 32,
        hot_buffer_size: int = 64,
        device=None,
        dtype: torch.dtype = torch.float16,
        seed: int = 42,
    ):
        self.bits = bits
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.hot_buffer_size = hot_buffer_size
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._seen_tokens = 0
        self._seq_lengths = [0] * num_layers

        # Per-layer quantizers
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

        # Cold storage: compressed (old tokens)
        self._cold_keys: list[MSEQuantized | None] = [None] * num_layers
        self._cold_values: list[MSEQuantized | None] = [None] * num_layers
        self._cold_lengths: list[int] = [0] * num_layers

        # Hot buffer: FP16 (recent tokens)
        self._hot_keys: list[torch.Tensor | None] = [None] * num_layers
        self._hot_values: list[torch.Tensor | None] = [None] * num_layers

    # ── Core cache operations ───────────────────────────────────────────

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        """
        Cache new K/V states using hybrid hot/cold storage.

        New tokens go to the FP16 hot buffer. When hot buffer exceeds
        hot_buffer_size, oldest tokens are flushed to compressed cold storage.
        Returns full cache (cold dequantized + hot) for attention.
        """
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        new_seq = key_states.shape[2]
        self._seq_lengths[layer_idx] += new_seq

        # Append new tokens to hot buffer
        k_new = key_states.to(self._dtype)
        v_new = value_states.to(self._dtype)

        if self._hot_keys[layer_idx] is None:
            self._hot_keys[layer_idx] = k_new
            self._hot_values[layer_idx] = v_new
        else:
            self._hot_keys[layer_idx] = torch.cat(
                [self._hot_keys[layer_idx], k_new], dim=2
            )
            self._hot_values[layer_idx] = torch.cat(
                [self._hot_values[layer_idx], v_new], dim=2
            )

        # Flush to cold if hot buffer exceeds limit
        hot_len = self._hot_keys[layer_idx].shape[2]
        if hot_len > self.hot_buffer_size:
            self._flush_to_cold(layer_idx)

        # Build full cache for attention: cold (dequantized) + hot
        all_keys, all_values = self._get_full_cache(layer_idx)

        return all_keys, all_values

    def _flush_to_cold(self, layer_idx: int):
        """Move tokens beyond hot_buffer_size from hot buffer to cold storage."""
        hot_k = self._hot_keys[layer_idx]
        hot_v = self._hot_values[layer_idx]
        hot_len = hot_k.shape[2]

        # Keep last hot_buffer_size tokens in hot, flush the rest to cold
        flush_len = hot_len - self.hot_buffer_size
        flush_k = hot_k[:, :, :flush_len, :]
        flush_v = hot_v[:, :, :flush_len, :]

        # Quantize flushed tokens
        k_q = self.key_quantizers[layer_idx].quantize(flush_k.float())
        v_q = self.val_quantizers[layer_idx].quantize(flush_v.float())

        # Merge into cold storage
        if self._cold_keys[layer_idx] is None:
            self._cold_keys[layer_idx] = k_q
            self._cold_values[layer_idx] = v_q
        else:
            self._cold_keys[layer_idx] = self._merge_quantized(
                self._cold_keys[layer_idx], k_q
            )
            self._cold_values[layer_idx] = self._merge_quantized(
                self._cold_values[layer_idx], v_q
            )
        self._cold_lengths[layer_idx] += flush_len

        # Trim hot buffer
        self._hot_keys[layer_idx] = hot_k[:, :, flush_len:, :].contiguous()
        self._hot_values[layer_idx] = hot_v[:, :, flush_len:, :].contiguous()

    def _merge_quantized(self, a: MSEQuantized, b: MSEQuantized) -> MSEQuantized:
        """Merge two quantized representations along seq dimension."""
        return MSEQuantized(
            indices=torch.cat([a.indices, b.indices], dim=2),
            norms=torch.cat([a.norms, b.norms], dim=2),
            bits=a.bits,
        )

    def _get_full_cache(self, layer_idx: int):
        """Dequantize cold + concat hot for full attention cache."""
        hot_k = self._hot_keys[layer_idx]
        hot_v = self._hot_values[layer_idx]

        if self._cold_keys[layer_idx] is not None:
            # Dequantize cold storage
            cold_k = self.key_quantizers[layer_idx].dequantize(
                self._cold_keys[layer_idx]
            ).to(self._dtype)
            cold_v = self.val_quantizers[layer_idx].dequantize(
                self._cold_values[layer_idx]
            ).to(self._dtype)
            # Concat: cold (old) + hot (recent)
            all_keys = torch.cat([cold_k, hot_k], dim=2)
            all_values = torch.cat([cold_v, hot_v], dim=2)
        else:
            all_keys = hot_k
            all_values = hot_v

        return all_keys, all_values

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
        pass

    def batch_repeat_interleave(self, repeats: int):
        pass

    def batch_select_indices(self, indices: torch.Tensor):
        pass

    def reorder_cache(self, beam_idx: torch.LongTensor):
        pass

    def reset(self):
        self._cold_keys = [None] * self.num_layers
        self._cold_values = [None] * self.num_layers
        self._cold_lengths = [0] * self.num_layers
        self._hot_keys = [None] * self.num_layers
        self._hot_values = [None] * self.num_layers
        self._seq_lengths = [0] * self.num_layers
        self._seen_tokens = 0

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def seen_tokens(self):
        return self._seen_tokens

    @property
    def key_cache(self):
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
        for k in self._hot_keys:
            if k is not None:
                return k.shape[0]
        return 0

    @property
    def max_cache_len(self):
        return max(self._seq_lengths) if self._seq_lengths else 0

    # ── Container protocol ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len([s for s in self._seq_lengths if s > 0])

    def __getitem__(self, layer_idx: int):
        if self._seq_lengths[layer_idx] > 0:
            return self._get_full_cache(layer_idx)
        raise IndexError(f"Layer {layer_idx} not in cache")

    def __iter__(self):
        for i in range(self.num_layers):
            if self._seq_lengths[i] > 0:
                yield self._get_full_cache(i)

    def __bool__(self):
        return True

    def to_legacy_cache(self):
        result = []
        for i in range(self.num_layers):
            if self._seq_lengths[i] > 0:
                result.append(self._get_full_cache(i))
        return tuple(result)

    # ── Diagnostics ─────────────────────────────────────────────────────

    def get_compressed_memory_bytes(self) -> int:
        """Memory used by cold compressed storage."""
        total = 0
        for q in self._cold_keys:
            if q is not None:
                total += q.indices.numel() * q.indices.element_size()
                total += q.norms.numel() * q.norms.element_size()
        for q in self._cold_values:
            if q is not None:
                total += q.indices.numel() * q.indices.element_size()
                total += q.norms.numel() * q.norms.element_size()
        return total

    def get_hot_memory_bytes(self) -> int:
        """Memory used by hot FP16 buffer."""
        total = 0
        for k in self._hot_keys:
            if k is not None:
                total += k.numel() * k.element_size()
        for v in self._hot_values:
            if v is not None:
                total += v.numel() * v.element_size()
        return total

    def get_total_cache_bytes(self) -> int:
        """Total cache memory (compressed cold + FP16 hot)."""
        return self.get_compressed_memory_bytes() + self.get_hot_memory_bytes()

    def get_fp16_equivalent_bytes(self) -> int:
        """What the same cache would cost entirely in FP16."""
        total = 0
        for i, s in enumerate(self._seq_lengths):
            if s > 0 and self._hot_keys[i] is not None:
                batch = self._hot_keys[i].shape[0]
                heads = self._hot_keys[i].shape[1]
                total += batch * heads * s * self.head_dim * 2  # K + V
        return total * 2  # FP16 = 2 bytes

    def get_compression_ratio(self) -> float:
        actual = self.get_total_cache_bytes()
        if actual == 0:
            return 0.0
        fp16 = self.get_fp16_equivalent_bytes()
        return fp16 / actual

    def print_memory_report(self):
        cold_mb = self.get_compressed_memory_bytes() / 1e6
        hot_mb = self.get_hot_memory_bytes() / 1e6
        total_mb = self.get_total_cache_bytes() / 1e6
        fp16_mb = self.get_fp16_equivalent_bytes() / 1e6
        ratio = self.get_compression_ratio()
        seq_len = max(self._seq_lengths) if self._seq_lengths else 0
        cold_len = max(self._cold_lengths) if self._cold_lengths else 0
        hot_len = seq_len - cold_len
        print(f"[TurboQuantCache] seq={seq_len} (cold={cold_len}, hot={hot_len}), "
              f"actual={total_mb:.1f} MB (cold={cold_mb:.1f} + hot={hot_mb:.1f}), "
              f"FP16 equiv={fp16_mb:.1f} MB, ratio={ratio:.1f}x")
