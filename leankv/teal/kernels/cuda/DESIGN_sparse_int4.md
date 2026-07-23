# Fused sparse-int4 GEMV — design & plan

Goal: a single hand-written CUDA kernel for batch-1 decode that fuses **three**
wins in one pass over the weight matrix, all in-register, never materializing
fp16 weights in HBM:

```
y[n] = Σ_k ( |x[k]| > t ?  x[k] · dequant(Wq[k,n])  : 0 )
```

1. **Sparsity** — skip loading quantized rows for zeroed activations.
2. **Quantization** — weights stored int4 (0.5 B) not fp16 (2 B) → 4× less HBM traffic.
3. **Fusion** — int4 stays packed in HBM; unpack happens in registers (LOP3), so
   dequant never round-trips through memory. This is the whole point: a separate
   dequant kernel would write fp16 back to HBM and *lose* the byte savings.

Why below Triton: the int4→fp16 unpack wants the `LOP3.LUT` bit trick and tight
load scheduling that Triton won't emit well. The SOTA int4 kernel (**Marlin**,
vLLM) is hand-written CUDA for exactly this reason.

## Quantization scheme (first version)

- **int4, symmetric, group-wise.** Group size G=128 along K. Each group shares one
  fp16 scale per output column: `w ≈ q · scale`, `q ∈ [-8,7]`.
- **Packing:** 2 int4 / byte. Packed buffer `Wq` is `[K, N/2]` uint8, column-major
  (consecutive n contiguous → coalesced), low/high nibble = adjacent n.
- **Scales:** fp16, shape `[K/G, N]`. For (k,n): `scale = scales[k/128, n]`.
- Asymmetric (zero-point, AWQ-style) is a later extension.

## Kernel (reuses the fp16 sparse_gemv skeleton)

Same split-K / VEC-columns-per-thread / shared-x / uniform-skip / fp32-accum /
atomicAdd structure as `sparse_gemv.cu`. Deltas:

- Load **packed int4** for VEC columns: one 32-bit load = 8 nibbles = 8 n-values.
- **Unpack** nibble → fp16 in registers.
- Load **scale** for `scales[k/G, n_base:+VEC]` (changes only every 128 k's; cached).
- `acc[j] += x[k] · (q[j] · scale[j])`.

Bytes moved (the win): weights = `active_k · N · 0.5 B` + scales (`K/G · N · 2 B`,
small). vs fp16 kernel: ~4× fewer weight bytes → up to ~4× faster at B=1 (memory
bound), on top of the sparsity skip.

## LOP3 unpack (the ISA move)

int4→fp16 fast path: OR the 4 mantissa bits into an fp16 with fixed exponent
(0x64 → value in [1024,1040)), then subtract 1024 to recover 0..15, shift to
signed. `LOP3.LUT` does the mask+OR in one instruction. **Build correctness first
with a naive shift+mask+convert unpack, then swap in LOP3 and re-verify.**

## Phases (each independently testable)

1. **Quant utils (Python/torch):** fp16 → int4 group-wise quantize, pack to uint8,
   scales. Dequant reference. Validate round-trip error (~int4 noise).
2. **Naive int4 kernel:** packed load + shift/mask unpack + scale + sparse skip +
   accumulate. Correctness vs dequant-then-GEMV reference (tight — rounding only).
   Measure GB/s and speedup vs the fp16 sparse kernel.
3. **LOP3 optimization:** replace naive unpack with the bit trick. Re-verify, measure.
4. **Sparsity + bench:** confirm skip works with quant; report combined speedup vs
   fp16 sparse and vs dense. Target: ~3–4× fewer bytes → large B=1 speedup.
5. **(optional) End-to-end:** quantize Mistral-7B weights to int4, wire kernel in,
   measure tok/s vs fp16 dense/sparse and vs Triton.
6. **(optional) HIP port** for MI300X (gfx942), same fusion.

## Correctness strategy

- Kernel output vs **dequant-then-GEMV** reference (same quantized weights) → must
  match to fp32 rounding. This validates the kernel math/layout.
- Separately report int4-vs-fp16 quantization error (inherent, larger) so the two
  error sources aren't conflated.

## Deliverables

- `sparse_int4_gemv.cu` — kernel + standalone bench + correctness harness
- `quant_utils.py` — quantize/pack/scales + dequant reference
- RESULTS update: "fused sparse-int4 GEMV: N× over fp16 sparse, M% bandwidth,
  correct vs dequant reference"

## Honest expectations

- int4 ≈ 4× fewer weight bytes → the real B=1 throughput lever; this is what
  *beats* Triton (fewer bytes), not just ties it.
- LOP3 vs naive unpack is a compute-side win; since we're memory-bound it mostly
  buys headroom, but it's the instruction-level showcase Nicolas cares about.
- Quality (int4 weight quant) is a separate axis — report it, don't conflate with
  the bandwidth/speed numbers.
