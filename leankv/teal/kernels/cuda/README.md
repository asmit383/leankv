# Raw CUDA sparse GEMV kernels (one level below Triton)

Hand-written CUDA (and a HIP/CDNA port) for the TEAL activation-sparsity path, in
the **batch-1 memory-bound decode** regime — where each projection is a
matrix–vector product bound entirely by HBM bandwidth. The goal: take leankv's
Triton kernel one level lower, to raw CUDA, and then add the byte-reduction lever
(int4 weights) that a plain fp16 kernel can't reach.

All numbers measured on an **NVIDIA L4** (24 GB GDDR6, 300 GB/s peak HBM), CUDA
12.4, `sm_89`.

---

## What leankv had before (on `main`)

- **TEAL activation sparsity via a Triton kernel** (`../sparse_gemv.py`, ported
  from FasterDecoding/TEAL). Zeroes small-magnitude activations, then a Triton
  SplitK kernel skips the corresponding weight rows.
  - Measured: **1.31× at B=1** on Mistral-7B (16.9 → 22.2 tok/s) with **calibrated**
    thresholds, on the L4.
- **TurboQuant** KV-cache compression (separate module).
- Everything lives at the **Triton** level — a DSL that compiles to PTX; you don't
  control the emitted instructions or memory layout.

## What this branch adds (`cuda-sparse-gemv`)

The same sparse-GEMV idea, rewritten **below Triton** in CUDA C++, then extended:

1. A raw CUDA fp16 sparse GEMV that **ties** the Triton kernel at the HBM wall.
2. A **fused sparse + int4-dequant** kernel that is **~3.3× faster than Triton** at
   B=1 — the byte-reduction lever fp16 can't reach.
3. In-model correctness on a real model, a fused gate+up variant, a HIP/CDNA port
   for MI300X, and a Mistral-7B end-to-end run.

---

## Results

### 1. fp16 sparse GEMV vs Triton (`sparse_gemv.cu`, `sparse_gemv_v2.cu`)

`float4` vectorized loads, split-K for occupancy, warp-uniform sparsity skip
(zero divergence), fp32 accumulate + atomicAdd.

| metric | result |
|---|---|
| bandwidth utilization | **85–87%** of 300 GB/s (large HBM-bound shapes) |
| sparsity → wall-clock | linear: 40% sparse = 0.60× time, 60% = 0.40× |
| vs Triton (14336×4096) | **tied** (~0.98×) — both saturate the same HBM ceiling |
| numerics | fp32 accum; ~60× tighter than the Triton kernel in the microbenchmark |

Both kernels hit the memory wall, so they tie — the point was to reach it by hand.

### 2. In-model correctness (`forward_test.py`)

Kernel drives **all 154 projections** of a Llama-arch model during decode,
compared to an **fp32 ground-truth** forward.

| path | cosine vs fp32 truth |
|---|---|
| PyTorch dense fp16 | 0.999956 |
| **raw CUDA kernel** | 0.999936 |

Argmax next-token matches; generated text identical. The kernel tracks ground
truth as closely as PyTorch's own dense path.

### 3. Fused gate+up (`fused_gate_up.cu`)

gate_proj and up_proj share input + mask → stream both weights in one pass,
`silu·up` in the epilogue. **Bit-identical** output, **~4%** faster at B=1
(overhead-bound by design — weight traffic is irreducible at B=1).

### 4. Fused sparse-int4 GEMV — the headline (`sparse_int4_gemv.cu`)

int4 weights (symmetric, group-wise G=128, packed 2/byte, column-major),
**unpacked in registers** so they never round-trip HBM as fp16. Sparsity skip on
top.

| sparsity | int4 time | fp16 sparse time | speedup | rel err |
|---:|---:|---:|---:|---:|
| 0% | 0.133 ms | ~0.46 ms | ~3.4× | 9e-5 |
| 40% | 0.087 ms | 0.281 ms | **3.25×** | 1e-5 |
| 60% | 0.064 ms | ~0.19 ms | ~3.0× | 3e-5 |

4× fewer weight bytes (17.6 MB vs 70.4 MB at 40%). Since the fp16 kernel ties
Triton, **this is ~3.3× faster than Triton at B=1** (kernel-level) via the one
lever Triton-fp16 lacks: fewer bytes. Correct to ~1e-5 vs the dequant reference.

### 5. Mistral-7B end-to-end, fp16 kernel (`bench_forward.py`)

| config | tok/s |
|---|---|
| PyTorch dense fp16 | 16.7 (matches leankv baseline) |
| CUDA kernel, 29% sparse | 18.7 (crosses dense) |
| CUDA kernel, 62% sparse | 21.8 (1.31×) |

The naive per-call integration (a `torch.zeros` + fp16 cast per projection) is the
overhead; the kernel itself ties Triton. Sparsity here is uncalibrated (speed
measurement, not quality).

---

## Files

| file | what |
|---|---|
| `sparse_gemv.cu` / `_v2.cu` | fp16 kernel + bandwidth benchmark (v2 = VEC=16) |
| `sparse_gemv_ext.cu` + `bench_compare.py` | torch ext + head-to-head vs Triton |
| `fused_gate_up.cu` | fused SwiGLU projection |
| `forward_test.py` | in-model correctness vs fp32 ground truth |
| `bench_forward.py` | end-to-end decode tok/s (fp16) |
| **`sparse_int4_gemv.cu`** | **fused sparse-int4 kernel + correctness + bench** |
| `sparse_int4_ext.cu` + `bench_int4_forward.py` | int4 end-to-end harness (Phase 5, WIP) |
| `sparse_gemv_hip.cpp` + `run_amd.sh` | HIP/CDNA port for MI300X (gfx942) |
| `DESIGN_sparse_int4.md` | full int4 design & phase plan |
| `RESULTS.md` | detailed write-up |

---

## Status & limitations

- **int4 end-to-end (Phase 5) is not yet run** — the harness is committed
  (`bench_int4_forward.py`) but untested on a full model. The int4 numbers above
  are the standalone kernel.
- **int4 = LOP3 unpack pending (Phase 3).** Current unpack is naive shift/mask, so
  bandwidth utilization drops as sparsity rises (turns compute-bound). LOP3 is the
  next optimization.
- **int4 weight-quant quality** is a separate axis — not yet measured (perplexity).
- **HIP kernel is untested on AMD hardware** — no MI300X available; runs pending an
  AMD Developer Cloud credit. Numbers above are all NVIDIA L4.
- End-to-end tok/s carries naive-integration overhead; the fair kernel measure is
  the microbenchmark.

## Build & run

```bash
# standalone fp16 kernel + bandwidth
nvcc -O3 -arch=sm_89 sparse_gemv.cu -o sparse_gemv && ./sparse_gemv

# fused sparse-int4 kernel + correctness
nvcc -O3 -arch=sm_89 sparse_int4_gemv.cu -o sparse_int4_gemv && ./sparse_int4_gemv

# vs Triton / in-model / end-to-end (need torch + triton)
python3 bench_compare.py
python3 forward_test.py
python3 bench_forward.py --model mistralai/Mistral-7B-v0.3

# AMD MI300X (on a ROCm box)
hipcc -O3 --offload-arch=gfx942 sparse_gemv_hip.cpp -o k && ./k
```
