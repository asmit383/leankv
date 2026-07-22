# Raw CUDA sparse GEMV — below Triton

Batch-1 memory-bound decode: each projection is a matrix–vector product bound by
HBM bandwidth. leankv already does this in **Triton**. This branch rewrites it in
**raw CUDA** (one level lower), then adds **int4 weights** — the byte-reduction
lever Triton-fp16 can't reach.

All measured on **NVIDIA L4** (300 GB/s HBM), shape 14336×4096, 40% sparsity.

## The comparison

| kernel | level | time/call | bandwidth | vs Triton | correct? |
|---|---|---:|---:|---:|:--:|
| **Triton fp16** (before, leankv) | Triton DSL | 0.275 ms | 85% | 1.0× | ✓ |
| **CUDA fp16** (this branch) | raw CUDA | 0.280 ms | 84% | ~1.0× (ties) | ✓ 1e-5 |
| **CUDA int4** (this branch) | raw CUDA | **0.087 ms** | 70% | **~3.2×** | ✓ 1e-5 |

**Takeaway:** at fp16 both kernels hit the same HBM wall, so raw CUDA *ties*
Triton — the point was to reach the wall by hand. The real win is **int4**: 4× fewer
weight bytes → **~3.2× faster than Triton at batch-1**, the thing Triton-fp16 has no
way to do.

## Why go below Triton at all

Triton can't be beaten at fp16 (both saturate HBM). It *can* do int4 — but the
fast int4 unpack (`LOP3` bit-trick), custom bit-packing, and fusing dequant with
the sparsity skip need instruction-level control Triton won't give you. The SOTA
int4 kernel (Marlin) is hand-written CUDA for exactly this reason.

## The int4 kernel, in one line

```
y[n] = Σ_k ( |x[k]| > t ?  x[k] · dequant_int4(Wq[k,n])  : 0 )
```
Sparsity skips zeroed rows; int4 weights stay packed in HBM and are unpacked in
registers (never written back as fp16). Symmetric, group-wise (G=128), 2 nibbles/byte.

## Files

| file | what |
|---|---|
| `sparse_gemv.cu` | fp16 kernel + bandwidth benchmark |
| `sparse_int4_gemv.cu` | **fused sparse-int4 kernel** (the headline) |
| `sparse_gemv_ext.cu`, `bench_compare.py` | head-to-head vs Triton |
| `forward_test.py` | in-model correctness vs fp32 ground truth |
| `bench_forward.py` | end-to-end decode tok/s (fp16) |
| `sparse_gemv_hip.cpp`, `run_amd.sh` | HIP/CDNA port for MI300X |
| `DESIGN_sparse_int4.md`, `RESULTS.md` | design + detailed write-up |

## Status (honest)

- int4 kernel is **standalone-tested** (correct, 3.2× at kernel level). **End-to-end
  int4 tok/s not yet run** (`bench_int4_forward.py` is committed but untested).
- int4 uses a **naive unpack**; `LOP3` optimization is next (raises the 70% back up).
- int4 **weight-quant quality** (perplexity) not yet measured — separate axis.
- HIP kernel **not run on AMD hardware** (no MI300X yet). All numbers are NVIDIA L4.

## Build

```bash
nvcc -O3 -arch=sm_89 sparse_gemv.cu       -o k && ./k    # fp16 bandwidth
nvcc -O3 -arch=sm_89 sparse_int4_gemv.cu  -o k && ./k    # fused sparse-int4
```
