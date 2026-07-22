# Raw CUDA sparse GEMV — batch-1 memory-bound decode

Hand-written CUDA kernels for the TEAL activation-sparsity path in leankv, one
level below the existing Triton kernel. Target regime: **batch size 1 decode**,
where each projection is a matrix–vector product that is purely HBM-bandwidth
bound — the whole job is to move the weight matrix through the memory bus while
skipping the rows whose activation TEAL has zeroed.

All numbers measured on an **NVIDIA L4** (24 GB GDDR6, 300 GB/s peak HBM),
CUDA 12.4, `sm_89`. No AMD hardware was available on the provider; a HIP/CDNA
port is included but has not been run on an MI300X.

## Kernel design

`y[n] = Σ_k ( |x[k]| > t ? x[k] · W[k,n] : 0 )`, weight stored column-major so
consecutive output columns are contiguous (coalesced).

- **128-bit vectorized loads** (`float4` = 8 fp16 weights per thread per row).
- **Split-K over `grid.y`** so enough blocks launch to fill the SMs and keep
  enough memory requests in flight to hide HBM latency.
- **Uniform per-block sparsity skip** — all threads test the same `x[k]` from
  shared memory, so a skipped row is skipped by the whole block with **zero warp
  divergence**. Partial sums merged via `atomicAdd` in fp32.

## Results

### Bandwidth utilization (standalone, 16384×16384)

| sparsity | time/call | effective BW | utilization |
|---------:|----------:|-------------:|------------:|
| 0%       | 2.089 ms  | 257 GB/s     | **85.7%**   |
| 40%      | 1.252 ms  | 257 GB/s     | 85.6%       |
| 60%      | 0.830 ms  | 255 GB/s     | 85.1%       |

Effective bandwidth is constant (~257 GB/s) whether dense or sparse → the kernel
is genuinely HBM-bound, not overhead-bound. **Sparsity converts linearly into
wall-clock:** 40% sparse runs at 0.60× dense time, 60% at 0.40×. A `VEC=16`
variant (two independent 128-bit loads for more memory-level parallelism)
reaches **87.1%** on the large matrix.

### Head-to-head vs the Triton kernel (identical tensors)

| shape (N×K) | Triton | CUDA | CUDA rel-err | note |
|-------------|-------:|-----:|-------------:|------|
| 14336×4096  | 85.3% bw | 83.7% bw | 1.8e-5 | HBM-bound: **tied** (0.98×) |
| 4096×14336  | 85.5% bw | 83.7% bw | 4.9e-6 | HBM-bound: **tied** (0.98×) |
| 4096×4096   | faster | slower | 2.3e-6 | fits in 48 MB L2 — not HBM-bound |

On the large, genuinely bandwidth-bound projections the hand kernel **ties**
Triton — both saturate the same HBM ceiling — while being **~100× more
numerically accurate** (fp32 accumulation vs Triton's ~1e-3). On small matrices
that fit in the L4's 48 MB L2 the problem is launch-overhead-bound, not
memory-bound, and Triton's lower per-call overhead wins.

### Fused gate+up (SwiGLU) — one level deeper

gate_proj and up_proj read the same `x` with the same sparsity mask. Fusing them
streams both weight matrices in one pass and applies `silu(gate)·up` in a single
epilogue instead of round-tripping `gate`/`up` through HBM.

| | time/call | bw util |
|--|----------:|--------:|
| unfused (2 matmuls + elementwise) | 0.576 ms | 81.6% |
| **fused (1 matmul + elementwise)** | 0.556 ms | 84.4% |

**~3.5–4.6%** across sparsity levels, output **bit-identical** to unfused. The
win is modest and overhead-bound *by design*: at B=1 both weight matrices must be
streamed regardless, so fusion recovers launch + intermediate-materialization
cost, not weight traffic. This is the honest ceiling for fusion in this regime.

## Files

| file | what |
|------|------|
| `sparse_gemv.cu` | standalone kernel + bandwidth benchmark (VEC=8) |
| `sparse_gemv_v2.cu` | VEC=16 variant (more MLP) — 87% on large matrices |
| `sparse_gemv_ext.cu` | torch extension for the head-to-head |
| `bench_compare.py` | CUDA vs Triton on identical tensors |
| `fused_gate_up.cu` | fused SwiGLU projection benchmark |
| `sparse_gemv_hip.cpp` | HIP/CDNA port (untested on AMD hardware) |

## Fused sparse-int4 GEMV — the byte-reduction lever (beats Triton)

`sparse_int4_gemv.cu`: one kernel fusing activation sparsity + int4 weight
dequant. Weights are int4 (symmetric, group-wise G=128), packed 2/byte,
column-major; unpacked in registers (naive shift/mask for now — LOP3 next) so
they never round-trip HBM as fp16.

| sparsity | int4 time | fp16 sparse time | speedup | int4 bw util | rel err |
|---------:|----------:|-----------------:|--------:|-------------:|--------:|
| 0%       | 0.133 ms  | ~0.46 ms         | ~3.4×   | 76%          | 9e-5    |
| 40%      | 0.087 ms  | 0.281 ms         | 3.25×   | 70%          | 1e-5    |
| 60%      | 0.064 ms  | ~0.19 ms         | ~3.0×   | 63%          | 3e-5    |

int4 moves 4× fewer weight bytes (17.6 MB vs 70.4 MB at 40%). Since the fp16
kernel *ties* Triton, this is **~3.3× faster than Triton at B=1** — the lever
Triton-fp16 doesn't have. bw% falls as sparsity rises because the naive unpack
becomes compute-bound; the LOP3 unpack (next) pushes it back toward memory-bound.
Correct to ~1e-5 vs the dequant-then-GEMV reference. int4 weight-quant quality is
a separate axis (measure end-to-end perplexity before claiming quality).

## What actually raises throughput at B=1

The kernel is at the HBM wall (~85%); raising utilization further buys little.
Real B=1 throughput comes from **moving fewer bytes**: activation sparsity (done,
linear) or **low-bit weight loading** (int8 → ~2×, int4 → ~4×). Weight
quantization is the next lever — it reduces bytes rather than improving
utilization, so it is orthogonal to (and compounds with) everything above.
