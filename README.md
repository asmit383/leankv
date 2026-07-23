# LeanKV — hand-written CUDA kernels for batch-1 LLM decode

Batch-1 decode is **memory-bandwidth bound**: each projection is a matrix–vector
product where the GPU spends all its time moving the weight matrix through HBM.
This repo takes that kernel from **Triton** down to **raw CUDA** (one level lower),
then adds **int4 weights** — the byte-reduction lever a plain fp16 kernel can't reach.

All numbers measured on **NVIDIA L4** (300 GB/s HBM), shape 14336×4096, 40% activation sparsity.

## The comparison

| kernel | level | time/call | bandwidth | vs Triton | correct? |
|---|---|---:|---:|---:|:--:|
| **Triton fp16** (starting point) | Triton DSL | 0.275 ms | 85% | 1.0× | ✓ |
| **CUDA fp16** (hand-written) | raw CUDA | 0.280 ms | 84% | ties | ✓ (1e-5) |
| **CUDA int4** (fused sparse+quant) | raw CUDA | **0.087 ms** | 70% | **~3.2×** | ✓ (1e-5) |

- **fp16: raw CUDA ties Triton.** Both saturate the same HBM wall — the point was to
  reach it by hand, and it does (85%→84%, matching output to 1e-5).
- **int4: raw CUDA beats Triton ~3.2×.** 4× fewer weight bytes → ~3.2× faster at
  batch-1. This is the lever Triton-fp16 has no way to pull.

## Why go below Triton

Triton can't be beaten at fp16 (both hit HBM). It *can* do int4 — but the fast
int4 unpack (`LOP3` bit-trick), custom bit-packing, and fusing dequant with the
sparsity skip need instruction-level control Triton won't emit. The state-of-the-art
int4 kernel (Marlin) is hand-written CUDA for exactly this reason.

## The fused sparse-int4 kernel, in one line

```
y[n] = Σ_k ( |x[k]| > t ?  x[k] · dequant_int4(Wq[k,n])  : 0 )
```

Two wins in one pass, in registers:
- **Sparsity** — skip whole weight rows for zeroed activations
- **int4** — weights stay packed in HBM (0.5 B vs fp16's 2 B), unpacked in registers,
  never written back as fp16

## End-to-end on Mistral-7B (batch-1 decode tok/s)

| config | tok/s | vs dense |
|---|---:|---:|
| PyTorch dense fp16 | 16.7 | 1.0× |
| Triton fp16 sparse (leankv original) | 22.2 | 1.31× |
| int4, unfused | 17.8 | 1.07× |
| **fused int4** (QKV + gate/up in one kernel) | **27.1** | **1.62×** |
| fused int4 + sparsity | **35.4** | **2.12×** |

Fusing the projections that share an input (q/k/v; gate/up) into single concatenated
int4 GEMVs takes int4 from 17.8 → 27.1 (+52%): fewer, bigger kernels → better
occupancy. The 27.1 (1.62×) case is coherent (beats the Triton fp16 result); the
2.12× case adds uncalibrated sparsity (speed only — int4 quant + sparsity quality is
a separate axis). Memory ceiling on L4 is ~85 tok/s (dense int4).

**CUDA graphs were a dead end here** (`bench_int4_graph.py`): once correct (a kernel
must launch on the capture stream, not the default stream, or the graph is empty),
graph replay is only ~1.02× — batch-1 decode on a 7B is GPU-bound, not
launch-overhead-bound. The real lever was kernel fusion, above.

## In-model correctness

The kernel drives **all 154 projections** of a real model during decode, vs an fp32
ground-truth forward: cosine **0.99994** (PyTorch dense is 0.99996), argmax token
matches, generated text identical. It tracks ground truth as tightly as PyTorch's
own dense path.

## Where the kernels live

```
leankv/teal/kernels/
├── sparse_gemv.py            # original Triton kernel (the starting point)
└── cuda/                     # this work — see cuda/README.md for details
    ├── sparse_gemv.cu        # fp16 kernel + bandwidth benchmark
    ├── sparse_int4_gemv.cu   # fused sparse-int4 kernel (the headline)
    ├── forward_test.py       # in-model correctness vs fp32 ground truth
    ├── bench_forward.py      # end-to-end decode tok/s
    ├── sparse_gemv_hip.cpp   # HIP/CDNA port for AMD MI300X
    └── DESIGN_sparse_int4.md # design + phase plan
```

## Build

```bash
nvcc -O3 -arch=sm_89 leankv/teal/kernels/cuda/sparse_gemv.cu      -o k && ./k
nvcc -O3 -arch=sm_89 leankv/teal/kernels/cuda/sparse_int4_gemv.cu -o k && ./k
```

## Status (honest)

- int4 kernel: standalone-tested (3.2× kernel-level) **and** end-to-end on Mistral-7B
  (1.07× dense at int4 alone, up to 1.45× with sparsity — beats Triton's 22.2 tok/s).
- int4 uses a **naive unpack**; the `LOP3` optimization (raises the 70% back up) is next.
- int4 **weight-quant quality** (perplexity) not yet measured — separate axis.
- HIP kernel is **not run on AMD hardware** yet (no MI300X). All numbers are NVIDIA L4.

---

<details>
<summary><b>Background: the original LeanKV research (TEAL + TurboQuant)</b></summary>

Two training-free inference optimizations, the foundation this kernel work builds on:

- **TEAL** — magnitude-based activation sparsity ([ICLR 2025](https://arxiv.org/abs/2408.14690)).
  Zeroes small activations so the GEMV can skip weight rows. **1.31× at B=1** on
  Mistral-7B (16.9 → 22.2 tok/s, L4) with calibrated thresholds. B=1 only — at B>1,
  dense batched GEMM wins.
- **TurboQuant** — KV-cache compression to 3-bit via random rotation ([arXiv 2025](https://arxiv.org/abs/2504.19874)).
  **4.5× compression, zero perplexity loss** (WikiText-2: 5.34 both FP16 and 3-bit).

Modules: `leankv/teal/` (sparsity), `leankv/turboquant/` (KV compression),
`scripts/` (calibration, benchmarks, quality eval). Both are research prototypes;
the CUDA kernels above are the step toward the fused kernels production needs.

</details>
