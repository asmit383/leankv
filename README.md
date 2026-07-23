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

## End-to-end on Mistral-7B (batch-1 decode tok/s, L4)

The optimization ladder — each step measured, dense fp16 = 16.7 tok/s baseline:

| step | tok/s | vs dense |
|---|---:|---:|
| PyTorch dense fp16 | 16.7 | 1.0× |
| Triton fp16 sparse (leankv original) | 22.2 | 1.31× |
| int4, unfused | 17.8 | 1.07× |
| + fuse QKV & gate/up into one kernel each | 27.1 | 1.62× |
| + tune split-K (BLOCK_K 512→128) | ~37 | ~2.2× |
| + CUDA graph (int4 weights, no sparsity) | **45.3** | **2.71×** |
| + calibrated 40% activation sparsity | **64.2** | **3.84×** |
| + 50% sparsity | 69.8 | 4.18× |

Levers by impact: **fusion** (project'ns that share an input → one concatenated int4
GEMV) > **split-K tuning** (more blocks fill the GPU) > **CUDA graph**. The graph
helps *only* once fusion+tuning made decode launch-bound rather than weight-bound —
it was ~1.02× when weight-bound. CUDA-graph gotcha: custom kernels must launch on
`at::cuda::getCurrentCUDAStream()` or capture records an empty graph (fast garbage).

## Quality (WikiText-2 perplexity) — the honest trade

| config | perplexity | Δ vs fp16 | decode tok/s |
|---|---:|---:|---:|
| fp16 baseline | 5.05 | — | 16.7 |
| **int4 weights** | 5.26 | **+4%** | **45.3** |
| int4 + 40% calibrated sparsity | 5.92 | +17% | 64.2 |

**int4 weight quantization is nearly lossless (+4% ppl) for a 2.7× speedup** — that's
the clean operating point. Adding 40% activation sparsity buys another 1.4× (to 3.8×)
but costs ~17% perplexity: a real speed/quality trade, not free. Calibration
(`calibrate_simple.py`) picks per-projection thresholds; the sparsity axis is where the
quality goes, so it's opt-in.

Try it: `python3 chat.py --thresholds thresholds.json` streams generation live at the
fast-path speed.

## In-model correctness

The kernel drives **all 154 projections** of a real model during decode, vs an fp32
ground-truth forward: cosine **0.99994** (PyTorch dense is 0.99996), argmax token
matches, generated text identical. It tracks ground truth as tightly as PyTorch's
own dense path.

## Where the kernels live

```
leankv/teal/kernels/
├── sparse_gemv.py               # original Triton kernel (the starting point)
└── cuda/                        # this work — see cuda/README.md for details
    ├── sparse_gemv.cu           # fp16 kernel + bandwidth benchmark (ties Triton, 85%)
    ├── sparse_int4_gemv.cu      # fused sparse-int4 kernel + correctness (3.2× kernel)
    ├── sparse_int4_gemv_lop3.cu # LOP3 half2 int4→fp16 unpack (naive vs LOP3)
    ├── sparse_int4_ext_gs.cu    # graph-safe fused-int4 ext (tunable split-K)
    ├── bench_int4_fused.py       # fused QKV + gate/up decode benchmark
    ├── bench_int4_fused_graph.py # fused + CUDA graph + calibrated thresholds
    ├── calibrate_simple.py       # per-layer activation-sparsity thresholds
    ├── ppl.py                    # WikiText-2 perplexity (fp16 / int4 / int4+sparse)
    ├── chat.py                   # streaming chat on the fast path (feel the speed)
    ├── forward_test.py           # in-model correctness vs fp32 ground truth
    ├── sparse_gemv_hip.cpp       # HIP/CDNA port for AMD MI300X
    └── DESIGN_sparse_int4.md     # design + phase plan
```

## Build & run

```bash
cd leankv/teal/kernels/cuda
nvcc -O3 -arch=sm_89 sparse_gemv.cu       -o k && ./k    # fp16 bandwidth (85%)
nvcc -O3 -arch=sm_89 sparse_int4_gemv.cu  -o k && ./k    # fused sparse-int4 kernel

# end-to-end (needs torch; L4 / sm_89)
python3 calibrate_simple.py --sparsity 0.40           # -> thresholds.json
python3 bench_int4_fused_graph.py --thresholds thresholds.json   # 64 tok/s
python3 ppl.py --thresholds thresholds.json           # quality
python3 chat.py --thresholds thresholds.json          # interactive, streaming
```

## Status (honest)

- Kernel: standalone 3.2× vs Triton; **end-to-end 45 tok/s (int4, +4% ppl) → 64 tok/s
  (40% sparse, +17% ppl)** on Mistral-7B, L4 — from a 16.7 dense baseline.
- **LOP3 unpack** done (~1.06× kernel, negligible in-model — decode is bandwidth/occupancy
  bound, not unpack bound).
- **CUDA graphs** help only after fusion+tuning make decode launch-bound (1.02× → 1.2–2×).
- **HIP kernel not run on AMD hardware** yet (no MI300X). All numbers are NVIDIA L4;
  MI300X (~5.3 TB/s, ~11× the bandwidth) is the next multiplier.

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
