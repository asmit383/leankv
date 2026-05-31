# LeanKV: TEAL + TurboQuant for Cost-Efficient LLM Inference

Two training-free, complementary optimization techniques for LLM inference:

- **TEAL** — activation sparsity, reduces compute per token
- **TurboQuant** — KV cache compression (16-bit to 3-bit), reduces memory

Model weights stay FP16. No retraining required.

## Papers

| Paper | Venue | What |
|---|---|---|
| [TEAL](https://arxiv.org/abs/2408.14690) | ICLR 2025 Spotlight | Magnitude-based activation sparsity |
| [TurboQuant](https://arxiv.org/abs/2504.19874) | arXiv 2025 | KV cache quantization via random rotation + scalar quantizers |

## Results (Measured)

All benchmarks on **NVIDIA L4 24GB**, Mistral 7B v0.3 FP16, via Runcrate.

### TEAL: 1.31x Speedup at B=1

Triton sparse GEMV kernel with calibrated per-layer thresholds (40% sparsity, 206/224 projections active):

| Batch | Baseline tok/s | TEAL tok/s | Speedup |
|---|---|---|---|
| **1** | **16.9** | **22.2** | **1.31x** |
| 4 | 64.8 | 66.2 | 1.02x |
| 8 | 127.7 | 129.8 | 1.02x |
| 16 | 248.1 | 253.1 | 1.02x |

TEAL is a B=1 technique. At B>1 the sparse GEMV kernel cannot beat dense batched GEMM (weight reloading overhead), so it falls back to dense matmul automatically.

### TurboQuant: Zero Quality Loss at 4.5x Compression

WikiText-2 perplexity (1.3M tokens):

| Config | Perplexity |
|---|---|
| Baseline FP16 | **5.34** |
| TurboQuant 3-bit | **5.34** |

### TurboQuant: 5x More Concurrent Requests

At 4096 tokens, KV cache = 537 MB/sequence (FP16). On L4-24GB (9.5 GB free after model):

| Config | KV per sequence | Max batch | Measured |
|---|---|---|---|
| Baseline FP16 | 537 MB | B=16 | B=32 OOMs |
| TurboQuant 3-bit | 119 MB | B=80 | — |

Baseline B=32 OOM confirmed by benchmark. TurboQuant capacity is theoretical (requires fused CUDA kernels for on-the-fly dequantization during attention).

### Baseline: Long Sequences (4096 tokens)

| Batch | Total tok/s | Per-seq tok/s | VRAM | Status |
|---|---|---|---|---|
| 1 | 16.4 | 16.4 | 14.90 GB | OK |
| 4 | 57.7 | 14.4 | 16.07 GB | OK |
| 8 | 99.4 | 12.4 | 17.64 GB | OK |
| 16 | 154.9 | 9.7 | 20.78 GB | OK |
| 32 | — | — | — | OOM |

## Key Findings

1. **TEAL and TurboQuant are complementary** — TEAL reduces compute (B=1), TurboQuant reduces memory (B>1)
2. **TEAL does not scale to B>1** — unstructured activation sparsity cannot beat dense batched GEMM due to per-item weight reloading
3. **TurboQuant is quality-neutral at 3-bit** — random rotation spreads information evenly before quantization
4. **Actual VRAM savings require fused CUDA kernels** — Python dequantize-during-attention creates temporary FP16 spikes that negate savings

## Architecture

```
leankv/
├── teal/
│   ├── sparse_fns.py          # magnitude-based activation masking
│   ├── patching.py            # monkey-patch HF models with sparsity
│   ├── calibration.py         # activation histogram collection
│   ├── greedy_opt.py          # per-layer sparsity allocation
│   └── kernels/
│       └── sparse_gemv.py     # Triton sparse GEMV (ported from TEAL repo)
├── turboquant/
│   ├── codebook.py            # Lloyd-Max codebook for Beta distribution
│   ├── rotation.py            # random orthogonal rotation + QJL
│   ├── quantizer.py           # TurboQuantMSE quantize/dequantize
│   └── cache.py               # HF-compatible KV cache with compression
├── combined.py                # unified API
└── utils.py                   # VRAM tracking, timing

scripts/
├── raw_baseline.py            # standalone baseline benchmark
├── benchmark.py               # TEAL + TurboQuant benchmark
├── calibrate.py               # TEAL threshold calibration
└── eval_quality.py            # perplexity evaluation
```

## Usage

```bash
# Install
pip install torch transformers accelerate scipy pyyaml triton datasets
pip install .

# Baseline
python3 scripts/raw_baseline.py --batch-sizes 1 4 8 16

# Calibrate TEAL (one-time, ~15 min)
python3 scripts/calibrate.py --model mistralai/Mistral-7B-v0.3 --sparsity 0.4 --samples 100

# TEAL with Triton kernel
python3 scripts/benchmark.py --model mistralai/Mistral-7B-v0.3 \
  --teal-thresholds thresholds/mistral_7b_v0.3/thresholds_s40.json \
  --triton --batch-sizes 1

# Quality evaluation
python3 scripts/eval_quality.py --model mistralai/Mistral-7B-v0.3 --eval ppl
python3 scripts/eval_quality.py --model mistralai/Mistral-7B-v0.3 --tq-bits 3 --eval ppl
```

## Limitations

This is a research prototype, not a production inference engine. The Python implementation validates the techniques and measures quality impact. Key limitations:

- **TEAL only helps at B=1** — unstructured activation sparsity cannot beat dense batched GEMM at B>1 due to per-item weight reloading overhead. This is a fundamental hardware constraint, not an implementation issue.
- **TurboQuant VRAM savings are theoretical** — our implementation dequantizes KV to FP16 during attention, creating a temporary memory spike. Actual VRAM reduction requires fused CUDA kernels that compute attention directly on compressed data.
- **No production serving** — no continuous batching, no PagedAttention, no HTTP API. This project measures techniques, not serves traffic.

Production deployment requires fused CUDA kernels, which is engineering work beyond the scope of a research project. Production inference engines have demonstrated this is feasible at scale.

## Production Impact

Despite prototype limitations, the measured results have real production implications:

| Current Production (vLLM FP8 KV) | With TurboQuant 3-bit KV |
|---|---|
| 2x KV compression | **4.5x KV compression** |
| ~35 concurrent requests (L4, 4096 tok) | **~80 concurrent requests** |
| Small quality loss | **Zero quality loss** |

TurboQuant provides 2.25x more concurrent requests than FP8 KV cache with no quality degradation — a direct improvement over current production standards, pending fused kernel integration.

## Future Work

- Fused CUDA kernels for TurboQuant attention (dequantize on-the-fly, no FP16 spike)
- Integration with vLLM/SGLang PagedAttention
- QTIP weight quantization (4-bit, nearly lossless) — reduces model from 14.5 GB to 3.6 GB, enabling B=171 combined with TurboQuant
- Multi-GPU benchmarks across GPU tiers (A100, RTX 4090, L4, A16)
