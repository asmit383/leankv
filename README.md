# TEAL + TurboQuant: Cost-Efficient LLM Inference Without Precision Drop

## Project Summary

Combine two proven, training-free, near-lossless techniques to achieve 2-3x LLM inference throughput on the same GPU without any precision drop:

- **TEAL** — activation sparsity → less compute per token → faster decode
- **TurboQuant** — KV cache compression → more concurrent requests in same VRAM → higher batch throughput

These are complementary (compute reduction + memory reduction) and nobody has stacked them together. That's the novel contribution.

---

## Why There's No Precision Drop

| Component | What happens | Precision |
|---|---|---|
| Model weights | Stay FP16, untouched | Full |
| Compute (matmul, attention) | Runs in FP16 | Full |
| TEAL | Skips activations that are already near-zero | ~0 loss (ICLR 2025 proved it) |
| TurboQuant | Compresses KV cache from 16-bit to 3.5-bit | Quality-neutral (paper proved it) |
| Final output | FP16 quality | Full |

The model itself is never modified. TEAL skips work that produces near-zero results. TurboQuant compresses temporary memory (KV cache), not weights.

---

## Papers

### Core (must implement)

1. **TEAL** — Training-Free Activation Sparsity in LLMs
   - Paper: https://arxiv.org/abs/2408.14690
   - Venue: ICLR 2025 Spotlight
   - What: Magnitude-based activation sparsity, skips 40-50% of near-zero activations
   - Result: 1.53x-1.8x wall-clock decode speedup
   - Works on: Llama-2, Llama-3, Mistral (7B-70B)
   - Key: Training-free, drop-in, no quality loss at 40% sparsity

2. **TurboQuant** — Near-Optimal Vector Quantization
   - Paper: https://arxiv.org/abs/2504.19874
   - What: KV cache quantization to 3.5 bits using random rotation + scalar quantizers
   - Result: Quality-neutral at 3.5 bits, allows 4.5x more context/bigger batches
   - Key: Data-oblivious, suitable for online applications

### Stretch goals

3. **QTIP** — Quantization with Trellises and Incoherence Processing
   - Paper: https://arxiv.org/abs/2406.11235
   - Venue: NeurIPS 2024 Spotlight
   - What: Trellis coded weight quantization, avoids exponential codebook
   - Use case: Fit 70B model on a single 48GB GPU (weight compression)

4. **SCMoE** — Self-Contrast Mixture-of-Experts
   - Paper: https://arxiv.org/abs/2405.14507
   - Venue: NeurIPS 2024
   - What: Uses unchosen MoE experts via contrastive decoding for better accuracy
   - Use case: Free quality boost on Mixtral-style models

5. **TD-MoE** — Tensor Decomposition for MoE
   - Paper: https://openreview.net/forum?id=D9cnZNZfxX
   - What: Cross-expert tensor decomposition, 20% compression nearly lossless
   - Use case: Compress MoE models to fit on cheaper hardware

---

## Baseline Benchmarks (Completed)

### Hardware
- GPU: NVIDIA A100-SXM4-80GB
- CUDA: 13.0
- Python: 3.10.12
- Framework: HuggingFace Transformers
- Instance: Runcrate (hopeful-feynman)

### Setup
- Single request, 128 output tokens
- Prompt: "Explain the theory of relativity in simple terms."
- 5-run average after warmup
- FP16 dtype

### Results

| Model | Parameters | Throughput (tok/s) | VRAM Used | 
|---|---|---|---|
| Mistral 7B v0.3 FP16 | 7.2B | **64.8 tok/s** | 14.5 GB |
| Llama 3 8B FP16 | 8.0B | **58.8 tok/s** | 16.1 GB |

### Notes
- Llama 3 is slightly slower due to larger param count + larger vocab (128k vs 32k)
- A100 80GB has plenty of headroom — only using ~16GB of 80GB
- This headroom is exactly what TurboQuant exploits (compress KV cache → fit bigger batches)

---

## What To Build

### Phase 1: TEAL only
1. Integrate TEAL activation sparsity into the inference loop
2. Test at 30%, 40%, 50% sparsity levels
3. Measure tok/s at each level
4. Measure quality (perplexity on WikiText-2, MMLU accuracy)
5. Expected: ~1.5x throughput, <0.1% quality drop at 40% sparsity

### Phase 2: TurboQuant only
1. Integrate TurboQuant KV cache compression
2. Test at 4-bit, 3.5-bit, 3-bit, 2.5-bit
3. Measure max batch size before OOM at each bit level
4. Measure throughput with batched requests (8, 16, 32 concurrent)
5. Measure quality at each bit level
6. Expected: 4.5x more concurrent requests at 3.5-bit, no quality drop

### Phase 3: TEAL + TurboQuant combined
1. Stack both techniques
2. Measure combined throughput with batching
3. Measure combined quality impact
4. Expected: 2-3x total throughput over baseline

### Phase 4: Cost analysis
1. Run same benchmarks on different GPUs (A100, RTX 4090, L40S, A16)
2. Calculate cost per 1M tokens for each GPU with and without optimizations
3. Find the crossover: at what point does a cheap GPU + optimizations beat an expensive GPU without
4. GPU access available via Runcrate platform

### Phase 5: Quality validation
1. Perplexity on WikiText-2 (lower = better, must match baseline within 0.5%)
2. MMLU accuracy (must match baseline within 1%)
3. GSM8K math reasoning (sensitive to precision, good stress test)
4. HumanEval code generation

---

## Expected Final Results Table

| Setup | Throughput | vs Baseline | Quality Drop |
|---|---|---|---|
| Baseline (FP16, no optimization) | 58-65 tok/s | 1.0x | 0% |
| + TEAL (40% sparsity) | ~90-100 tok/s | ~1.5x | <0.1% |
| + TurboQuant (3.5-bit KV cache) | ~100-120 tok/s (batched) | ~1.8x | ~0% |
| + Both combined | ~180-220 tok/s (batched) | ~2.5-3x | <0.1% |

These are estimates. The actual measurements are the research contribution.

---

## Key Research Question

"Can stacking activation sparsity (TEAL) and KV cache quantization (TurboQuant) achieve 2-3x inference throughput on the same GPU, with no measurable quality degradation, and what is the cost-per-token improvement across different GPU tiers?"

---

## Why This Works as a Final Year Project

1. **Novel combination** — both techniques exist independently, nobody combined them
2. **Training-free** — no GPU days wasted on retraining
3. **Measurable** — clear metrics (tok/s, perplexity, $/1M tokens)
4. **Reproducible** — real hardware via Runcrate, standard benchmarks
5. **Publishable** — if results hold, it's a workshop paper at minimum
6. **Practical** — directly applicable to production inference serving

---

## Benchmark Script

The baseline benchmark script used:

```python
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'meta-llama/Meta-Llama-3-8B'  # or 'mistralai/Mistral-7B-v0.3'

print(f'Loading {model_name}...')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print('Model loaded.')

prompt = 'Explain the theory of relativity in simple terms.'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

# Warmup
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=10)

# Benchmark
times = []
for i in range(5):
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    tokens = out.shape[1] - inputs['input_ids'].shape[1]
    tps = tokens / elapsed
    times.append(tps)
    print(f'Run {i+1}: {tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s')

print(f'Average: {sum(times)/len(times):.1f} tok/s')
print(f'GPU memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB')
```

---

## Related Work (Position Against)

- **Petals** (2022) — Distributed inference across volunteer machines. Different problem (multi-node), we're single-node optimization.
- **Exo** (2024) — Distributed across consumer devices. Same distinction.
- **vLLM** — Production serving with PagedAttention. Our techniques are additive on top of vLLM.
- **TEAL paper** — Benchmarks sparsity alone. We add KV cache compression.
- **TurboQuant paper** — Benchmarks KV cache alone. We add activation sparsity.
- **Nobody has combined TEAL + TurboQuant** — that's the gap.
