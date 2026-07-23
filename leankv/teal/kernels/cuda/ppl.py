"""Perplexity of int4 + calibrated-sparsity vs fp16 baseline (WikiText-2).

The decode kernels only fire at seq_len=1, so to measure quality on full sequences
we apply the numerically-equivalent op: dequantize the int4 weights IN PLACE (one
model copy — no OOM), then optionally sparsify the input (|x|<=thr -> 0). This is
exactly what the fused sparse-int4 kernel computes, over any sequence length.
"""
import warnings, os, json, argparse
warnings.filterwarnings("ignore"); os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers; transformers.logging.set_verbosity_error()

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
ap.add_argument("--thresholds", default="thresholds.json")
ap.add_argument("--maxlen", type=int, default=1024)
args = ap.parse_args()
DEV, GS = "cuda", 128
TH = json.load(open(args.thresholds)) if os.path.exists(args.thresholds) else {}

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(DEV).eval()

from datasets import load_dataset
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
enc = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
S = enc.size(1); stride = args.maxlen // 2

@torch.no_grad()
def ppl():
    nlls, ntok, prev = [], 0, 0
    for b in range(0, S, stride):
        e = min(b + args.maxlen, S); trg = e - prev
        ids = enc[:, b:e].to(DEV); tgt = ids.clone(); tgt[:, :-trg] = -100
        nlls.append(model(ids, labels=tgt).loss.float() * trg); ntok += trg; prev = e
        if e == S: break
    return torch.exp(torch.stack(nlls).sum() / ntok).item()

@torch.no_grad()
def dequant_(W):
    out, inn = W.shape
    Wg = W.float().reshape(out, inn // GS, GS)
    sc = (Wg.abs().amax(2, keepdim=True) / 7).clamp_min(1e-8)
    q = (Wg / sc).round().clamp(-8, 7)
    return (q * sc).reshape(out, inn).half()

projs = []
for i, L in enumerate(model.model.layers):
    for mod, kind in [(L.self_attn.q_proj, "qkv"), (L.self_attn.k_proj, "qkv"), (L.self_attn.v_proj, "qkv"),
                      (L.self_attn.o_proj, "o"), (L.mlp.gate_proj, "gateup"), (L.mlp.up_proj, "gateup"),
                      (L.mlp.down_proj, "down")]:
        projs.append((i, mod, kind))

print(f"WikiText-2 perplexity (maxlen={args.maxlen}):\n")
print(f"  {'fp16':<14} {ppl():.3f}")

# dequantize all weights IN PLACE (one copy — frees originals)
with torch.no_grad():
    for _, mod, _ in projs: mod.weight.data = dequant_(mod.weight.data)
torch.cuda.empty_cache()
print(f"  {'int4':<14} {ppl():.3f}")

# add calibrated sparsity on the input
with torch.no_grad():
    for i, mod, kind in projs:
        t = TH.get(f"{i}_{kind}", 0.0); w = mod.weight.data
        mod.forward = (lambda x, w=w, t=t: F.linear(torch.where(x.abs() > t, x, torch.zeros_like(x)), w)) if t > 0 \
            else (lambda x, w=w: F.linear(x, w))
print(f"  {'int4+sparse':<14} {ppl():.3f}   (calibrated {args.thresholds})")
