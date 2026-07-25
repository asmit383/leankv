"""Greedy TEAL-style calibration over the 4 shared input points per layer.

Uniform calibration prunes 40% at EVERY projection — including sensitive ones —
which costs perplexity. This allocates a global (param-weighted) sparsity budget
greedily: it repeatedly adds sparsity to the point with the lowest marginal
damage-per-byte, so tolerant points (small activations, large fan-out) get more
sparsity and sensitive points get less. Same global sparsity, lower error.

Points (they share an input in the fused kernel, so one threshold each):
  qkv  = input to q/k/v   |  o = input to o_proj
  gateup = input to gate/up |  down = input to down_proj

Damage-per-byte derivation: pruning Δs of a point's input channels skips
Δs·in·out weights (bytes ∝ params = in·out) and zeroes Δs·in channels of energy
~thr² each, so damage/byte = thr² / out. Prune where thr²/out is smallest.
Global budget is param-weighted: Σ(in·out·s) / Σ(in·out) = target.
"""
import warnings, os, json, argparse
warnings.filterwarnings("ignore"); os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers; transformers.logging.set_verbosity_error()

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
ap.add_argument("--sparsity", type=float, default=0.40)
ap.add_argument("--samples", type=int, default=40)
ap.add_argument("--out", default="thresholds_greedy.json")
args = ap.parse_args()
DEV = "cuda"

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(DEV).eval()

try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t) > 200][:args.samples]
    print(f"calibration: {len(texts)} wikitext passages")
except Exception as e:
    print(f"dataset failed ({e}); generic text")
    texts = ["The transformer processes tokens through attention and MLP layers, "
             "streaming weights from memory for each generated token."] * args.samples

store = {}
def mk(key):
    def h(mod, inp):
        x = inp[0].detach().abs().flatten().float()
        if x.numel() > 4000: x = x[torch.randint(0, x.numel(), (4000,), device=x.device)]
        store.setdefault(key, []).append(x.cpu())
    return h

# per point: params (in*out, for the global budget) and out (for damage/byte)
meta = {}  # key -> (params, out)
handles = []
for i, L in enumerate(model.model.layers):
    at, mlp = L.self_attn, L.mlp
    qin, qout = at.q_proj.in_features, at.q_proj.out_features + at.k_proj.out_features + at.v_proj.out_features
    oin, oout = at.o_proj.in_features, at.o_proj.out_features
    gin, gout = mlp.gate_proj.in_features, mlp.gate_proj.out_features + mlp.up_proj.out_features
    din, dout = mlp.down_proj.in_features, mlp.down_proj.out_features
    for mod, key, (pin, pout) in [(at.q_proj, f"{i}_qkv", (qin, qout)), (at.o_proj, f"{i}_o", (oin, oout)),
                                  (mlp.gate_proj, f"{i}_gateup", (gin, gout)), (mlp.down_proj, f"{i}_down", (din, dout))]:
        meta[key] = (pin * pout, pout)
        handles.append(mod.register_forward_pre_hook(mk(key)))

with torch.no_grad():
    for j, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=512).to(DEV)
        model(**ids)
        if (j + 1) % 10 == 0: print(f"  {j+1}/{len(texts)}")
for h in handles: h.remove()

sorted_abs = {k: torch.cat(v).sort().values for k, v in store.items()}
keys = list(sorted_abs.keys())
def thr_at(k, sp):
    a = sorted_abs[k]; return a[min(int(sp * len(a)), len(a) - 1)].item()

# greedy allocation
s = {k: 0.0 for k in keys}
Wtot = sum(meta[k][0] for k in keys)
STEP = 0.005
gsp = lambda: sum(meta[k][0] * s[k] for k in keys) / Wtot
while gsp() < args.sparsity:
    best, bestm = None, 1e30
    for k in keys:
        if s[k] >= 0.9: continue
        t = thr_at(k, s[k]); m = (t * t) / meta[k][1]   # thr^2 / out
        if m < bestm: bestm, best = m, k
    if best is None: break
    s[best] += STEP

thresholds = {k: thr_at(k, s[k]) for k in keys}
json.dump(thresholds, open(args.out, "w"))
import statistics as st
print(f"\ngreedy -> {args.out}  (global param-weighted sparsity {gsp():.0%})")
for typ in ["qkv", "o", "gateup", "down"]:
    avg = st.mean([s[k] for k in keys if k.endswith(typ)])
    print(f"  {typ:<7} avg sparsity {avg:.0%}")
