"""Quick TEAL calibration: per-layer, per-input-point activation thresholds.

For each projection input (q/k/v share the attn input; gate/up share the mlp
input; o and down have their own), collect |activation| magnitudes on calibration
text and set the threshold to the target-sparsity percentile. Saves thresholds.json
keyed "<layer>_<kind>" (kind in {qkv, o, gateup, down}).
"""
import warnings, os, json, argparse
warnings.filterwarnings("ignore"); os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers; transformers.logging.set_verbosity_error()

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
ap.add_argument("--sparsity", type=float, default=0.40)
ap.add_argument("--samples", type=int, default=40)
ap.add_argument("--out", default="thresholds.json")
args = ap.parse_args()
DEV = "cuda"

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(DEV).eval()

# calibration text (wikitext, fallback to generic passages)
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t) > 200][:args.samples]
    print(f"calibration: {len(texts)} wikitext passages")
except Exception as e:
    print(f"dataset load failed ({e}); using generic passages")
    base = ["The transformer architecture relies on self-attention to weigh the importance of "
            "different tokens in a sequence, enabling parallel processing of language.",
            "Memory bandwidth is the primary bottleneck for batch-size-one inference, where the "
            "entire weight matrix must be streamed from HBM for each generated token.",
            "Quantization reduces the number of bits used to represent weights, cutting memory "
            "traffic at the cost of some numerical precision in the dequantized values."]
    texts = (base * ((args.samples // len(base)) + 1))[:args.samples]

store = {}
def mk(key):
    def h(mod, inp):
        x = inp[0].detach().abs().flatten().float()
        if x.numel() > 4000:
            x = x[torch.randint(0, x.numel(), (4000,), device=x.device)]
        store.setdefault(key, []).append(x)
    return h

handles = []
for i, layer in enumerate(model.model.layers):
    handles.append(layer.self_attn.q_proj.register_forward_pre_hook(mk(f"{i}_qkv")))
    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(mk(f"{i}_o")))
    handles.append(layer.mlp.gate_proj.register_forward_pre_hook(mk(f"{i}_gateup")))
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(mk(f"{i}_down")))

with torch.no_grad():
    for j, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=512).to(DEV)
        model(**ids)
        if (j + 1) % 10 == 0: print(f"  {j+1}/{len(texts)}")
for h in handles: h.remove()

thr = {k: torch.quantile(torch.cat(v), args.sparsity).item() for k, v in store.items()}
json.dump(thr, open(args.out, "w"))
vals = list(thr.values())
print(f"saved {len(thr)} thresholds to {args.out}  (target sparsity {args.sparsity:.0%})")
print(f"threshold range: {min(vals):.4f} – {max(vals):.4f}")
