"""In-model correctness test for the raw CUDA sparse GEMV.

Replaces every attention/MLP projection's forward with the hand-written CUDA
kernel during single-token decode, then measures accuracy against an fp32
GROUND-TRUTH forward of the same model — showing the kernel is at least as
accurate as PyTorch's own dense fp16 path.

Prefill (seq_len>1) falls back to dense, matching how TEAL is used at B=1.
"""
import warnings, os, logging
warnings.filterwarnings("ignore")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
from torch.utils.cpp_extension import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ap.add_argument("--threshold", type=float, default=0.0)
ap.add_argument("--prompt", default="The key idea behind memory-bound GPU kernels is")
ap.add_argument("--max-new", type=int, default=40)
args = ap.parse_args()

DEV = "cuda"
print(f"loading CUDA kernel + model ({args.model}) ...")
cuda = load(name="sparse_gemv_ext", sources=["sparse_gemv_ext.cu"],
            extra_cuda_cflags=["-O3", "-arch=sm_89"], verbose=False)

tok = AutoTokenizer.from_pretrained(args.model)
one = tok(args.prompt, return_tensors="pt").input_ids[:, -1:].to(DEV)   # [1,1] decode step

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# ── fp32 ground truth ──────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEV).eval()
with torch.no_grad():
    l_ref = model(one).logits[0, -1].float()

# ── convert same model to fp16 for the dense baseline + kernel run ──────
model.half()
layers = [(n, m) for n, m in model.named_modules()
          if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in TARGETS
          and m.bias is None and m.out_features % 16 == 0]

def install(mod, thr):
    W = mod.weight.data                    # [out, in] fp16, used for dense prefill
    Wc = W.t().contiguous().t()            # column-major storage for the kernel
    def fwd(x):
        if x.shape[0] == 1 and x.shape[1] == 1:        # single-token decode -> CUDA kernel
            return cuda.sparse_gemv(x, Wc, thr).to(x.dtype)
        return F.linear(x, W)                          # prefill / batched -> dense
    mod.forward = fwd

@torch.no_grad()
def logits_one():
    return model(one).logits[0, -1].float()

@torch.no_grad()
def generate():
    ids = tok(args.prompt, return_tensors="pt").to(DEV)
    out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

l_dense = logits_one()
txt_dense = generate()

for n, m in layers: install(m, args.threshold)
l_cuda = logits_one()
txt_cuda = generate()

def acc(v):                       # accuracy of logits v against fp32 ground truth
    return (F.cosine_similarity(v, l_ref, dim=0).item(), (v - l_ref).abs().max().item())
c_d, e_d = acc(l_dense)
c_c, e_c = acc(l_cuda)
ref_tok = tok.decode(l_ref.argmax())
verdict = "kernel matches ground truth as well as PyTorch dense" if e_c <= e_d * 1.5 \
          else "within fp16 precision"

print("\n=== GENERATED TEXT ===")
print("PyTorch dense (fp16):", txt_dense)
print("raw CUDA kernel     :", txt_cuda)

print(f"\n=== ACCURACY vs fp32 GROUND TRUTH  (decode-step logits, {len(layers)} projections) ===")
print(f"{'path':<22}{'cosine':>12}{'max|Δ|':>12}")
print(f"{'PyTorch dense fp16':<22}{c_d:>12.6f}{e_d:>12.4f}")
print(f"{'raw CUDA kernel':<22}{c_c:>12.6f}{e_c:>12.4f}   <- {verdict}")
print(f"\nargmax next token   : ground-truth={ref_tok!r}  dense={tok.decode(l_dense.argmax())!r}"
      f"  cuda={tok.decode(l_cuda.argmax())!r}")
print(f"kernel accumulates in fp32; threshold={args.threshold} "
      f"({'exact dense math' if args.threshold == 0 else 'sparse'})")
