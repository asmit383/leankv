"""End-to-end decode tokens/s: PyTorch dense vs the raw CUDA sparse GEMV.

Measures single-stream (batch-1) decode throughput. Reports dense, kernel at
threshold=0 (pure overhead cost, full dense work), and kernel with sparsity
(rows skipped). Also prints achieved activation sparsity.
"""
import warnings, os, logging, time
warnings.filterwarnings("ignore")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers; transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
from torch.utils.cpp_extension import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ap.add_argument("--ntok", type=int, default=128)
ap.add_argument("--prompt", default="Explain memory-bound GPU kernels in detail:")
args = ap.parse_args()

DEV = "cuda"
cuda = load(name="sparse_gemv_ext", sources=["sparse_gemv_ext.cu"],
            extra_cuda_cflags=["-O3", "-arch=sm_89"], verbose=False)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(DEV).eval()

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
layers = [(n, m) for n, m in model.named_modules()
          if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in TARGETS
          and m.bias is None and m.out_features % 16 == 0]
stats = {"active": 0, "total": 0}

def install(mod, thr):
    W = mod.weight.data
    Wc = W.t().contiguous().t()      # column-major storage, logical [out,in]
    mod.weight.data = Wc             # replace in place → frees original, one copy (fits 7B)
    def fwd(x):                      # NO per-call sync in the hot path
        if x.shape[0] == 1 and x.shape[1] == 1:
            return cuda.sparse_gemv(x, Wc, thr)
        return F.linear(x, Wc)       # prefill / batched -> dense
    mod.forward = fwd

@torch.no_grad()
def bench(label, note=""):
    ids = tok(args.prompt, return_tensors="pt").to(DEV)
    model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    t = time.time()
    out = model.generate(**ids, max_new_tokens=args.ntok, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t
    n = out.shape[1] - ids.input_ids.shape[1]
    print(f"{label:<28} {n/dt:6.1f} tok/s{note}")
    return n / dt

@torch.no_grad()
def probe_sparsity(thr):                 # achieved sparsity on one real decode activation
    ids = tok(args.prompt, return_tensors="pt").to(DEV)
    h = {}
    tgt = model.model.layers[0].mlp.gate_proj
    def hook(m, inp): h["x"] = inp[0]
    hd = tgt.register_forward_pre_hook(hook)
    model(ids.input_ids[:, -1:])
    hd.remove()
    x = h["x"].reshape(-1)
    return 100.0 * (x.abs() <= thr).float().mean().item()

print(f"model: {args.model}   decoding {args.ntok} tokens, batch-1\n")
d = bench("PyTorch dense (fp16)")
for thr in [0.0, 0.1, 0.2, 0.35]:
    sp = probe_sparsity(thr) if thr > 0 else 0.0
    for _, m in layers: install(m, thr)
    bench(f"CUDA kernel, thr={thr}", note=f"   (~{sp:.0f}% sparse)")
print(f"\n(dense = {d:.1f} tok/s. kernel ties Triton at the kernel level; naive per-call"
      f"\n integration — torch.zeros + fp16 cast each call — is the remaining overhead.)")
