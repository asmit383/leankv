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
    Wc = W.t().contiguous().t()
    def fwd(x):
        if x.shape[0] == 1 and x.shape[1] == 1:
            xf = x.reshape(-1)
            stats["active"] += int((xf.abs() > thr).sum()); stats["total"] += xf.numel()
            return cuda.sparse_gemv(x, Wc, thr).to(x.dtype)
        return F.linear(x, W)
    mod.forward = fwd

@torch.no_grad()
def bench(label):
    ids = tok(args.prompt, return_tensors="pt").to(DEV)
    model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); stats["active"] = stats["total"] = 0
    t = time.time()
    out = model.generate(**ids, max_new_tokens=args.ntok, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t
    n = out.shape[1] - ids.input_ids.shape[1]
    sp = 100.0 * (1 - stats["active"] / max(stats["total"], 1)) if stats["total"] else 0.0
    print(f"{label:<28} {n/dt:6.1f} tok/s" + (f"   (sparsity {sp:.0f}%)" if stats["total"] else ""))
    return n / dt

print(f"model: {args.model}   decoding {args.ntok} tokens, batch-1\n")
d = bench("PyTorch dense (fp16)")
for _, m in layers: install(m, 0.0)
bench("CUDA kernel, threshold=0")
for _, m in layers: install(m, 0.05)
bench("CUDA kernel, sparse")
print(f"\n(dense baseline = {d:.1f} tok/s; kernel overhead vs dense is the gap at threshold=0)")
