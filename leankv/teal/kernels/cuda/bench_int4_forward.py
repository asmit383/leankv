"""End-to-end decode tok/s with the fused sparse-int4 GEMV, on a real model.

Quantizes every attention/MLP projection to int4 (symmetric, group-wise G=128),
runs batch-1 decode through the hand-written kernel, compares to PyTorch dense.
fp16 weights are kept for prefill fallback (this is a speed benchmark, not a
memory benchmark).
"""
import warnings, os, logging, time
warnings.filterwarnings("ignore")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers; transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
from torch.utils.cpp_extension import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
ap.add_argument("--ntok", type=int, default=128)
ap.add_argument("--prompt", default="Explain memory-bound GPU kernels in detail:")
args = ap.parse_args()
DEV, GS = "cuda", 128

ext = load(name="sparse_int4_ext", sources=["sparse_int4_ext.cu"],
           extra_cuda_cflags=["-O3", "-arch=sm_89"], verbose=False)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(DEV).eval()

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
layers = [(n, m) for n, m in model.named_modules()
          if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in TARGETS
          and m.bias is None and m.out_features % 8 == 0 and m.in_features % GS == 0]

@torch.no_grad()
def quantize_int4(W):                              # W [out,in] fp16 -> packed[in,out/2] u8, scales[in/G,out] fp16
    out, inn = W.shape
    Wg = W.float().reshape(out, inn // GS, GS)
    scale = (Wg.abs().amax(2, keepdim=True) / 7).clamp_min(1e-8)          # [out,nG,1]
    q = (Wg / scale).round().clamp(-8, 7).reshape(out, inn)              # [out,in], -8..7
    u = (q + 8).to(torch.uint8).t().contiguous()                        # [in,out]
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()              # [in,out/2]
    scales = scale.squeeze(2).t().contiguous().half()                   # [nG,out]
    return packed.to(DEV), scales.to(DEV)

@torch.no_grad()
def install(mod, thr):
    W = mod.weight.data
    Wq, sc = quantize_int4(W)
    def fwd(x):
        if x.shape[0] == 1 and x.shape[1] == 1:
            return ext.sparse_int4_gemv(x, Wq, sc, thr)
        return F.linear(x, W)                      # fp16 prefill fallback
    mod.forward = fwd

@torch.no_grad()
def bench(label, note=""):
    ids = tok(args.prompt, return_tensors="pt").to(DEV)
    model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); t = time.time()
    out = model.generate(**ids, max_new_tokens=args.ntok, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t
    n = out.shape[1] - ids.input_ids.shape[1]
    print(f"{label:<34} {n/dt:6.1f} tok/s{note}")
    return n / dt

print(f"model: {args.model}   decoding {args.ntok} tokens, batch-1\n")
d = bench("PyTorch dense (fp16)")
for thr in [0.0, 0.1, 0.2]:
    for _, m in layers: install(m, thr)
    bench(f"sparse-int4 kernel, thr={thr}")
print(f"\n(dense fp16 = {d:.1f} tok/s. int4 weights = 4x fewer bytes; thr>0 adds sparsity skip."
      f"\n int4 quality is a separate axis — this is the speed measurement.)")
