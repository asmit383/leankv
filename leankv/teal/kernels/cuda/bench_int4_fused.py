"""Fused int4 decode: fuse QKV and gate+up into single kernels.

q/k/v share one input+mask -> concatenate weights, one int4 GEMV, split output.
gate/up likewise. Cuts 7 projection launches/layer to 4 and makes outputs bigger
(better occupancy). Uses the fast single-pass int4 ext (eager; graphs don't help).
Measures decode tok/s vs dense and reports correctness (coherent text).
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
ap.add_argument("--thr", type=float, default=0.0)
ap.add_argument("--prompt", default="Explain memory-bound GPU kernels in detail:")
ap.add_argument("--ext", default="sparse_int4_ext", help="ext source (without .cu)")
ap.add_argument("--blockk", type=int, default=0, help="override BLOCK_K (split-K tile)")
args = ap.parse_args()
DEV, GS = "cuda", 128

_flags = ["-O3", "-arch=sm_89"]
_name = args.ext
if args.blockk:
    _flags.append(f"-DBLOCK_K={args.blockk}"); _name = f"{args.ext}_bk{args.blockk}"
ext = load(name=_name, sources=[args.ext + ".cu"], extra_cuda_cflags=_flags, verbose=False)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(DEV).eval()

@torch.no_grad()
def quantize_int4(W):
    out, inn = W.shape
    Wg = W.float().reshape(out, inn // GS, GS)
    scale = (Wg.abs().amax(2, keepdim=True) / 7).clamp_min(1e-8)
    q = (Wg / scale).round().clamp(-8, 7).reshape(out, inn)
    u = (q + 8).to(torch.uint8).t().contiguous()
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
    scales = scale.squeeze(2).t().contiguous().half()
    return packed.to(DEV), scales.to(DEV)

thr = args.thr
def dec(x): return x.shape[0] == 1 and x.shape[1] == 1     # single-token decode?

@torch.no_grad()
def install_fused(model):
    for layer in model.model.layers:
        at, mlp = layer.self_attn, layer.mlp
        # ---- fused QKV ----
        Wqkv = torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0)
        Wp, Sc = quantize_int4(Wqkv)
        nq, nk = at.q_proj.out_features, at.k_proj.out_features
        st = {"W": Wp, "S": Sc, "nq": nq, "nk": nk, "k": None, "v": None,
              "wq": at.q_proj.weight.data, "wk": at.k_proj.weight.data, "wv": at.v_proj.weight.data}
        def q_fwd(x, st=st):
            if dec(x):
                y = ext.sparse_int4_gemv(x, st["W"], st["S"], thr)      # [1,1,nq+nk+nv]
                nq, nk = st["nq"], st["nk"]
                st["k"] = y[..., nq:nq + nk].contiguous()
                st["v"] = y[..., nq + nk:].contiguous()
                return y[..., :nq].contiguous()
            return F.linear(x, st["wq"])
        at.q_proj.forward = q_fwd
        at.k_proj.forward = (lambda x, st=st: st["k"] if dec(x) else F.linear(x, st["wk"]))
        at.v_proj.forward = (lambda x, st=st: st["v"] if dec(x) else F.linear(x, st["wv"]))
        # ---- o_proj (single) ----
        install_single(at.o_proj)
        # ---- fused gate+up ----
        Wgu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0)
        Wp2, Sc2 = quantize_int4(Wgu)
        ng = mlp.gate_proj.out_features
        gt = {"W": Wp2, "S": Sc2, "ng": ng, "u": None,
              "wg": mlp.gate_proj.weight.data, "wu": mlp.up_proj.weight.data}
        def gate_fwd(x, gt=gt):
            if dec(x):
                y = ext.sparse_int4_gemv(x, gt["W"], gt["S"], thr)      # [1,1,2*ng]
                ng = gt["ng"]; gt["u"] = y[..., ng:].contiguous()
                return y[..., :ng].contiguous()
            return F.linear(x, gt["wg"])
        mlp.gate_proj.forward = gate_fwd
        mlp.up_proj.forward = (lambda x, gt=gt: gt["u"] if dec(x) else F.linear(x, gt["wu"]))
        # ---- down_proj (single) ----
        install_single(mlp.down_proj)

@torch.no_grad()
def install_single(mod):
    Wp, Sc = quantize_int4(mod.weight)
    w = mod.weight.data
    mod.forward = lambda x, Wp=Wp, Sc=Sc, w=w: ext.sparse_int4_gemv(x, Wp, Sc, thr) if dec(x) else F.linear(x, w)

@torch.no_grad()
def bench(label):
    ids = tok(args.prompt, return_tensors="pt").to(DEV)
    model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); t = time.time()
    out = model.generate(**ids, max_new_tokens=args.ntok, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t
    txt = tok.decode(out[0][ids.input_ids.shape[1]:][:40], skip_special_tokens=True)
    print(f"{label:<22} {args.ntok/dt:6.1f} tok/s"); return args.ntok/dt, txt

print(f"model: {args.model}  int4 thr={thr}  decoding {args.ntok} tokens\n")
d, dtxt = bench("dense fp16")
install_fused(model)
f, ftxt = bench("fused int4")
print(f"\nspeedup vs dense: {f/d:.2f}x")
print(f"dense: {dtxt!r}")
print(f"fused: {ftxt!r}")
