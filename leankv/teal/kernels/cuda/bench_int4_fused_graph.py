"""Fused int4 + CUDA graph: the combined push.

Fuses q/k/v and gate/up into single graph-safe int4 GEMVs (tunable split-K), then
captures the decode step as a CUDA graph and replays it. Now that fusion + BLOCK_K
tuning made decode launch/overhead-bound (not weight-bound), graphs should pay off.
"""
import warnings, os, logging, time
warnings.filterwarnings("ignore")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import transformers; transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
from torch.utils.cpp_extension import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
ap.add_argument("--ntok", type=int, default=128)
ap.add_argument("--thr", type=float, default=0.0)
ap.add_argument("--blockk", type=int, default=128)
ap.add_argument("--thresholds", default="", help="calibrated thresholds.json (per-layer)")
ap.add_argument("--ext", default="sparse_int4_ext_gs", help="ext source (gs=two-pass safe, lop3=atomic)")
ap.add_argument("--prompt", default="Explain memory-bound GPU kernels in detail:")
args = ap.parse_args()
DEV, GS = "cuda", 128
import json
TH = json.load(open(args.thresholds)) if args.thresholds else {}
def tget(i, kind): return TH.get(f"{i}_{kind}", args.thr)

ext = load(name=f"{args.ext}_bk{args.blockk}", sources=[args.ext + ".cu"],
           extra_cuda_cflags=["-O3", "-arch=sm_89", f"-DBLOCK_K={args.blockk}"], verbose=False)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(DEV).eval()
thr = args.thr
def dec(x): return x.shape[0] == 1 and x.shape[1] == 1

@torch.no_grad()
def quantize_int4(W):
    out, inn = W.shape
    Wg = W.float().reshape(out, inn // GS, GS)
    scale = (Wg.abs().amax(2, keepdim=True) / 7).clamp_min(1e-8)
    q = (Wg / scale).round().clamp(-8, 7).reshape(out, inn)
    u = (q + 8).to(torch.uint8).t().contiguous()
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
    return packed.to(DEV), scale.squeeze(2).t().contiguous().half().to(DEV)

@torch.no_grad()
def install_single(mod, tv):
    Wp, Sc = quantize_int4(mod.weight); w = mod.weight.data
    mod.forward = lambda x, Wp=Wp, Sc=Sc, w=w, tv=tv: ext.sparse_int4_gemv(x, Wp, Sc, tv) if dec(x) else F.linear(x, w)

@torch.no_grad()
def install_fused(model):
    for i, layer in enumerate(model.model.layers):
        at, mlp = layer.self_attn, layer.mlp
        Wp, Sc = quantize_int4(torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0))
        st = {"W": Wp, "S": Sc, "nq": at.q_proj.out_features, "nk": at.k_proj.out_features, "k": None, "v": None,
              "t": tget(i, "qkv"), "wq": at.q_proj.weight.data, "wk": at.k_proj.weight.data, "wv": at.v_proj.weight.data}
        def q_fwd(x, st=st):
            if dec(x):
                y = ext.sparse_int4_gemv(x, st["W"], st["S"], st["t"]); nq, nk = st["nq"], st["nk"]
                st["k"] = y[..., nq:nq + nk].contiguous(); st["v"] = y[..., nq + nk:].contiguous()
                return y[..., :nq].contiguous()
            return F.linear(x, st["wq"])
        at.q_proj.forward = q_fwd
        at.k_proj.forward = (lambda x, st=st: st["k"] if dec(x) else F.linear(x, st["wk"]))
        at.v_proj.forward = (lambda x, st=st: st["v"] if dec(x) else F.linear(x, st["wv"]))
        install_single(at.o_proj, tget(i, "o"))
        Wp2, Sc2 = quantize_int4(torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0))
        gt = {"W": Wp2, "S": Sc2, "ng": mlp.gate_proj.out_features, "u": None,
              "t": tget(i, "gateup"), "wg": mlp.gate_proj.weight.data, "wu": mlp.up_proj.weight.data}
        def gate_fwd(x, gt=gt):
            if dec(x):
                y = ext.sparse_int4_gemv(x, gt["W"], gt["S"], gt["t"]); ng = gt["ng"]
                gt["u"] = y[..., ng:].contiguous(); return y[..., :ng].contiguous()
            return F.linear(x, gt["wg"])
        mlp.gate_proj.forward = gate_fwd
        mlp.up_proj.forward = (lambda x, gt=gt: gt["u"] if dec(x) else F.linear(x, gt["wu"]))
        install_single(mlp.down_proj, tget(i, "down"))

install_fused(model)
cfg = model.config
prompt_ids = tok(args.prompt, return_tensors="pt").input_ids.to(DEV)
P = prompt_ids.shape[1]; MAXLEN = P + args.ntok + 8

@torch.no_grad()
def eager():
    ids = tok(args.prompt, return_tensors="pt").to(DEV)
    model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); t = time.time()
    out = model.generate(**ids, max_new_tokens=args.ntok, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return args.ntok / (time.time() - t), out[0][P:].tolist()

@torch.no_grad()
def graphed():
    cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=MAXLEN, device=DEV, dtype=torch.float16)
    pos = torch.arange(P, device=DEV)
    cur = model(prompt_ids, position_ids=pos.unsqueeze(0), cache_position=pos,
                past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
    si = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    sp = torch.zeros((1,), dtype=torch.long, device=DEV)
    pid = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    def step():
        return model(si, position_ids=pid, cache_position=sp, past_key_values=cache, use_cache=True).logits
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            si.copy_(cur); sp.fill_(P + i); pid.fill_(P + i); step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    si.copy_(cur); sp.fill_(P); pid.fill_(P)
    with torch.cuda.graph(g):
        so = step()
    cache.reset()
    cur = model(prompt_ids, position_ids=pos.unsqueeze(0), cache_position=pos,
                past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize(); t = time.time()
    toks = [cur.item()]; nxt = cur
    for i in range(args.ntok):
        si.copy_(nxt); sp.fill_(P + i); pid.fill_(P + i); g.replay()
        nxt = so[:, -1].argmax(-1, keepdim=True); toks.append(nxt.item())
    torch.cuda.synchronize()
    return args.ntok / (time.time() - t), toks[:args.ntok]

print(f"model: {args.model}  fused int4  BLOCK_K={args.blockk}  thr={thr}  ntok={args.ntok}\n")
e, et = eager();   print(f"eager   : {e:6.1f} tok/s")
gr, gt = graphed(); print(f"graphed : {gr:6.1f} tok/s   ({gr/e:.2f}x)")
print(f"\ncorrectness: {sum(1 for a,b in zip(et,gt) if a==b)}/{len(et)} tokens match eager")
print(f"graphed text: {tok.decode(gt[:36])!r}")
