"""Streaming chat/completion on the fused-int4 + CUDA-graph fast path.

Feel the speed: fused sparse-int4 GEMV (calibrated per-layer thresholds), decode
captured as a CUDA graph and replayed per token, streamed live to the terminal.

Usage:
  python3 chat.py --thresholds thresholds.json          # interactive
  echo "Once upon a time" | python3 chat.py --thresholds thresholds.json --once
"""
import warnings, os, logging, sys, time, json, argparse
warnings.filterwarnings("ignore")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import transformers; transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
from torch.utils.cpp_extension import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
ap.add_argument("--thresholds", default="thresholds_uniform.json")
ap.add_argument("--blockk", type=int, default=128)
ap.add_argument("--maxtok", type=int, default=200)
ap.add_argument("--once", action="store_true", help="read one prompt from stdin and exit")
args = ap.parse_args()
DEV, GS, MAXLEN = "cuda", 128, 2048

TH = json.load(open(args.thresholds)) if os.path.exists(args.thresholds) else {}
def tget(i, k): return TH.get(f"{i}_{k}", 0.0)

print("loading model + int4 kernel ...", file=sys.stderr)
ext = load(name=f"chat_gs_bk{args.blockk}", sources=["sparse_int4_ext_gs.cu"],
           extra_cuda_cflags=["-O3", "-arch=sm_89", f"-DBLOCK_K={args.blockk}"], verbose=False)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(DEV).eval()
def dec(x): return x.shape[0] == 1 and x.shape[1] == 1

@torch.no_grad()
def quantize_int4(W):
    out, inn = W.shape
    Wg = W.float().reshape(out, inn // GS, GS)
    sc = (Wg.abs().amax(2, keepdim=True) / 7).clamp_min(1e-8)
    q = (Wg / sc).round().clamp(-8, 7).reshape(out, inn)
    u = (q + 8).to(torch.uint8).t().contiguous()
    return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous().to(DEV), sc.squeeze(2).t().contiguous().half().to(DEV)

@torch.no_grad()
def single(mod, tv):
    Wp, Sc = quantize_int4(mod.weight); w = mod.weight.data
    mod.forward = lambda x, Wp=Wp, Sc=Sc, w=w, tv=tv: ext.sparse_int4_gemv(x, Wp, Sc, tv) if dec(x) else F.linear(x, w)

@torch.no_grad()
def install(model):
    for i, L in enumerate(model.model.layers):
        at, mlp = L.self_attn, L.mlp
        Wp, Sc = quantize_int4(torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0))
        st = {"W": Wp, "S": Sc, "nq": at.q_proj.out_features, "nk": at.k_proj.out_features, "k": None, "v": None,
              "t": tget(i, "qkv"), "wq": at.q_proj.weight.data, "wk": at.k_proj.weight.data, "wv": at.v_proj.weight.data}
        def qf(x, st=st):
            if dec(x):
                y = ext.sparse_int4_gemv(x, st["W"], st["S"], st["t"]); nq, nk = st["nq"], st["nk"]
                st["k"] = y[..., nq:nq + nk].contiguous(); st["v"] = y[..., nq + nk:].contiguous()
                return y[..., :nq].contiguous()
            return F.linear(x, st["wq"])
        at.q_proj.forward = qf
        at.k_proj.forward = (lambda x, st=st: st["k"] if dec(x) else F.linear(x, st["wk"]))
        at.v_proj.forward = (lambda x, st=st: st["v"] if dec(x) else F.linear(x, st["wv"]))
        single(at.o_proj, tget(i, "o"))
        Wp2, Sc2 = quantize_int4(torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0))
        gt = {"W": Wp2, "S": Sc2, "ng": mlp.gate_proj.out_features, "u": None,
              "t": tget(i, "gateup"), "wg": mlp.gate_proj.weight.data, "wu": mlp.up_proj.weight.data}
        def gf(x, gt=gt):
            if dec(x):
                y = ext.sparse_int4_gemv(x, gt["W"], gt["S"], gt["t"]); ng = gt["ng"]
                gt["u"] = y[..., ng:].contiguous(); return y[..., :ng].contiguous()
            return F.linear(x, gt["wg"])
        mlp.gate_proj.forward = gf
        mlp.up_proj.forward = (lambda x, gt=gt: gt["u"] if dec(x) else F.linear(x, gt["wu"]))
        single(mlp.down_proj, tget(i, "down"))

install(model)
cfg = model.config
cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=MAXLEN, device=DEV, dtype=torch.float16)
si = torch.zeros((1, 1), dtype=torch.long, device=DEV)
sp = torch.zeros((1,), dtype=torch.long, device=DEV)
pid = torch.zeros((1, 1), dtype=torch.long, device=DEV)
def step(): return model(si, position_ids=pid, cache_position=sp, past_key_values=cache, use_cache=True).logits

# capture the decode graph once (warmup on a side stream, then capture)
print("capturing CUDA graph ...", file=sys.stderr)
warm = tok("hello world", return_tensors="pt").input_ids.to(DEV)
with torch.no_grad():
    model(warm, position_ids=torch.arange(warm.shape[1], device=DEV).unsqueeze(0),
          cache_position=torch.arange(warm.shape[1], device=DEV), past_key_values=cache, use_cache=True)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), torch.no_grad():
    for i in range(3):
        si.fill_(1); sp.fill_(warm.shape[1] + i); pid.fill_(warm.shape[1] + i); step()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
si.fill_(1); sp.fill_(0); pid.fill_(0)
with torch.cuda.graph(g), torch.no_grad():
    so = step()

messages = []  # conversation history (multi-turn)

@torch.no_grad()
def generate(user_msg):
    messages.append({"role": "user", "content": user_msg})
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(DEV)
    if ids.shape[1] >= MAXLEN - args.maxtok:      # conversation too long → start fresh
        del messages[:-1]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(DEV)
    cache.reset()
    P = ids.shape[1]
    ar = torch.arange(P, device=DEV)
    nxt = model(ids, position_ids=ar.unsqueeze(0), cache_position=ar,
                past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
    out, printed = [], 0
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(args.maxtok):
        tid = nxt.item()
        if tid == tok.eos_token_id: break
        out.append(tid)
        txt = tok.decode(out)
        sys.stdout.write(txt[printed:]); sys.stdout.flush(); printed = len(txt)
        si.copy_(nxt); sp.fill_(P + i); pid.fill_(P + i); g.replay()
        nxt = so[:, -1].argmax(-1, keepdim=True)
    dt = time.time() - t0
    messages.append({"role": "assistant", "content": tok.decode(out, skip_special_tokens=True)})
    print(f"\n\033[90m[{len(out)} tokens · {len(out)/dt:.1f} tok/s]\033[0m")

if args.once:
    generate(sys.stdin.read().strip()); sys.exit(0)

print(f"\n{args.model.split('/')[-1]} · fused int4 + CUDA graph. Chat away (Ctrl-C to quit).\n")
while True:
    try:
        p = input("\033[1m>>> \033[0m")
    except (EOFError, KeyboardInterrupt):
        print(); break
    if p.strip():
        generate(p)
        print()
