"""CUDA-graph decode for the fused sparse-int4 kernel.

Eliminates per-call launch + Python overhead by capturing one decode step as a
CUDA graph (static KV cache + fixed input/position buffers) and replaying it per
token. Compares eager vs graphed tok/s on the same int4-quantized model.
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
ap.add_argument("--thr", type=float, default=0.2)
ap.add_argument("--prompt", default="Explain memory-bound GPU kernels in detail:")
args = ap.parse_args()
DEV, GS = "cuda", 128

ext = load(name="sparse_int4_ext", sources=["sparse_int4_ext.cu"],
           extra_cuda_cflags=["-O3", "-arch=sm_89"], verbose=False)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                             attn_implementation="sdpa").to(DEV).eval()

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
layers = [(n, m) for n, m in model.named_modules()
          if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in TARGETS
          and m.bias is None and m.out_features % 8 == 0 and m.in_features % GS == 0]

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

@torch.no_grad()
def install(mod, thr):
    W = mod.weight.data
    Wq, sc = quantize_int4(W)
    def fwd(x):
        if x.shape[0] == 1 and x.shape[1] == 1:
            return ext.sparse_int4_gemv(x, Wq, sc, thr)
        return F.linear(x, W)
    mod.forward = fwd

for _, m in layers: install(m, args.thr)
cfg = model.config
prompt_ids = tok(args.prompt, return_tensors="pt").input_ids.to(DEV)
P = prompt_ids.shape[1]
MAXLEN = P + args.ntok + 8

# ── eager decode (baseline) ─────────────────────────────────────────────
@torch.no_grad()
def eager():
    ids = tok(args.prompt, return_tensors="pt").to(DEV)
    model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); t = time.time()
    out = model.generate(**ids, max_new_tokens=args.ntok, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    new = out[0][ids.input_ids.shape[1]:]
    return args.ntok / (time.time() - t), new.tolist()

# ── graphed decode ──────────────────────────────────────────────────────
@torch.no_grad()
def graphed():
    cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=MAXLEN, device=DEV, dtype=torch.float16)
    # prefill
    pos = torch.arange(P, device=DEV)
    out = model(prompt_ids, past_key_values=cache, cache_position=pos, use_cache=True)
    cur = out.logits[:, -1].argmax(-1, keepdim=True)

    static_in = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    static_pos = torch.zeros((1,), dtype=torch.long, device=DEV)      # cache_position
    static_pid = torch.zeros((1, 1), dtype=torch.long, device=DEV)    # position_ids (for RoPE)

    def step():
        return model(static_in, position_ids=static_pid, cache_position=static_pos,
                     past_key_values=cache, use_cache=True).logits

    # warmup on a side stream (required before capture). Pollutes slots P..P+2 with
    # cur's KV — harmless: each real step overwrites its slot before reading, and
    # future slots are masked. Cache then flows continuously through capture+replay.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            static_in.copy_(cur); static_pos.fill_(P + i); static_pid.fill_(P + i)
            step()
    torch.cuda.current_stream().wait_stream(s)

    # capture at pos P (input = cur, which lives at position P) -> predicts token P+1
    g = torch.cuda.CUDAGraph()
    static_in.copy_(cur); static_pos.fill_(P); static_pid.fill_(P)
    with torch.cuda.graph(g):
        static_out = step()

    # timed loop — NO reset; cache flows continuously. seq = [cur, capture out, replays...]
    torch.cuda.synchronize(); t = time.time()
    toks = [cur.item()]
    nxt = static_out[:, -1].argmax(-1, keepdim=True)          # token at P+1 from capture
    toks.append(nxt.item())
    for i in range(2, args.ntok):
        static_in.copy_(nxt); static_pos.fill_(P + i - 1); static_pid.fill_(P + i - 1)
        g.replay()
        nxt = static_out[:, -1].argmax(-1, keepdim=True)
        toks.append(nxt.item())
    torch.cuda.synchronize()
    return args.ntok / (time.time() - t), toks

print(f"model: {args.model}  int4 thr={args.thr}  decoding {args.ntok} tokens\n")
e, e_toks = eager();   print(f"eager   decode : {e:6.1f} tok/s")
try:
    gr, g_toks = graphed()
    print(f"graphed decode : {gr:6.1f} tok/s   ({gr/e:.2f}x)")
    match = sum(1 for a, b in zip(e_toks, g_toks) if a == b)
    print(f"\ncorrectness: {match}/{len(e_toks)} tokens match eager")
    print(f"eager  : {tok.decode(e_toks[:40])!r}")
    print(f"graphed: {tok.decode(g_toks[:40])!r}")
except Exception as ex:
    import traceback; traceback.print_exc()
    print(f"graph capture failed: {type(ex).__name__}: {ex}")
