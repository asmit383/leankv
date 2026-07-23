"""Head-to-head: raw CUDA sparse GEMV vs the Triton sparse GEMV, identical tensors.

Measures correctness (both vs dense reference), latency, and effective HBM
bandwidth in the batch-1 memory-bound decode regime.
"""
import time, torch
from torch.utils.cpp_extension import load

# Triton kernel under test (uploaded alongside this file).
from sparse_gemv_triton import sparse_gemv as triton_sparse_gemv, prepare_weight_for_sparse_gemv

cuda = load(name="sparse_gemv_ext", sources=["sparse_gemv_ext.cu"],
            extra_cuda_cflags=["-O3", "-arch=sm_89"], verbose=False)

DEV = "cuda"
THRESH = 0.5

def peak_bw_gbps():
    # NVIDIA L4: 24GB GDDR6, 300 GB/s (matches cudaDeviceProp in the standalone .cu).
    return 300.0

def make_inputs(N, K, sparsity):
    torch.manual_seed(0)
    x = torch.where(torch.rand(K, device=DEV) < sparsity,
                    torch.zeros(K, device=DEV),
                    1.0 + torch.rand(K, device=DEV)).half().reshape(1, 1, K)
    W = (torch.rand(N, K, device=DEV) * 0.02 - 0.01).half()
    Wc = prepare_weight_for_sparse_gemv(W)   # column-major storage, shared by both
    return x, W, Wc

def timed(fn, iters=200):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms

def run(N, K, sparsity):
    x, W, Wc = make_inputs(N, K, sparsity)
    active = int((x.abs() > THRESH).sum().item())

    # Dense reference (masked) for correctness.
    xm = torch.where(x.abs() > THRESH, x, torch.zeros_like(x)).reshape(K).float()
    ref = (xm @ W.float().t()).half()

    y_tri = triton_sparse_gemv(x, Wc, THRESH).reshape(N)
    y_cud = cuda.sparse_gemv(x, Wc, THRESH).reshape(N)

    def rel(a): return ((a.float() - ref.float()).norm() / (ref.float().norm() + 1e-6)).item()

    t_tri = timed(lambda: triton_sparse_gemv(x, Wc, THRESH))
    t_cud = timed(lambda: cuda.sparse_gemv(x, Wc, THRESH))

    wbytes = active * N * 2
    bw = lambda ms: (wbytes + K * 2 + N * 4) / (ms * 1e-3) / 1e9
    peak = peak_bw_gbps()

    print(f"\nN×K={N}×{K}  sparsity={100*(1-active/K):.0f}%  ({active}/{K} active)")
    print(f"  {'kernel':<8} {'ms':>8} {'GB/s':>8} {'bw%':>6} {'relerr':>10}")
    print(f"  {'triton':<8} {t_tri:>8.4f} {bw(t_tri):>8.1f} {100*bw(t_tri)/peak:>5.1f}% {rel(y_tri):>10.2e}")
    print(f"  {'cuda':<8} {t_cud:>8.4f} {bw(t_cud):>8.1f} {100*bw(t_cud)/peak:>5.1f}% {rel(y_cud):>10.2e}")
    print(f"  speedup (triton/cuda): {t_tri/t_cud:.2f}x")

if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)}   peak HBM: {peak_bw_gbps():.0f} GB/s")
    # Mistral-7B projection shapes at B=1, 40% activation sparsity.
    for N, K in [(4096, 4096), (14336, 4096), (4096, 14336), (1024, 4096)]:
        run(N, K, 0.40)
