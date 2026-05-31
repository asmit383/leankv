"""
Triton sparse kernels for TEAL activation sparsity.

Two kernels:
  1. sparse_gemv: B=1 decode, per-element sparsity (~50%), maximum speedup
  2. batched_sparse_gemm: B>1 decode, per-element sparsity per batch item
     Uses 3D grid (output × input × batch) — all batch items run in parallel

Both fuse threshold comparison with matmul — never loads weight rows
for zeroed activations.

Ported from FasterDecoding/TEAL (MIT License) and extended for batching.
"""

import torch
import triton
import triton.language as tl


def init_to_zero(*names):
    def init_func(nargs):
        for name in names:
            nargs[name].zero_()
    return init_func


# ── Autotune configs ────────────────────────────────────────────────────

_CONFIGS_B1 = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
]

_CONFIGS_BATCHED = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
]


# ── B=1 Sparse GEMV (original TEAL kernel) ─────────────────────────────

@triton.autotune(configs=_CONFIGS_B1, key=["CACHE_KEY_M", "CACHE_KEY_N", "SPARSITY_BIN"])
@triton.jit
def _splitk_sparse_gemv_kernel(
    Y, A, X, threshold,
    N, M,
    CACHE_KEY_N, CACHE_KEY_M,
    SPARSITY_BIN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_n = tl.program_id(0)
    start_m = tl.program_id(1)

    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    A_ptr = A + (rm[:, None] * N + rn[None, :])
    X_ptr = X + rm
    Y_ptr = Y + rn

    x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy='evict_last')
    idx = tl.abs(x0) > threshold
    a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy='evict_first')
    acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y_ptr, acc0, mask=rn < N)


# ── B>1 Batched Sparse GEMM ────────────────────────────────────────────

@triton.autotune(configs=_CONFIGS_BATCHED, key=["CACHE_KEY_M", "CACHE_KEY_N", "SPARSITY_BIN"])
@triton.jit
def _batched_sparse_gemm_kernel(
    Y,  # (B, N) output, contiguous
    A,  # (M, N) weight, column-major (shared across batch)
    X,  # (B, M) input, contiguous
    threshold,
    B, N, M,
    CACHE_KEY_N, CACHE_KEY_M,
    SPARSITY_BIN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Batched sparse GEMM with per-element sparsity per batch item.

    Grid: (N // BLOCK_N, M // BLOCK_M, B)
    Each batch item gets its own thread blocks with its own sparsity mask.
    Weights are shared (same A for all batch items).
    """
    pid_n = tl.program_id(0)  # output dim chunk
    pid_m = tl.program_id(1)  # input dim chunk (SplitK)
    pid_b = tl.program_id(2)  # batch item

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Pointers for this batch item
    X_ptr = X + pid_b * M + rm
    Y_ptr = Y + pid_b * N + rn
    A_ptr = A + (rm[:, None] * N + rn[None, :])

    # Load input for this batch item
    x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy='evict_last')

    # Per-element sparsity mask for THIS batch item
    idx = tl.abs(x0) > threshold

    # Selectively load weights (skip rows where x is near-zero)
    a = tl.load(A_ptr, mask=idx[:, None] & (rn[None, :] < N), other=0.0, eviction_policy='evict_first')

    # Accumulate in FP32
    acc = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    # Atomic add for SplitK merge (per batch item output)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y_ptr, acc, mask=rn < N)


# ── Python wrappers ─────────────────────────────────────────────────────

def sparse_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
    sparsity_bin: int = 5,
) -> torch.Tensor:
    """
    Sparse GEMV for B=1: y = sparse(x) @ weight.

    Args:
        x: [1, 1, D] input tensor
        weight: [N, D] column-major weight matrix
        threshold: magnitude threshold
        sparsity_bin: for autotuning cache key

    Returns:
        [1, 1, N] output tensor
    """
    N, Z = weight.shape
    x = x.reshape(1, Z).contiguous()

    output = torch.empty(1, N, device=x.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(Z, META["BLOCK_M"]),
    )

    _splitk_sparse_gemv_kernel[grid](
        output, weight, x, threshold,
        N, Z,
        N // 16, Z // 16,
        sparsity_bin,
    )

    result = output.reshape(1, 1, N)
    if x.dtype is not torch.float16:
        result = result.to(dtype=x.dtype)
    return result


def batched_sparse_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
    sparsity_bin: int = 5,
) -> torch.Tensor:
    """
    Batched sparse GEMM for B>1: Y = sparse(X) @ weight for each batch item.

    Each batch item has its own per-element sparsity mask.
    All batch items run in parallel via 3D grid.

    Args:
        x: [B, D] input tensor (batch of hidden states)
        weight: [N, D] column-major weight matrix
        threshold: magnitude threshold
        sparsity_bin: for autotuning cache key

    Returns:
        [B, N] output tensor
    """
    B_size = x.shape[0]
    N, Z = weight.shape
    x = x.contiguous()

    output = torch.empty(B_size, N, device=x.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(Z, META["BLOCK_M"]),
        B_size,  # batch dimension in grid
    )

    _batched_sparse_gemm_kernel[grid](
        output, weight, x, threshold,
        B_size, N, Z,
        N // 16, Z // 16,
        sparsity_bin,
    )

    if x.dtype is not torch.float16:
        output = output.to(dtype=x.dtype)
    return output


def prepare_weight_for_sparse_gemv(weight: torch.Tensor) -> torch.Tensor:
    """
    Convert weight to column-major layout for sparse kernels.
    Call once at model load time.
    """
    return weight.t().contiguous().t()
