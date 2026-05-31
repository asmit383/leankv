"""
Triton sparse GEMV kernel for TEAL activation sparsity.

Ported from FasterDecoding/TEAL (MIT License).
https://github.com/FasterDecoding/TEAL

Core idea: compute y = sparse(x) @ W where activations below a magnitude
threshold are skipped entirely. The kernel fuses threshold comparison with
the matmul — never loads weight rows for zeroed activations.

Key optimizations:
  - Fused mask: abs(x) > threshold computed inside kernel, no separate pass
  - evict_last for x: kept in L2 cache (reused across thread blocks)
  - evict_first for W: streamed through (block-specific, no reuse)
  - SplitK: output dimension split across blocks, atomic_add to merge
  - Autotuned: Triton finds best BLOCK_M/BLOCK_N per GPU

Requires:
  - FP16 inputs (Triton atomic_add doesn't support BF16)
  - Column-major weights: weight.t().contiguous()
  - Batch size 1 (GEMV, not GEMM)
"""

import torch
import triton
import triton.language as tl


def init_to_zero(*names):
    def init_func(nargs):
        for name in names:
            nargs[name].zero_()
    return init_func


# Autotune configurations — Triton tries each and picks the fastest
_CONFIGS = [
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


@triton.autotune(configs=_CONFIGS, key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE", "SPARSITY_BIN"])
@triton.jit
def _splitk_sparse_gemv_kernel(
    Y,  # output pointer
    A,  # weight pointer (column-major)
    X,  # input pointer
    threshold,  # magnitude threshold
    N, M,  # dimensions: output_dim, input_dim
    CACHE_KEY_N, CACHE_KEY_M,
    BATCHSIZE: tl.constexpr,
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

    if BATCHSIZE == 1:
        # Load input with evict_last (keep in L2 — reused across blocks)
        x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy='evict_last')
        # Fused threshold comparison — this IS the sparsification
        idx = tl.abs(x0) > threshold
        # Selectively load weights with evict_first (stream through, no reuse)
        a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy='evict_first')
        # Accumulate in FP32 for precision
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y_ptr, acc0, mask=rn < N)


@triton.autotune(configs=_CONFIGS, key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE", "SPARSITY_BIN"])
@triton.jit
def _qkv_kernel(
    Y, A, X,
    threshold_q, threshold_k, threshold_v,
    N, N_q, N_kv, M,
    CACHE_KEY_N, CACHE_KEY_M,
    BATCHSIZE: tl.constexpr,
    SPARSITY_BIN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused Q/K/V projection with per-projection thresholds."""
    start_n = tl.program_id(0)
    start_m = tl.program_id(1)

    is_q = start_n * BLOCK_N < N_q
    is_v = N_q + N_kv <= start_n * BLOCK_N

    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A_ptr = A + rm[:, None] * N + rn[None, :]
    X_ptr = X + rm
    Y_ptr = Y + rn

    threshold = tl.where(is_q, threshold_q, tl.where(is_v, threshold_v, threshold_k))

    if BATCHSIZE == 1:
        x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy='evict_last')
        idx = tl.abs(x0) > threshold
        a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy='evict_first')
        acc = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y_ptr, acc, mask=rn < N)


# ── Python wrappers ─────────────────────────────────────────────────────

def sparse_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
    sparsity_bin: int = 5,
) -> torch.Tensor:
    """
    Sparse GEMV: y = sparse(x) @ weight.

    Args:
        x: input tensor [batch, seq, hidden_dim] (batch and seq must be 1)
        weight: weight matrix [output_dim, hidden_dim] (must be column-major)
        threshold: magnitude threshold for activation sparsity
        sparsity_bin: approximate sparsity level for kernel autotuning

    Returns:
        output tensor [batch, seq, output_dim]
    """
    N, Z = weight.shape
    beam_width, seq_len, _ = x.shape
    assert x.shape[2] == Z
    x = x.contiguous()
    assert weight.stride(1) > 1, "weight must be column-major (call weight.t().contiguous())"

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(Z, META["BLOCK_M"]),
    )

    output = torch.empty(beam_width, seq_len, N, device=x.device, dtype=torch.float16)

    _splitk_sparse_gemv_kernel[grid](
        output, weight, x, threshold,
        N, Z,
        N // 16, Z // 16,
        beam_width,
        sparsity_bin,
    )

    if x.dtype is not output.dtype:
        return output.to(dtype=x.dtype)
    return output


def sparse_qkv_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold_q: float,
    threshold_k: float,
    threshold_v: float,
    kv_size: int,
    sparsity_bin: int = 5,
) -> torch.Tensor:
    """
    Fused sparse Q/K/V projection with per-projection thresholds.

    Args:
        x: input tensor [batch, seq, hidden_dim]
        weight: concatenated QKV weight [q_dim + k_dim + v_dim, hidden_dim]
        threshold_q/k/v: per-projection magnitude thresholds
        kv_size: dimension of K (and V) projection
        sparsity_bin: for autotuning

    Returns:
        output tensor [batch, seq, q_dim + k_dim + v_dim]
    """
    N, Z = weight.shape
    beam_width, seq_len, _ = x.shape
    assert x.shape[2] == Z
    x = x.contiguous()
    assert weight.stride(1) > 1, "weight must be column-major"

    N_q = N - 2 * kv_size

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(Z, META["BLOCK_M"]),
    )

    output = torch.empty(beam_width, seq_len, N, device=x.device, dtype=torch.float16)

    _qkv_kernel[grid](
        output, weight, x,
        threshold_q, threshold_k, threshold_v,
        N, N_q, kv_size, Z,
        N // 16, Z // 16,
        beam_width,
        sparsity_bin,
    )

    if x.dtype is not output.dtype:
        return output.to(dtype=x.dtype)
    return output


def prepare_weight_for_sparse_gemv(weight: torch.Tensor) -> torch.Tensor:
    """
    Convert weight to column-major layout for the sparse GEMV kernel.
    Call this once at model load time.

    Args:
        weight: [output_dim, input_dim] row-major tensor

    Returns:
        [output_dim, input_dim] column-major tensor
    """
    return weight.t().contiguous().t()
