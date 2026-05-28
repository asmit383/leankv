"""
Random rotation utilities for TurboQuant.

Uses QR decomposition of a random Gaussian matrix to produce Π.
For typical head_dim (64-256), the full QR approach is fine — a 128x128
matrix is only 64KB in float32.
"""

import torch


def generate_rotation_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a random orthogonal matrix Π ∈ R^{d×d} via QR decomposition."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    G = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)

    # Fix signs so the decomposition is unique
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)

    return Q.to(device=device, dtype=dtype)


def generate_qjl_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 12345,
) -> torch.Tensor:
    """Generate random projection matrix S ∈ R^{d×d} for QJL (i.i.d. N(0,1))."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    S = torch.randn(d, d, generator=rng, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)


def rotate_forward(x: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Apply random rotation: y = x @ Π^T."""
    return torch.matmul(x, Pi.T)


def rotate_backward(y: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Apply inverse rotation: x = y @ Π."""
    return torch.matmul(y, Pi)
