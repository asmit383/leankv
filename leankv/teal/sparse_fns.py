"""
Threshold-based activation sparsification for TEAL.

Applies magnitude-based pruning: activations with |x_i| <= threshold are
zeroed out. This is the core sparsification primitive used by the patching
module.
"""

import torch
from typing import Optional


class SparsifyFn:
    """
    Magnitude-based activation sparsifier.

    For a given threshold t, zeros out all activations where |x_i| <= t.
    When used as a forward pre-hook on nn.Linear, this eliminates the
    corresponding weight columns from the matmul (input sparsity).

    In hook-based mode, the matmul still processes zeros (no speedup).
    When swapped for the Triton sparse GEMV kernel (Step 6), the zeros
    are skipped entirely for actual wall-clock gains.
    """

    def __init__(self, threshold: float = 0.0, enabled: bool = True):
        self.threshold = threshold
        self.enabled = enabled
        # Track actual sparsity for diagnostics
        self._last_sparsity = 0.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply magnitude-based sparsification to activation tensor."""
        if not self.enabled or self.threshold <= 0:
            return x

        mask = x.abs() > self.threshold
        self._last_sparsity = 1.0 - mask.float().mean().item()
        return x * mask

    @property
    def last_sparsity(self) -> float:
        """Fraction of activations that were zeroed in the last call."""
        return self._last_sparsity


def compute_threshold_for_sparsity(
    activations: torch.Tensor, target_sparsity: float
) -> float:
    """
    Compute the magnitude threshold that achieves target_sparsity fraction
    of zeros on the given activation tensor.

    Args:
        activations: tensor of activation values (any shape).
        target_sparsity: fraction in [0, 1] of activations to zero out.

    Returns:
        Threshold value t such that ~target_sparsity of |activations| <= t.
    """
    magnitudes = activations.abs().flatten()
    k = int(target_sparsity * magnitudes.numel())
    if k <= 0:
        return 0.0
    if k >= magnitudes.numel():
        return magnitudes.max().item()
    # k-th smallest magnitude is the threshold
    threshold = torch.kthvalue(magnitudes, k).values.item()
    return threshold


def compute_threshold_from_histogram(
    hist_counts: torch.Tensor,
    hist_edges: torch.Tensor,
    target_sparsity: float,
) -> float:
    """
    Compute threshold from a precomputed histogram (used in calibration).

    Args:
        hist_counts: bin counts from torch.histogram.
        hist_edges: bin edges from torch.histogram.
        target_sparsity: fraction in [0, 1].

    Returns:
        Threshold value.
    """
    # Compute CDF from histogram
    cdf = hist_counts.cumsum(0).float()
    cdf = cdf / cdf[-1]

    # Find the edge where CDF crosses target_sparsity
    idx = torch.searchsorted(cdf, target_sparsity).item()
    idx = min(idx, len(hist_edges) - 2)

    # Interpolate within the bin
    if idx > 0:
        frac = (target_sparsity - cdf[idx - 1]) / (cdf[idx] - cdf[idx - 1] + 1e-10)
        threshold = hist_edges[idx] + frac * (hist_edges[idx + 1] - hist_edges[idx])
    else:
        threshold = hist_edges[idx]

    return threshold.item()
