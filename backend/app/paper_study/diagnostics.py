from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import KernelDiagnosticsResult


def kernel_diagnostics(K: ArrayLike) -> KernelDiagnosticsResult:
    """Summarize one finite Gram matrix without claiming predictive quality.

    The effective rank is ``exp(H(p))`` where ``p`` contains the non-negative
    eigenvalues normalized by their sum.  Tiny negative eigenvalues caused by
    floating-point roundoff are clipped only after rejecting materially
    indefinite matrices.  These diagnostics describe this matrix and do not
    demonstrate quantum advantage, expressivity, or generalization.
    """

    matrix = np.asarray(K, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("kernel diagnostics require a square matrix of size >= 2")
    if not np.isfinite(matrix).all():
        raise ValueError("kernel matrix contains NaN or infinity")
    if float(np.max(np.abs(matrix - matrix.T))) > 1.0e-10:
        raise ValueError("kernel matrix must be symmetric")
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues: NDArray[np.float64] = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues[0]) < -1.0e-9 * scale:
        raise ValueError("kernel matrix is materially indefinite")
    clipped = np.clip(eigenvalues, 0.0, None)
    trace = float(np.sum(clipped))
    if trace <= 1.0e-15:
        raise ValueError("kernel matrix has zero positive trace")
    probabilities = clipped[clipped > 0.0] / trace
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    diagonal = np.diag(matrix)
    off_diagonal = matrix[~np.eye(len(matrix), dtype=np.bool_)]
    positive = clipped[clipped > 1.0e-12 * scale]
    condition = None
    if len(positive) > 1:
        condition = float(positive[-1] / positive[0])
    return KernelDiagnosticsResult(
        eigenvalues=tuple(float(value) for value in clipped),
        trace=trace,
        maximum_eigenvalue_trace_ratio=float(clipped[-1] / trace),
        effective_rank=float(np.exp(entropy)),
        minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]),
        diagonal_minimum=float(np.min(diagonal)),
        diagonal_maximum=float(np.max(diagonal)),
        diagonal_mean=float(np.mean(diagonal)),
        symmetry_max_abs_error=float(np.max(np.abs(matrix - matrix.T))),
        off_diagonal_minimum=float(np.min(off_diagonal)),
        off_diagonal_maximum=float(np.max(off_diagonal)),
        off_diagonal_mean=float(np.mean(off_diagonal)),
        off_diagonal_standard_deviation=float(np.std(off_diagonal)),
        positive_condition_number=condition,
    )
