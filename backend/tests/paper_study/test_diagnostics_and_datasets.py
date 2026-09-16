from __future__ import annotations

import numpy as np
import pytest

from app.paper_study.datasets import (
    positive_control_dataset,
    split_binary_dataset,
    validate_replicated_dataset,
)
from app.paper_study.diagnostics import kernel_diagnostics


def test_kernel_diagnostics_are_invariant_to_simultaneous_permutation() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.3],
            [0.1, 0.3, 1.0],
        ]
    )
    expected = kernel_diagnostics(matrix)
    permutation = np.asarray([2, 0, 1])
    actual = kernel_diagnostics(matrix[np.ix_(permutation, permutation)])
    assert actual.eigenvalues == pytest.approx(expected.eigenvalues, abs=1.0e-12)
    assert actual.effective_rank == pytest.approx(expected.effective_rank, abs=1.0e-12)
    assert actual.maximum_eigenvalue_trace_ratio == pytest.approx(
        expected.maximum_eigenvalue_trace_ratio, abs=1.0e-12
    )


@pytest.mark.parametrize(
    "matrix",
    (
        np.ones((2, 3)),
        np.zeros((3, 3)),
        np.asarray([[1.0, 2.0], [0.0, 1.0]]),
        np.asarray([[1.0, 2.0], [2.0, 1.0]]),
    ),
)
def test_kernel_diagnostics_reject_degenerate_inputs(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError):
        kernel_diagnostics(matrix)


def test_split_is_invariant_to_input_order() -> None:
    rng = np.random.default_rng(71)
    X = rng.normal(size=(80, 8))
    y = np.repeat(np.asarray([-1, 1], dtype=np.int8), 40)
    expected = split_binary_dataset(X, y, split_seed=901)
    order = rng.permutation(len(X))
    actual = split_binary_dataset(X[order], y[order], split_seed=901)
    for name in ("train", "validation", "test"):
        assert getattr(actual, f"{name}_ids") == getattr(expected, f"{name}_ids")
        assert np.array_equal(getattr(actual, f"X_{name}"), getattr(expected, f"X_{name}"))
        assert np.array_equal(getattr(actual, f"y_{name}"), getattr(expected, f"y_{name}"))
    validate_replicated_dataset(actual)


def test_positive_control_is_deterministic_balanced_and_validated() -> None:
    X, y = positive_control_dataset(80, 77)
    repeated_X, repeated_y = positive_control_dataset(80, 77)
    assert X.shape == (80, 8)
    assert np.array_equal(X, repeated_X)
    assert np.array_equal(y, repeated_y)
    assert np.count_nonzero(y == -1) == np.count_nonzero(y == 1) == 40
    assert not X.flags.writeable
    assert not y.flags.writeable
    with pytest.raises(ValueError, match="even n"):
        positive_control_dataset(39, 77)


def test_split_rejects_constant_labels_and_small_datasets() -> None:
    X = np.zeros((20, 8), dtype=np.float64)
    with pytest.raises(ValueError, match=r"labels -1/\+1"):
        split_binary_dataset(X, np.ones(20, dtype=np.int8), split_seed=1)
    with pytest.raises(ValueError, match="at least 20"):
        split_binary_dataset(
            np.zeros((18, 8), dtype=np.float64),
            np.asarray([-1, 1] * 9, dtype=np.int8),
            split_seed=1,
        )
