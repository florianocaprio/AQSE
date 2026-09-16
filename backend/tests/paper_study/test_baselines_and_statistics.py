from __future__ import annotations

import numpy as np
import pytest

from app.paper_study.baselines import BASELINE_NAMES, fit_classical_comparators
from app.paper_study.models import (
    KernelDiagnosticsResult,
    QuantumSelection,
    ReplicaResult,
    ReplicaSeeds,
    SplitSizes,
)
from app.paper_study.statistics import aggregate_replicas, paired_wilcoxon


def _replica(index: int, quantum: float, offset: float) -> ReplicaResult:
    baselines = {
        name: quantum - offset - method_index * 0.01
        for method_index, name in enumerate(BASELINE_NAMES)
    }
    diagnostics = KernelDiagnosticsResult(
        eigenvalues=(0.5, 1.5),
        trace=2.0,
        maximum_eigenvalue_trace_ratio=0.75,
        effective_rank=1.75,
    )
    return ReplicaResult(
        replica_index=index,
        seeds=ReplicaSeeds(
            replica=10 + index,
            dataset=20 + index,
            split=30 + index,
            qng=40 + index,
            classical=50 + index,
            permutation=60 + index,
        ),
        split_sizes=SplitSizes(train=12, validation=4, test=4),
        dataset_metadata={"fixture": True},
        quantum_balanced_accuracy=quantum,
        baseline_balanced_accuracy=baselines,
        delta_by_baseline={name: quantum - score for name, score in baselines.items()},
        quantum_selection=QuantumSelection(
            checkpoint_index=0,
            accepted_qng_updates=0,
            svc_c=1.0,
            theta=(0.0,) * 16,
            validation_balanced_accuracy=0.5,
            validation_macro_f1=0.5,
        ),
        kernel_diagnostics_by_checkpoint=(diagnostics,),
        selected_kernel_diagnostics=diagnostics,
        test_ledger_path=f"replica-{index}/ledger.jsonl",
    )


def test_classical_comparators_include_all_declared_methods() -> None:
    rng = np.random.default_rng(12)
    X_train = rng.normal(size=(32, 8))
    y_train = np.asarray([-1, 1] * 16, dtype=np.int8)
    X_validation = rng.normal(size=(12, 8))
    y_validation = np.asarray([-1, 1] * 6, dtype=np.int8)
    fitted = fit_classical_comparators(
        X_train,
        y_train,
        X_validation,
        y_validation,
        seed=99,
    )
    assert tuple(fitted) == BASELINE_NAMES
    assert fitted["mlp_11_parameter"].configuration["trainable_parameter_count"] == 11
    assert fitted["rff_256"].configuration["n_components"] == 256


def test_classical_comparators_reject_constant_labels() -> None:
    with pytest.raises(ValueError, match=r"both -1 and \+1"):
        fit_classical_comparators(
            np.zeros((12, 8)),
            np.ones(12, dtype=np.int8),
            np.zeros((4, 8)),
            np.asarray([-1, -1, 1, 1], dtype=np.int8),
            seed=1,
        )


def test_replica_aggregation_is_order_invariant() -> None:
    first = _replica(0, 0.65, 0.05)
    second = _replica(1, 0.75, 0.10)
    expected = aggregate_replicas(
        (first, second), bootstrap_seed=81, bootstrap_resamples=200
    )
    actual = aggregate_replicas(
        (second, first), bootstrap_seed=81, bootstrap_resamples=200
    )
    assert actual == expected
    assert set(actual.delta_by_baseline) == set(BASELINE_NAMES)


def test_wilcoxon_handles_all_zero_differences_explicitly() -> None:
    result = paired_wilcoxon((0.5, 0.5), (0.5, 0.5))
    assert result.statistic == 0.0
    assert result.p_value == 1.0
