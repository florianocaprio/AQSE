from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import wilcoxon

from .models import (
    IntervalEstimate,
    PairedAggregate,
    ReplicaResult,
    ReplicatedStudyAggregate,
    WilcoxonResult,
)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = 2_000,
) -> IntervalEstimate:
    """Estimate an unrounded percentile CI; it does not prove a null or alternative."""

    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("bootstrap requires at least two finite independent values")
    if resamples < 100:
        raise ValueError("bootstrap requires at least 100 resamples")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = np.mean(array[indices], axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    point = float(np.mean(array))
    return IntervalEstimate(
        point=point,
        lower=min(float(lower), point),
        upper=max(float(upper), point),
        bootstrap_resamples=resamples,
    )


def paired_wilcoxon(
    quantum: Sequence[float], classical: Sequence[float]
) -> WilcoxonResult:
    """Run a two-sided paired Wilcoxon test without asserting practical importance."""

    left = np.asarray(quantum, dtype=np.float64)
    right = np.asarray(classical, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("Wilcoxon requires at least two aligned pairs")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Wilcoxon inputs must be finite")
    differences = left - right
    if np.all(differences == 0.0):
        return WilcoxonResult(
            statistic=0.0,
            p_value=1.0,
            zero_difference_policy="all differences zero; p=1 by explicit convention",
        )
    result = wilcoxon(left, right, alternative="two-sided", zero_method="pratt", method="auto")
    return WilcoxonResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        zero_difference_policy="Pratt includes zero differences in ranking",
    )


def aggregate_replicas(
    replicas: Sequence[ReplicaResult],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int = 2_000,
) -> ReplicatedStudyAggregate:
    """Aggregate independent replicas; association is not a causal quantum claim."""

    ordered = sorted(replicas, key=lambda item: item.replica_index)
    if len(ordered) < 2 or len({item.replica_index for item in ordered}) != len(ordered):
        raise ValueError("aggregation requires at least two unique replicas")
    baseline_names = tuple(sorted(ordered[0].baseline_balanced_accuracy))
    if any(tuple(sorted(item.baseline_balanced_accuracy)) != baseline_names for item in ordered):
        raise ValueError("replicas do not contain the same baseline methods")

    quantum_values = [item.quantum_balanced_accuracy for item in ordered]
    baseline_intervals: dict[str, IntervalEstimate] = {}
    delta_aggregates: dict[str, PairedAggregate] = {}
    for offset, name in enumerate(baseline_names, start=1):
        classical_values = [item.baseline_balanced_accuracy[name] for item in ordered]
        deltas = [left - right for left, right in zip(quantum_values, classical_values, strict=True)]
        baseline_intervals[name] = bootstrap_mean_interval(
            classical_values,
            seed=bootstrap_seed + offset,
            resamples=bootstrap_resamples,
        )
        delta_aggregates[name] = PairedAggregate(
            delta=bootstrap_mean_interval(
                deltas,
                seed=bootstrap_seed + 100 + offset,
                resamples=bootstrap_resamples,
            ),
            wilcoxon=paired_wilcoxon(quantum_values, classical_values),
            pair_count=len(ordered),
        )

    diagnostic_values = {
        "trace": [item.selected_kernel_diagnostics.trace for item in ordered],
        "maximum_eigenvalue_trace_ratio": [
            item.selected_kernel_diagnostics.maximum_eigenvalue_trace_ratio
            for item in ordered
        ],
        "effective_rank": [
            item.selected_kernel_diagnostics.effective_rank for item in ordered
        ],
    }
    diagnostics = {
        name: bootstrap_mean_interval(
            values,
            seed=bootstrap_seed + 500 + index,
            resamples=bootstrap_resamples,
        )
        for index, (name, values) in enumerate(sorted(diagnostic_values.items()))
    }
    return ReplicatedStudyAggregate(
        quantum_balanced_accuracy=bootstrap_mean_interval(
            quantum_values,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        ),
        baseline_balanced_accuracy=baseline_intervals,
        delta_by_baseline=delta_aggregates,
        selected_kernel_diagnostics=diagnostics,
    )
