from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class IntervalEstimate(FrozenModel):
    """A point estimate and unrounded percentile interval.

    The interval quantifies sampling uncertainty under the declared bootstrap
    unit.  It does not prove practical relevance or causal superiority.
    """

    point: float
    lower: float
    upper: float
    confidence_level: Literal[0.95] = 0.95
    bootstrap_resamples: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_order(self) -> IntervalEstimate:
        if not self.lower <= self.point <= self.upper:
            raise ValueError("interval must contain its point estimate")
        return self


class KernelDiagnosticsResult(FrozenModel):
    """Finite-sample Gram spectrum summary, not a generalization guarantee."""

    eigenvalues: tuple[float, ...]
    trace: float = Field(gt=0.0)
    maximum_eigenvalue_trace_ratio: float = Field(ge=0.0, le=1.0)
    effective_rank: float = Field(ge=1.0)
    minimum_eigenvalue: float
    maximum_eigenvalue: float = Field(gt=0.0)
    diagonal_minimum: float
    diagonal_maximum: float
    diagonal_mean: float
    symmetry_max_abs_error: float = Field(ge=0.0)
    off_diagonal_minimum: float
    off_diagonal_maximum: float
    off_diagonal_mean: float
    off_diagonal_standard_deviation: float = Field(ge=0.0)
    positive_condition_number: float | None = Field(default=None, ge=1.0)


class ClassificationMetricPair(FrozenModel):
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)


class SplitSizes(FrozenModel):
    train: int = Field(gt=0)
    validation: int = Field(gt=0)
    test: int = Field(gt=0)


class ReplicaSeeds(FrozenModel):
    replica: int = Field(ge=0, le=2**32 - 1)
    dataset: int = Field(ge=0, le=2**32 - 1)
    split: int = Field(ge=0, le=2**32 - 1)
    qng: int = Field(ge=0, le=2**32 - 1)
    classical: int = Field(ge=0, le=2**32 - 1)
    permutation: int = Field(ge=0, le=2**32 - 1)


class QuantumSelection(FrozenModel):
    checkpoint_index: int = Field(ge=0)
    accepted_qng_updates: int = Field(ge=0)
    svc_c: float = Field(gt=0.0)
    theta: tuple[float, ...] = Field(min_length=16, max_length=16)
    validation_balanced_accuracy: float = Field(ge=0.0, le=1.0)
    validation_macro_f1: float = Field(ge=0.0, le=1.0)


class ReplicaResult(FrozenModel):
    """One isolated TEST result; it is not an aggregate scientific claim."""

    replica_index: int = Field(ge=0)
    base_seed: int = Field(ge=0, le=2**32 - 1)
    seeds: ReplicaSeeds
    split_sizes: SplitSizes
    split_identities: dict[str, tuple[str, ...]]
    class_balance: dict[str, dict[str, int]]
    dataset_metadata: dict[str, Any]
    quantum_balanced_accuracy: float = Field(ge=0.0, le=1.0)
    baseline_balanced_accuracy: dict[str, float]
    delta_by_baseline: dict[str, float]
    quantum_selection: QuantumSelection
    kernel_diagnostics_by_checkpoint: tuple[KernelDiagnosticsResult, ...]
    selected_kernel_diagnostics: KernelDiagnosticsResult
    test_kernel_statistics: dict[str, float]
    train_metrics: dict[str, ClassificationMetricPair]
    validation_metrics: dict[str, ClassificationMetricPair]
    test_metrics: dict[str, ClassificationMetricPair]
    runtime_seconds: dict[str, float]
    source_hashes: dict[str, str]
    abstention_count: int = Field(ge=0)
    failure: str | None = None
    test_ledger_path: str

    @model_validator(mode="after")
    def validate_methods(self) -> ReplicaResult:
        if set(self.baseline_balanced_accuracy) != set(self.delta_by_baseline):
            raise ValueError("baseline scores and deltas must name the same methods")
        for name, score in self.baseline_balanced_accuracy.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"baseline {name} balanced accuracy is outside [0, 1]")
            expected = self.quantum_balanced_accuracy - score
            if abs(self.delta_by_baseline[name] - expected) > 1.0e-12:
                raise ValueError(f"baseline {name} delta is inconsistent")
        expected_methods = {"quantum", *self.baseline_balanced_accuracy}
        for field in (self.train_metrics, self.validation_metrics, self.test_metrics):
            if set(field) != expected_methods:
                raise ValueError("TRAIN/VALIDATION/TEST metrics must contain every method")
        return self


class WilcoxonResult(FrozenModel):
    statistic: float = Field(ge=0.0)
    p_value: float = Field(ge=0.0, le=1.0)
    alternative: Literal["two-sided"] = "two-sided"
    zero_difference_policy: str


class PairedAggregate(FrozenModel):
    delta: IntervalEstimate
    wilcoxon: WilcoxonResult
    pair_count: int = Field(ge=2)


class ReplicatedStudyAggregate(FrozenModel):
    quantum_balanced_accuracy: IntervalEstimate
    baseline_balanced_accuracy: dict[str, IntervalEstimate]
    delta_by_baseline: dict[str, PairedAggregate]
    selected_kernel_diagnostics: dict[str, IntervalEstimate]


class ReplicatedStudyResult(FrozenModel):
    """Replicated simulation evidence, not proof of quantum advantage."""

    schema_version: Literal["aqse.paper-replicated-study.v1"] = (
        "aqse.paper-replicated-study.v1"
    )
    base_seed: int = Field(ge=0, le=2**32 - 1)
    replica_count: int = Field(ge=2)
    qng_steps: int = Field(ge=0)
    replicas: tuple[ReplicaResult, ...]
    aggregate: ReplicatedStudyAggregate
    scientific_scope: str = (
        "independent simulator replicas; no field-validity or quantum-advantage claim"
    )

    @model_validator(mode="after")
    def validate_replicas(self) -> ReplicatedStudyResult:
        if len(self.replicas) != self.replica_count:
            raise ValueError("replica count does not match stored results")
        indices = [item.replica_index for item in self.replicas]
        if sorted(indices) != list(range(self.replica_count)):
            raise ValueError("replica indices must be unique and contiguous")
        return self


class PermutationControlResult(FrozenModel):
    """Label-permutation results; chance compatibility is not model validity."""

    replica_count: int = Field(ge=2)
    balanced_accuracy: dict[str, IntervalEstimate]
    compatible_with_chance: dict[str, bool]
    chance_value: Literal[0.5] = 0.5
    study: ReplicatedStudyResult

    @model_validator(mode="after")
    def validate_methods(self) -> PermutationControlResult:
        if set(self.balanced_accuracy) != set(self.compatible_with_chance):
            raise ValueError("permutation metrics and decisions must align")
        return self
