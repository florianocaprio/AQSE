from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from app.training.models import FrozenModel

MethodName = Literal[
    "snr_threshold",
    "rbf_svc_circular",
    "fixed_theta_tqk",
    "gradient_tqk",
    "qng_tqk",
]


class EvaluationProtocol(FrozenModel):
    schema_version: Literal["aqse.comparative-protocol.v1"] = (
        "aqse.comparative-protocol.v1"
    )
    protocol_id: str = Field(pattern=r"^aqse-comparative-protocol-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    milestone: Literal["1D.4a"] = "1D.4a"
    dataset_id: Literal["aqse-development-064acca20fc788c6"]
    dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    encoder_id: Literal["aqse-encoder-f9cf4bc12a767410"]
    encoder_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    train_bank_id: Literal["aqse-train-bank-32b5f93897676535"]
    train_bank_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_bank_id: Literal["aqse-validation-bank-105467fa5a4a5cc7"]
    validation_bank_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    seed_schedule: tuple[int, ...]
    methods: tuple[MethodName, ...]
    snr_rule: Literal["predict +1 when snr <= TRAIN-selected threshold"]
    snr_training_tie_break: Literal[
        "max TRAIN balanced_accuracy then macro_f1 then lowest threshold"
    ]
    rbf_c_grid: tuple[float, ...]
    rbf_gamma_grid: tuple[float, ...]
    quantum_svc_c_grid: tuple[float, ...]
    maximum_updates: Literal[10] = 10
    qng_learning_rate: Literal[0.2] = 0.2
    qng_damping: Literal[0.001] = 0.001
    qng_maximum_step_norm: Literal[0.4] = 0.4
    qng_armijo: Literal["protected-backtracking-unchanged"] = (
        "protected-backtracking-unchanged"
    )
    gradient_learning_rate: Literal[0.2] = 0.2
    gradient_maximum_step_norm: Literal[0.4] = 0.4
    gradient_line_search: Literal["none"] = "none"
    selection_primary: Literal["validation_balanced_accuracy"] = (
        "validation_balanced_accuracy"
    )
    selection_secondary: Literal["validation_macro_f1"] = "validation_macro_f1"
    selection_tie_break: Literal[
        "earliest_checkpoint_then_lowest_C_then_lowest_gamma_then_lowest_seed"
    ] = "earliest_checkpoint_then_lowest_C_then_lowest_gamma_then_lowest_seed"
    bootstrap_resamples: Literal[2000] = 2000
    bootstrap_seed: Literal[1001010] = 1_001_010
    bootstrap_policy: Literal["stratified-independent-lineage-percentile-95"] = (
        "stratified-independent-lineage-percentile-95"
    )
    validation_role: Literal["selection-only"] = "selection-only"
    final_fit_policy: Literal["TRAIN-only; no refit after VALIDATION selection"] = (
        "TRAIN-only; no refit after VALIDATION selection"
    )
    future_test_policy: Literal[
        "separate-G5-authorization; one fixed central window per TEST lineage"
    ] = "separate-G5-authorization; one fixed central window per TEST lineage"
    test_state_required: Literal["sealed"] = "sealed"
    create_test_bank: Literal[False] = False

    @model_validator(mode="after")
    def validate_frozen_budget(self) -> EvaluationProtocol:
        if self.seed_schedule != (1_001_005, 1_001_006, 1_001_007, 1_001_008, 1_001_009):
            raise ValueError("comparative seed schedule differs from the approved freeze")
        if self.methods != (
            "snr_threshold",
            "rbf_svc_circular",
            "fixed_theta_tqk",
            "gradient_tqk",
            "qng_tqk",
        ):
            raise ValueError("comparative method order differs from the approved protocol")
        if self.rbf_c_grid != (0.1, 1.0, 10.0):
            raise ValueError("RBF C budget differs from the frozen grid")
        if self.rbf_gamma_grid != (0.01, 0.1, 1.0):
            raise ValueError("RBF gamma budget differs from the frozen grid")
        if self.quantum_svc_c_grid != (0.1, 1.0, 10.0):
            raise ValueError("quantum evaluator C budget differs from the frozen grid")
        return self


class ComparisonInputIdentity(FrozenModel):
    schema_version: Literal["aqse.comparison-input.v1"] = "aqse.comparison-input.v1"
    fingerprint_id: str = Field(pattern=r"^aqse-comparison-input-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_id: str
    dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    encoder_id: str
    encoder_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    train_bank_id: str
    train_bank_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_bank_id: str
    validation_bank_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    train_raw_identity: dict[str, object]
    train_encoded_identity: dict[str, object]
    train_label_identity: dict[str, object]
    validation_raw_identity: dict[str, object]
    validation_encoded_identity: dict[str, object]
    validation_label_identity: dict[str, object]
    train_lineage_ids: tuple[str, ...]
    validation_lineage_ids: tuple[str, ...]
    train_window_ids: tuple[str, ...]
    validation_window_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_population(self) -> ComparisonInputIdentity:
        if len(self.train_lineage_ids) != 32 or len(set(self.train_lineage_ids)) != 32:
            raise ValueError("comparison input requires 32 distinct TRAIN lineages")
        if (
            len(self.validation_lineage_ids) != 24
            or len(set(self.validation_lineage_ids)) != 24
        ):
            raise ValueError("comparison input requires 24 distinct VALIDATION lineages")
        if set(self.train_lineage_ids) & set(self.validation_lineage_ids):
            raise ValueError("TRAIN and VALIDATION lineages must be disjoint")
        if len(self.train_window_ids) != 32 or len(self.validation_window_ids) != 24:
            raise ValueError("comparison window identities have invalid cardinality")
        return self


class ClassificationMetrics(FrozenModel):
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    roc_auc: float = Field(ge=0.0, le=1.0)
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    negative_support: int = Field(gt=0)
    positive_support: int = Field(gt=0)
    negative_recall: float = Field(ge=0.0, le=1.0)
    positive_recall: float = Field(ge=0.0, le=1.0)


class MetricInterval(FrozenModel):
    method: MethodName
    metric: Literal["balanced_accuracy", "macro_f1"]
    lower: float = Field(ge=0.0, le=1.0)
    point: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    resamples: Literal[2000] = 2000
    seed: Literal[1001010] = 1_001_010
    unit: Literal["independent_validation_lineage"] = (
        "independent_validation_lineage"
    )


class CandidateEvaluation(FrozenModel):
    candidate_id: str = Field(pattern=r"^aqse-comparison-candidate-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    method: MethodName
    seed: int | None = None
    checkpoint_index: int | None = Field(default=None, ge=0, le=10)
    svc_c: float | None = Field(default=None, gt=0.0)
    rbf_gamma: float | None = Field(default=None, gt=0.0)
    snr_threshold: float | None = None
    theta: tuple[float, ...] | None = Field(default=None, min_length=16, max_length=16)
    train_alignment_loss: float | None = None
    validation_alignment_loss: float | None = None
    train_metrics: ClassificationMetrics
    validation_metrics: ClassificationMetrics
    validation_prediction_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_score_digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_method_parameters(self) -> CandidateEvaluation:
        quantum = self.method in {"fixed_theta_tqk", "gradient_tqk", "qng_tqk"}
        if quantum != (self.theta is not None):
            raise ValueError("candidate theta presence differs from method semantics")
        if quantum and (self.seed is None or self.checkpoint_index is None or self.svc_c is None):
            raise ValueError("quantum candidate parameters are incomplete")
        if self.method == "rbf_svc_circular" and (
            self.svc_c is None or self.rbf_gamma is None
        ):
            raise ValueError("RBF candidate hyperparameters are incomplete")
        if self.method == "snr_threshold" and self.snr_threshold is None:
            raise ValueError("SNR threshold candidate is incomplete")
        return self


class MethodSelection(FrozenModel):
    method: MethodName
    selected_candidate_id: str
    selected_seed: int | None = None
    selected_checkpoint_index: int | None = None
    selected_svc_c: float | None = None
    selected_rbf_gamma: float | None = None
    selected_snr_threshold: float | None = None
    selected_theta: tuple[float, ...] | None = Field(
        default=None,
        min_length=16,
        max_length=16,
    )
    validation_metrics: ClassificationMetrics
    confidence_intervals: tuple[MetricInterval, MetricInterval]


class ClassicalPreprocessingSnapshot(FrozenModel):
    schema_version: Literal["aqse.circular-classical-scaler.v1"] = (
        "aqse.circular-classical-scaler.v1"
    )
    feature_order: tuple[str, ...]
    fitted_mean: tuple[float, ...] = Field(min_length=9, max_length=9)
    fitted_scale: tuple[float, ...] = Field(min_length=9, max_length=9)
    source_partition: Literal["TRAIN"] = "TRAIN"


class ComparativeEvaluationArtifact(FrozenModel):
    schema_version: Literal["aqse.comparative-evaluation.v1"] = (
        "aqse.comparative-evaluation.v1"
    )
    evaluation_id: str = Field(pattern=r"^aqse-comparative-evaluation-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    scientific_scope: Literal["TRAIN/VALIDATION selection only; TEST sealed"] = (
        "TRAIN/VALIDATION selection only; TEST sealed"
    )
    protocol_id: str
    protocol_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    comparison_input: ComparisonInputIdentity
    classical_preprocessing: ClassicalPreprocessingSnapshot
    candidate_count: int = Field(gt=0)
    candidates: tuple[CandidateEvaluation, ...]
    selections: tuple[MethodSelection, ...]
    selected_method_order: tuple[MethodName, ...]
    qng_primary_seed_source_run_id: Literal["aqse-qng-run-f00c702ad790df2b"]
    test_state_after: Literal["sealed"] = "sealed"
    test_ledger_sha256_after: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_evaluation(self) -> ComparativeEvaluationArtifact:
        if len(self.candidates) != self.candidate_count:
            raise ValueError("comparison candidate count is inconsistent")
        if len({item.candidate_id for item in self.candidates}) != len(self.candidates):
            raise ValueError("comparison candidate identities are not unique")
        selected = tuple(item.method for item in self.selections)
        if selected != self.selected_method_order:
            raise ValueError("comparison selection method order is inconsistent")
        candidate_ids = {item.candidate_id for item in self.candidates}
        if any(item.selected_candidate_id not in candidate_ids for item in self.selections):
            raise ValueError("comparison selection references an absent candidate")
        return self


class FinalEvaluationProcedure(FrozenModel):
    schema_version: Literal["aqse.final-evaluation-procedure.v1"] = (
        "aqse.final-evaluation-procedure.v1"
    )
    gate_required: Literal["G5 closed by separate external review"] = (
        "G5 closed by separate external review"
    )
    test_bank_creation: Literal[
        "after authorization only; all 24 TEST lineages; fixed window ordinal 9"
    ] = "after authorization only; all 24 TEST lineages; fixed window ordinal 9"
    refit_policy: Literal["no refit; reconstruct selected TRAIN-only models"] = (
        "no refit; reconstruct selected TRAIN-only models"
    )
    evaluation_count: Literal["one per frozen selected method"] = (
        "one per frozen selected method"
    )
    metrics: tuple[str, ...]
    interval_policy: Literal["stratified-independent-lineage-percentile-95"] = (
        "stratified-independent-lineage-percentile-95"
    )
    interval_resamples: Literal[2000] = 2000
    interval_seed: Literal[1001010] = 1_001_010
    no_winner_predeclared: Literal[True] = True
    negative_or_inconclusive_results_retained: Literal[True] = True


class ModelSelectionFreeze(FrozenModel):
    schema_version: Literal["aqse.model-selection-freeze.v1"] = (
        "aqse.model-selection-freeze.v1"
    )
    freeze_id: str = Field(pattern=r"^aqse-model-selection-freeze-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    protocol_id: str
    protocol_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_id: str
    evaluation_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    selections: tuple[MethodSelection, ...]
    final_evaluation_procedure: FinalEvaluationProcedure
    test_state: Literal["sealed"] = "sealed"
    test_ledger_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class MethodRuntime(FrozenModel):
    method: MethodName
    wall_time_ms: float = Field(ge=0.0)
    evaluated_kernel_coordinates: int = Field(ge=0)


class EvaluationExecutionMetadata(FrozenModel):
    schema_version: Literal["aqse.comparative-evaluation-execution.v1"] = (
        "aqse.comparative-evaluation-execution.v1"
    )
    evaluation_id: str
    artifact_path: str
    created_at_utc: str
    repository_base_sha: str
    repository_dirty: bool | None
    runtime_versions: dict[str, str]
    total_wall_time_ms: float = Field(ge=0.0)
    method_runtimes: tuple[MethodRuntime, ...]


class ProtocolExecutionMetadata(FrozenModel):
    schema_version: Literal["aqse.comparative-protocol-execution.v1"] = (
        "aqse.comparative-protocol-execution.v1"
    )
    protocol_id: str
    artifact_path: str
    created_at_utc: str
    repository_base_sha: str
    repository_dirty: bool | None
    runtime_versions: dict[str, str]


class SelectionFreezeExecutionMetadata(FrozenModel):
    schema_version: Literal["aqse.model-selection-freeze-execution.v1"] = (
        "aqse.model-selection-freeze-execution.v1"
    )
    freeze_id: str
    artifact_path: str
    created_at_utc: str
    repository_base_sha: str
    repository_dirty: bool | None
    runtime_versions: dict[str, str]
