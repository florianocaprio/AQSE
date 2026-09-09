from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from app.training.evaluation_models import MethodName, MethodSelection
from app.training.models import FrozenModel


class SemanticTestLoad(FrozenModel):
    sequence: Literal[1, 2]
    operation: Literal["load_observations(TEST)", "load_labels(TEST)"]
    reason: str = Field(min_length=1, max_length=300)


class G5AuthorizationArtifact(FrozenModel):
    schema_version: Literal["aqse.g5-authorization.v1"] = "aqse.g5-authorization.v1"
    authorization_id: str = Field(pattern=r"^aqse-g5-authorization-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    milestone: Literal["1D.4b"] = "1D.4b"
    gate_state: Literal["G5 closed before TEST opening"] = (
        "G5 closed before TEST opening"
    )
    entry_commit: Literal["81045762d69c6a82c63e47196a2ae36899786233"]
    purpose: Literal["milestone-1d4b-final-held-out-evaluation"]
    dataset_id: str
    dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    encoder_id: str
    encoder_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    train_bank_id: str
    train_bank_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    protocol_id: str
    protocol_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_id: str
    evaluation_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    freeze_id: str
    freeze_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    initial_ledger_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    initial_ledger_event_count: Literal[1] = 1
    test_lineage_count: Literal[24] = 24
    window_ordinal: Literal[9] = 9
    cohort_rule: Literal["all-TEST-lineages-fixed-window-ordinal-9"] = (
        "all-TEST-lineages-fixed-window-ordinal-9"
    )
    selection_independent_of_test_semantics: Literal[True] = True
    no_fallback: Literal[True] = True
    common_eligible_cohort: Literal[True] = True
    selections: tuple[MethodSelection, ...]
    metrics: tuple[str, ...]
    bootstrap_resamples: Literal[2000] = 2000
    bootstrap_seed: Literal[1001010] = 1_001_010
    bootstrap_policy: Literal["stratified-independent-test-lineage-percentile-95"] = (
        "stratified-independent-test-lineage-percentile-95"
    )
    expected_semantic_loads: tuple[SemanticTestLoad, SemanticTestLoad]
    generation_plan_access_allowed: Literal[False] = False
    no_refit: Literal[True] = True
    no_training_rerun: Literal[True] = True
    no_winner_predeclared: Literal[True] = True
    contains_test_values: Literal[False] = False

    @model_validator(mode="after")
    def validate_authorization(self) -> G5AuthorizationArtifact:
        methods = tuple(item.method for item in self.selections)
        if methods != (
            "snr_threshold",
            "rbf_svc_circular",
            "fixed_theta_tqk",
            "gradient_tqk",
            "qng_tqk",
        ):
            raise ValueError("G5 authorization must bind the exact five frozen methods")
        if tuple(item.sequence for item in self.expected_semantic_loads) != (1, 2):
            raise ValueError("G5 authorization must bind exactly two ordered TEST loads")
        if tuple(item.operation for item in self.expected_semantic_loads) != (
            "load_observations(TEST)",
            "load_labels(TEST)",
        ):
            raise ValueError("G5 authorization semantic TEST load plan is invalid")
        return self


class TestBankRowReference(FrozenModel):
    episode_id: str
    lineage_id: str
    partition: Literal["test"] = "test"
    window_ordinal: Literal[9] = 9
    window_id: str
    valid_for_quantum: bool
    abstention_reason: Literal["fixed_ordinal_9_not_valid_for_quantum"] | None = None

    @model_validator(mode="after")
    def validate_disposition(self) -> TestBankRowReference:
        if self.valid_for_quantum == (self.abstention_reason is not None):
            raise ValueError("TEST row eligibility and abstention reason are inconsistent")
        return self


class TestBankArtifact(FrozenModel):
    schema_version: Literal["aqse.test-bank.v1"] = "aqse.test-bank.v1"
    artifact_id: str = Field(pattern=r"^aqse-test-bank-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    authorization_id: str
    authorization_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_dataset_id: str
    source_dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    encoder_id: str
    encoder_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_partition: Literal["test"] = "test"
    selection_policy: Literal["all-TEST-lineages-fixed-window-ordinal-9"] = (
        "all-TEST-lineages-fixed-window-ordinal-9"
    )
    selection_independent_of_labels: Literal[True] = True
    no_fallback: Literal[True] = True
    window_ordinal: Literal[9] = 9
    row_count: Literal[24] = 24
    lineage_count: Literal[24] = 24
    eligible_count: int = Field(ge=0, le=24)
    abstained_count: int = Field(ge=0, le=24)
    rows: tuple[TestBankRowReference, ...]

    @model_validator(mode="after")
    def validate_rows(self) -> TestBankArtifact:
        if len(self.rows) != 24:
            raise ValueError("TEST bank must contain all 24 fixed row references")
        if len({item.episode_id for item in self.rows}) != 24:
            raise ValueError("TEST bank episode references must be distinct")
        if len({item.lineage_id for item in self.rows}) != 24:
            raise ValueError("TEST bank lineage references must be distinct")
        if tuple(item.episode_id for item in self.rows) != tuple(
            sorted(item.episode_id for item in self.rows)
        ):
            raise ValueError("TEST bank row order must be deterministic")
        eligible = sum(item.valid_for_quantum for item in self.rows)
        if eligible != self.eligible_count or 24 - eligible != self.abstained_count:
            raise ValueError("TEST bank eligibility accounting is inconsistent")
        return self


class HeldOutInputIdentity(FrozenModel):
    schema_version: Literal["aqse.held-out-input.v1"] = "aqse.held-out-input.v1"
    fingerprint_id: str = Field(pattern=r"^aqse-held-out-input-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    train_encoded_identity: dict[str, object]
    train_label_identity: dict[str, object]
    test_observation_tensor_identity: dict[str, object]
    test_observation_content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    test_raw_row_identity: dict[str, object]
    test_encoded_row_identity: dict[str, object]
    test_label_identity: dict[str, object]
    eligible_test_label_identity: dict[str, object]
    eligible_lineage_ids: tuple[str, ...]
    eligible_window_ids: tuple[str, ...]
    abstained_lineage_ids: tuple[str, ...]
    abstained_window_ids: tuple[str, ...]
    quantum_test_train_kernel_shape: tuple[int, int]


class HeldOutClassificationMetrics(FrozenModel):
    balanced_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    roc_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    negative_support: int = Field(ge=0)
    positive_support: int = Field(ge=0)
    negative_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    positive_recall: float | None = Field(default=None, ge=0.0, le=1.0)


class HeldOutMetricInterval(FrozenModel):
    metric: Literal["balanced_accuracy", "macro_f1"]
    lower: float | None = Field(default=None, ge=0.0, le=1.0)
    point: float | None = Field(default=None, ge=0.0, le=1.0)
    upper: float | None = Field(default=None, ge=0.0, le=1.0)
    resamples: Literal[2000] = 2000
    seed: Literal[1001010] = 1_001_010
    unit: Literal["independent_test_lineage"] = "independent_test_lineage"


class MethodComputeCost(FrozenModel):
    train_rows: Literal[32] = 32
    test_rows: int = Field(ge=0, le=24)
    model_fit_count: int = Field(ge=0, le=1)
    prediction_count: int = Field(ge=0, le=24)
    train_kernel_coordinates: int = Field(ge=0)
    test_train_kernel_coordinates: int = Field(ge=0)


class HeldOutMethodResult(FrozenModel):
    method: MethodName
    frozen_selection: MethodSelection
    metrics: HeldOutClassificationMetrics
    confidence_intervals: tuple[HeldOutMetricInterval, HeldOutMetricInterval]
    eligible_count: int = Field(ge=0, le=24)
    abstained_count: int = Field(ge=0, le=24)
    coverage: float = Field(ge=0.0, le=1.0)
    prediction_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    score_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    compute_cost: MethodComputeCost


class HeldOutMethodRuntime(FrozenModel):
    method: MethodName
    wall_time_ms: float = Field(ge=0.0)


class LedgerEventIdentity(FrozenModel):
    sequence: int = Field(ge=0)
    event: Literal["sealed", "opened"]
    actor: str
    reason: str
    fixture_only: bool
    entry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class FinalHeldOutArtifact(FrozenModel):
    schema_version: Literal["aqse.final-held-out-evaluation.v1"] = (
        "aqse.final-held-out-evaluation.v1"
    )
    artifact_id: str = Field(pattern=r"^aqse-final-held-out-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    milestone: Literal["1D.4b"] = "1D.4b"
    authorization_id: str
    authorization_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_id: str
    dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    protocol_id: str
    protocol_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_id: str
    evaluation_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    freeze_id: str
    freeze_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    encoder_id: str
    encoder_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    train_bank_id: str
    train_bank_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    test_bank_id: str
    test_bank_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    held_out_input: HeldOutInputIdentity
    total_test_lineages: Literal[24] = 24
    eligible_count: int = Field(ge=0, le=24)
    abstained_count: int = Field(ge=0, le=24)
    coverage: float = Field(ge=0.0, le=1.0)
    selections: tuple[MethodSelection, ...]
    results: tuple[HeldOutMethodResult, ...]
    initial_ledger_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    final_ledger_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    final_ledger_event_count: Literal[3] = 3
    ledger_events: tuple[LedgerEventIdentity, LedgerEventIdentity, LedgerEventIdentity]
    semantic_test_load_count: Literal[2] = 2
    generation_plan_access_count: Literal[0] = 0
    no_winner_predeclared: Literal[True] = True
    test_results_not_used_for_model_selection: Literal[True] = True
    no_test_refit: Literal[True] = True
    fixed_gradient_qng_identical: Literal[True] = True
    protected_source_hashes: dict[str, str]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_final_record(self) -> FinalHeldOutArtifact:
        methods = tuple(item.method for item in self.results)
        if methods != (
            "snr_threshold",
            "rbf_svc_circular",
            "fixed_theta_tqk",
            "gradient_tqk",
            "qng_tqk",
        ):
            raise ValueError("held-out result order differs from the frozen five methods")
        if tuple(item.method for item in self.selections) != methods:
            raise ValueError("held-out results differ from the frozen method order")
        if self.eligible_count + self.abstained_count != 24:
            raise ValueError("held-out coverage accounting is inconsistent")
        if any(
            item.eligible_count != self.eligible_count
            or item.abstained_count != self.abstained_count
            or item.coverage != self.coverage
            for item in self.results
        ):
            raise ValueError("all methods must use the same eligible TEST cohort")
        if tuple(item.sequence for item in self.ledger_events) != (0, 1, 2):
            raise ValueError("final TEST ledger must contain exactly sequences 0, 1 and 2")
        return self


class ArtifactExecutionMetadata(FrozenModel):
    schema_version: Literal["aqse.held-out-artifact-execution.v1"] = (
        "aqse.held-out-artifact-execution.v1"
    )
    artifact_kind: Literal["g5_authorization", "test_bank"]
    artifact_id: str
    artifact_path: str
    created_at_utc: str
    repository_base_sha: str
    repository_dirty: bool | None
    runtime_versions: dict[str, str]


class FinalHeldOutExecutionMetadata(FrozenModel):
    schema_version: Literal["aqse.final-held-out-execution.v1"] = (
        "aqse.final-held-out-execution.v1"
    )
    artifact_id: str
    artifact_path: str
    created_at_utc: str
    repository_base_sha: str
    repository_dirty: bool | None
    runtime_versions: dict[str, str]
    total_wall_time_ms: float = Field(ge=0.0)
    method_runtimes: tuple[HeldOutMethodRuntime, ...]
