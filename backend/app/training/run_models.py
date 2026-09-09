from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, model_validator

from app.training.encoding_models import BankRowReference
from app.training.models import FrozenModel


class TrainingInputIdentity(FrozenModel):
    schema_version: Literal["aqse.training-input.v1"] = "aqse.training-input.v1"
    fingerprint_id: str = Field(pattern=r"^aqse-training-input-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_dataset_id: str
    source_dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    encoding_policy_id: Literal["aqse.tqk8.encoding.phase-direct.v1"]
    encoder_artifact_id: str
    encoder_content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    scaler_id: str
    train_bank_id: str
    train_bank_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    label_policy: Literal["aqse.white-noise-regime.v1"]
    ordered_rows: tuple[BankRowReference, ...]
    encoded_matrix_identity: dict[str, Any]
    ordered_label_identity: dict[str, Any]

    @model_validator(mode="after")
    def validate_shape_contract(self) -> TrainingInputIdentity:
        if len(self.ordered_rows) != 32:
            raise ValueError("training-input identity must bind exactly 32 rows")
        if self.encoded_matrix_identity.get("shape") != [32, 8]:
            raise ValueError("training-input matrix identity must have shape 32x8")
        if self.ordered_label_identity.get("shape") != [32]:
            raise ValueError("training-input label identity must have shape 32")
        return self


class OptimizerContract(FrozenModel):
    name: Literal["protected.fit_qng"] = "protected.fit_qng"
    maximum_updates: Literal[10] = 10
    learning_rate: Literal[0.2] = 0.2
    damping: Literal[0.001] = 0.001
    maximum_step_norm: Literal[0.4] = 0.4
    armijo_policy: Literal["protected-backtracking-unchanged"] = (
        "protected-backtracking-unchanged"
    )


class EvaluationCounters(FrozenModel):
    differential_calls: int = Field(ge=0)
    gram_calls: int = Field(ge=0)
    statevector_evaluations: int = Field(ge=0)


class TrainingStepRecord(FrozenModel):
    step_index: int = Field(ge=0)
    theta_before: tuple[float, ...] = Field(min_length=16, max_length=16)
    theta_after: tuple[float, ...] = Field(min_length=16, max_length=16)
    loss_before: float
    loss_after: float
    gradient_norm: float = Field(ge=0.0)
    step_norm: float = Field(ge=0.0)
    metric_min_eigenvalue: float
    step_wall_time_ms: float = Field(ge=0.0)
    counters_delta: EvaluationCounters


class TrainingRunArtifact(FrozenModel):
    schema_version: Literal["aqse.qng-training-run.v1"] = "aqse.qng-training-run.v1"
    run_id: str = Field(pattern=r"^aqse-qng-run-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    scientific_scope: Literal["frozen TRAIN candidate-theta trajectory only"] = (
        "frozen TRAIN candidate-theta trajectory only"
    )
    training_input: TrainingInputIdentity
    backend_semantics: Literal["numpy_statevector"] = "numpy_statevector"
    exact_simulator: Literal[True] = True
    tqk8_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    theta_init_policy_id: Literal["aqse.theta-init.uniform-v1"] = (
        "aqse.theta-init.uniform-v1"
    )
    theta_seed: Literal[1001005] = 1_001_005
    theta0: tuple[float, ...] = Field(min_length=16, max_length=16)
    optimizer: OptimizerContract
    requested_maximum_updates: Literal[10] = 10
    accepted_update_count: int = Field(ge=0, le=10)
    stop_reason: Literal["MAX_UPDATES_REACHED", "NO_ACCEPTED_UPDATE", "CANCELLED"]
    initial_train_alignment_loss: float
    final_train_alignment_loss: float
    steps: tuple[TrainingStepRecord, ...]
    final_theta: tuple[float, ...] = Field(min_length=16, max_length=16)
    total_wall_time_ms: float = Field(ge=0.0)
    peak_process_rss_bytes: int = Field(ge=0)
    counters: EvaluationCounters
    effective_source_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_trajectory(self) -> TrainingRunArtifact:
        if len(self.steps) != self.accepted_update_count:
            raise ValueError("accepted-update count differs from trajectory length")
        if tuple(step.step_index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("training step indices must be monotonic from zero")
        previous = self.theta0
        for step in self.steps:
            if step.theta_before != previous:
                raise ValueError("training theta trajectory is discontinuous")
            if step.loss_after > step.loss_before + 1.0e-14:
                raise ValueError("accepted QNG history cannot increase protected loss")
            previous = step.theta_after
        if self.final_theta != previous:
            raise ValueError("final theta differs from the final trajectory coordinate")
        expected_loss = (
            self.steps[-1].loss_after if self.steps else self.initial_train_alignment_loss
        )
        if self.final_train_alignment_loss != expected_loss:
            raise ValueError("final loss differs from the protected trajectory")
        if self.stop_reason == "MAX_UPDATES_REACHED" and self.accepted_update_count != 10:
            raise ValueError("maximum-update stop requires ten accepted updates")
        if self.stop_reason == "NO_ACCEPTED_UPDATE" and self.accepted_update_count >= 10:
            raise ValueError("no-update stop cannot follow the full update budget")
        if not self.effective_source_hashes:
            raise ValueError("training run effective source hashes are required")
        return self


class TrainingRunExecutionMetadata(FrozenModel):
    schema_version: Literal["aqse.qng-training-execution.v1"] = (
        "aqse.qng-training-execution.v1"
    )
    run_id: str
    artifact_path: str
    created_at_utc: str
    repository_base_sha: str
    repository_dirty: bool | None
    runtime_versions: dict[str, str]


class TrainingJobState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class TrainingJobView(FrozenModel):
    job_id: str
    state: TrainingJobState
    created_at_utc: str
    updated_at_utc: str
    accepted_update_count: int = Field(ge=0, le=10)
    stop_reason: str | None = None
    run_artifact_id: str | None = None
    error: str | None = Field(default=None, max_length=500)
