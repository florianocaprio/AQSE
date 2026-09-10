from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from app.training.canonical import canonical_json_bytes
from app.training.models import FrozenModel
from app.training.run_models import EvaluationCounters, TrainingRunArtifact


class TrajectoryInputIdentity(FrozenModel):
    fingerprint_id: str = Field(pattern=r"^aqse-training-input-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_dataset_id: str
    source_dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    encoding_policy_id: str
    encoder_artifact_id: str
    encoder_content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    scaler_id: str
    train_bank_id: str
    train_bank_digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class TrajectoryOptimizerContract(FrozenModel):
    name: str
    maximum_updates: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    damping: float = Field(ge=0.0)
    maximum_step_norm: float = Field(gt=0.0)
    armijo_policy: str


class TrajectoryStep(FrozenModel):
    step_index: int = Field(ge=0)
    theta_before: tuple[float, ...] = Field(min_length=16, max_length=16)
    theta_after: tuple[float, ...] = Field(min_length=16, max_length=16)
    loss_before: float
    loss_after: float
    gradient_norm: float = Field(ge=0.0)
    step_norm: float = Field(ge=0.0)
    metric_min_eigenvalue: float
    counters_delta: EvaluationCounters


class QngTrajectoryContent(FrozenModel):
    schema_version: Literal["aqse.qng-trajectory.v1"] = "aqse.qng-trajectory.v1"
    training_input: TrajectoryInputIdentity
    backend_semantics: str
    tqk8_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    theta_init_policy_id: str
    theta_seed: int = Field(ge=0)
    theta0: tuple[float, ...] = Field(min_length=16, max_length=16)
    optimizer: TrajectoryOptimizerContract
    requested_maximum_updates: int = Field(gt=0)
    accepted_update_count: int = Field(ge=0)
    stop_reason: str
    initial_train_alignment_loss: float
    final_train_alignment_loss: float
    steps: tuple[TrajectoryStep, ...]
    final_theta: tuple[float, ...] = Field(min_length=16, max_length=16)
    counters: EvaluationCounters

    @model_validator(mode="after")
    def validate_trajectory(self) -> QngTrajectoryContent:
        if self.requested_maximum_updates != self.optimizer.maximum_updates:
            raise ValueError("trajectory update budgets are inconsistent")
        if len(self.steps) != self.accepted_update_count:
            raise ValueError("trajectory accepted-update count is inconsistent")
        if self.accepted_update_count > self.requested_maximum_updates:
            raise ValueError("trajectory exceeds its requested update budget")
        if tuple(step.step_index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("trajectory step indices are not contiguous")
        previous = self.theta0
        for step in self.steps:
            if step.theta_before != previous:
                raise ValueError("trajectory theta coordinates are discontinuous")
            previous = step.theta_after
        if self.final_theta != previous:
            raise ValueError("trajectory final theta is inconsistent")
        expected_loss = (
            self.steps[-1].loss_after
            if self.steps
            else self.initial_train_alignment_loss
        )
        if self.final_train_alignment_loss != expected_loss:
            raise ValueError("trajectory final loss is inconsistent")
        return self


class QngTrajectoryIdentity(FrozenModel):
    schema_version: Literal["aqse.qng-trajectory-identity.v1"] = (
        "aqse.qng-trajectory-identity.v1"
    )
    trajectory_id: str = Field(pattern=r"^aqse-qng-trajectory-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    content: QngTrajectoryContent


class TrajectoryAuditMapping(FrozenModel):
    schema_version: Literal["aqse.qng-trajectory-audit.v1"] = (
        "aqse.qng-trajectory-audit.v1"
    )
    mapping_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    trajectory: QngTrajectoryIdentity
    designated_execution_id: str = Field(pattern=r"^aqse-qng-run-[a-f0-9]{16}$")
    equivalent_execution_ids: tuple[str, ...]
    execution_run_sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_execution_links(self) -> TrajectoryAuditMapping:
        if self.designated_execution_id in self.equivalent_execution_ids:
            raise ValueError("designated execution cannot also be an equivalent entry")
        linked = {self.designated_execution_id, *self.equivalent_execution_ids}
        if set(self.execution_run_sha256) != linked:
            raise ValueError("trajectory audit run hashes do not match linked executions")
        if any(len(value) != 64 for value in self.execution_run_sha256.values()):
            raise ValueError("trajectory audit contains an invalid run SHA-256")
        return self


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def derive_trajectory_identity(run: TrainingRunArtifact) -> QngTrajectoryIdentity:
    input_identity = run.training_input
    content = QngTrajectoryContent(
        training_input=TrajectoryInputIdentity(
            fingerprint_id=input_identity.fingerprint_id,
            content_digest=input_identity.content_digest,
            source_dataset_id=input_identity.source_dataset_id,
            source_dataset_digest=input_identity.source_dataset_digest,
            feature_profile_fingerprint=input_identity.feature_profile_fingerprint,
            encoding_policy_id=input_identity.encoding_policy_id,
            encoder_artifact_id=input_identity.encoder_artifact_id,
            encoder_content_digest=input_identity.encoder_content_digest,
            scaler_id=input_identity.scaler_id,
            train_bank_id=input_identity.train_bank_id,
            train_bank_digest=input_identity.train_bank_digest,
        ),
        backend_semantics=run.backend_semantics,
        tqk8_source_sha256=run.tqk8_source_sha256,
        theta_init_policy_id=run.theta_init_policy_id,
        theta_seed=run.theta_seed,
        theta0=run.theta0,
        optimizer=TrajectoryOptimizerContract(
            name=run.optimizer.name,
            maximum_updates=run.optimizer.maximum_updates,
            learning_rate=run.optimizer.learning_rate,
            damping=run.optimizer.damping,
            maximum_step_norm=run.optimizer.maximum_step_norm,
            armijo_policy=run.optimizer.armijo_policy,
        ),
        requested_maximum_updates=run.requested_maximum_updates,
        accepted_update_count=run.accepted_update_count,
        stop_reason=run.stop_reason,
        initial_train_alignment_loss=run.initial_train_alignment_loss,
        final_train_alignment_loss=run.final_train_alignment_loss,
        steps=tuple(
            TrajectoryStep(
                step_index=step.step_index,
                theta_before=step.theta_before,
                theta_after=step.theta_after,
                loss_before=step.loss_before,
                loss_after=step.loss_after,
                gradient_norm=step.gradient_norm,
                step_norm=step.step_norm,
                metric_min_eigenvalue=step.metric_min_eigenvalue,
                counters_delta=step.counters_delta,
            )
            for step in run.steps
        ),
        final_theta=run.final_theta,
        counters=run.counters,
    )
    return build_trajectory_identity(content)


def build_trajectory_identity(
    content: QngTrajectoryContent,
) -> QngTrajectoryIdentity:
    validated = QngTrajectoryContent.model_validate(content.model_dump(mode="json"))
    digest = _digest(validated.model_dump(mode="json"))
    return QngTrajectoryIdentity(
        trajectory_id=f"aqse-qng-trajectory-{digest[:16]}",
        content_digest=digest,
        content=validated,
    )


def validate_trajectory_identity(identity: QngTrajectoryIdentity) -> None:
    expected = _digest(identity.content.model_dump(mode="json"))
    if identity.content_digest != expected:
        raise ValueError("QNG trajectory content digest is invalid")
    if identity.trajectory_id != f"aqse-qng-trajectory-{expected[:16]}":
        raise ValueError("QNG trajectory identity is invalid")


def build_trajectory_audit_mapping(
    trajectory: QngTrajectoryIdentity,
    *,
    designated_execution_id: str,
    equivalent_execution_ids: tuple[str, ...],
    execution_run_sha256: dict[str, str],
) -> TrajectoryAuditMapping:
    validate_trajectory_identity(trajectory)
    payload = {
        "schema_version": "aqse.qng-trajectory-audit.v1",
        "trajectory": trajectory.model_dump(mode="json"),
        "designated_execution_id": designated_execution_id,
        "equivalent_execution_ids": equivalent_execution_ids,
        "execution_run_sha256": execution_run_sha256,
    }
    return TrajectoryAuditMapping(
        **payload,
        mapping_digest=_digest(payload),
    )


def validate_trajectory_audit_mapping(mapping: TrajectoryAuditMapping) -> None:
    validate_trajectory_identity(mapping.trajectory)
    payload = mapping.model_dump(mode="json", exclude={"mapping_digest"})
    if mapping.mapping_digest != _digest(payload):
        raise ValueError("QNG trajectory audit mapping digest is invalid")
