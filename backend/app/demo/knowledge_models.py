from __future__ import annotations

import hashlib
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from app.demo.bundle_models import (
    LOCAL_TASK_ID,
    NETWORK_TASK_ID,
    TaskId,
)
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
    State8FeatureRecord,
    State8ProfileId,
)
from app.training.canonical import canonical_json_bytes
from app.training.models import FrozenModel

UTC_TIMESTAMP_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
HUMAN_REVIEW_DECLARATION = "label assigned by explicit human review"
TRAIN_APPROVAL_DECLARATION = "explicitly approved for bounded TRAIN-only retraining"
KnowledgeObservationId = Annotated[
    str,
    Field(pattern=r"^aqse-knowledge-observation-[a-f0-9]{16}$"),
]


class KnowledgeTask(str, Enum):
    """The two frozen prediction tasks accepted by the knowledge registry."""

    LOCAL = "local"
    NETWORK = "network"


class KnowledgeCaptureRequest(FrozenModel):
    """Select one current observation result without accepting model output."""

    schema_version: Literal["aqse.network-demo.knowledge-capture-request.v1"] = (
        "aqse.network-demo.knowledge-capture-request.v1"
    )
    session_id: str = Field(min_length=1, max_length=256)
    sensor_id: str = Field(min_length=1, max_length=128)
    task: KnowledgeTask


def task_id_for(task: KnowledgeTask) -> TaskId:
    return LOCAL_TASK_ID if task is KnowledgeTask.LOCAL else NETWORK_TASK_ID


def profile_id_for(task: KnowledgeTask) -> State8ProfileId:
    return LOCAL_STATE8_PROFILE_ID if task is KnowledgeTask.LOCAL else NETWORK_STATE8_PROFILE_ID


def class_order_for(task: KnowledgeTask) -> tuple[str, ...]:
    if task is KnowledgeTask.LOCAL:
        return ("NORMAL", "CHANGE_DETECTED")
    return (
        "NORMAL",
        "ENVIRONMENT_COMPATIBLE",
        "DEVICE_COMPATIBLE",
        "MIXED_OR_AMBIGUOUS",
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def observation_identity_payload(
    *,
    task: KnowledgeTask,
    node_count: int,
    feature: State8FeatureRecord,
) -> dict[str, Any]:
    return {
        "schema_version": "aqse.network-demo.knowledge-observation.v1",
        "task": task.value,
        "task_id": task_id_for(task),
        "profile_id": profile_id_for(task),
        "profile_fingerprint": feature.profile_fingerprint,
        "node_count": node_count,
        "feature": feature.model_dump(mode="json"),
    }


class KnowledgeObservationEpisode(FrozenModel):
    """Truth-free observed state8 episode captured for later human review."""

    schema_version: Literal["aqse.network-demo.knowledge-observation.v1"] = (
        "aqse.network-demo.knowledge-observation.v1"
    )
    observation_id: KnowledgeObservationId
    acquisition_id: str = Field(pattern=r"^aqse-knowledge-acquisition-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    task: KnowledgeTask
    task_id: TaskId
    profile_id: State8ProfileId
    profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    node_count: int = Field(ge=1, le=8)
    feature: State8FeatureRecord

    @model_validator(mode="after")
    def validate_observation_contract(self) -> KnowledgeObservationEpisode:
        expected_profile = profile_id_for(self.task)
        if self.task_id != task_id_for(self.task):
            raise ValueError("knowledge task and task identifier are incompatible")
        if self.profile_id != expected_profile or self.feature.profile_id != expected_profile:
            raise ValueError("knowledge task and state8 profile are incompatible")
        if self.profile_fingerprint != self.feature.profile_fingerprint:
            raise ValueError("knowledge profile fingerprints are inconsistent")
        if self.task is KnowledgeTask.NETWORK:
            if self.node_count < 3:
                raise ValueError("network knowledge requires at least three nodes")
            if not 2 <= len(self.feature.peer_sensor_ids) <= self.node_count - 1:
                raise ValueError(
                    "network knowledge requires at least two valid peers and cannot "
                    "exceed the session node count"
                )
        payload = observation_identity_payload(
            task=self.task,
            node_count=self.node_count,
            feature=self.feature,
        )
        digest = _digest(payload)
        if self.content_digest != digest:
            raise ValueError("knowledge observation content digest is invalid")
        if self.observation_id != f"aqse-knowledge-observation-{digest[:16]}":
            raise ValueError("knowledge observation identifier is invalid")
        if self.acquisition_id != f"aqse-knowledge-acquisition-{digest[16:32]}":
            raise ValueError("knowledge acquisition identifier is invalid")
        return self


def reviewed_label_identity_payload(
    *,
    observation_id: str,
    task: KnowledgeTask,
    label: str,
    reviewer_id: str,
    reviewed_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": "aqse.network-demo.human-reviewed-label.v1",
        "observation_id": observation_id,
        "task": task.value,
        "task_id": task_id_for(task),
        "label": label,
        "review_source": "human-review",
        "reviewer_id": reviewer_id,
        "reviewed_at_utc": reviewed_at_utc,
        "review_declaration": HUMAN_REVIEW_DECLARATION,
    }


class KnowledgeReviewedLabel(FrozenModel):
    """Independent human-supervision channel; never produced by inference."""

    schema_version: Literal["aqse.network-demo.human-reviewed-label.v1"] = (
        "aqse.network-demo.human-reviewed-label.v1"
    )
    label_id: str = Field(pattern=r"^aqse-human-label-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    observation_id: KnowledgeObservationId
    task: KnowledgeTask
    task_id: TaskId
    label: str = Field(min_length=1, max_length=128)
    review_source: Literal["human-review"] = "human-review"
    reviewer_id: str = Field(min_length=1, max_length=128)
    reviewed_at_utc: str = Field(pattern=UTC_TIMESTAMP_PATTERN)
    review_declaration: Literal["label assigned by explicit human review"] = (
        HUMAN_REVIEW_DECLARATION
    )

    @model_validator(mode="after")
    def validate_review_contract(self) -> KnowledgeReviewedLabel:
        if self.task_id != task_id_for(self.task):
            raise ValueError("reviewed-label task identity is incompatible")
        if self.label not in class_order_for(self.task):
            raise ValueError("human-reviewed label is incompatible with its task")
        payload = reviewed_label_identity_payload(
            observation_id=self.observation_id,
            task=self.task,
            label=self.label,
            reviewer_id=self.reviewer_id,
            reviewed_at_utc=self.reviewed_at_utc,
        )
        digest = _digest(payload)
        if self.content_digest != digest:
            raise ValueError("human-reviewed label content digest is invalid")
        if self.label_id != f"aqse-human-label-{digest[:16]}":
            raise ValueError("human-reviewed label identifier is invalid")
        return self


class KnowledgeReviewedLabelRequest(FrozenModel):
    schema_version: Literal["aqse.network-demo.reviewed-label-request.v1"] = (
        "aqse.network-demo.reviewed-label-request.v1"
    )
    observation_id: KnowledgeObservationId
    task: KnowledgeTask
    label: str = Field(min_length=1, max_length=128)
    reviewer_id: str = Field(min_length=1, max_length=128)
    reviewed_at_utc: str = Field(pattern=UTC_TIMESTAMP_PATTERN)


class KnowledgeTrainApprovalRequest(FrozenModel):
    """Explicit authorization to freeze reviewed observations as TRAIN only."""

    schema_version: Literal["aqse.network-demo.train-approval-request.v1"] = (
        "aqse.network-demo.train-approval-request.v1"
    )
    partition: Literal["TRAIN"] = "TRAIN"
    task: KnowledgeTask
    observation_ids: tuple[KnowledgeObservationId, ...] = Field(
        min_length=1,
        max_length=4096,
    )
    approved_by: str = Field(min_length=1, max_length=128)
    approved_at_utc: str = Field(pattern=UTC_TIMESTAMP_PATTERN)
    approval_declaration: Literal["explicitly approved for bounded TRAIN-only retraining"] = (
        TRAIN_APPROVAL_DECLARATION
    )

    @model_validator(mode="after")
    def validate_explicit_approval(self) -> KnowledgeTrainApprovalRequest:
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("TRAIN approval observation identifiers must be distinct")
        if tuple(sorted(self.observation_ids)) != self.observation_ids:
            raise ValueError("TRAIN approval observation identifiers must be sorted")
        return self


class KnowledgeCollectionEntry(FrozenModel):
    observation_id: KnowledgeObservationId
    observation_content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    observation_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    label_id: str = Field(pattern=r"^aqse-human-label-[a-f0-9]{16}$")
    label_content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    label_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class KnowledgeTrainCollectionManifest(FrozenModel):
    """Immutable content-addressed approval of one human-labelled TRAIN set."""

    schema_version: Literal["aqse.network-demo.train-collection.v1"] = (
        "aqse.network-demo.train-collection.v1"
    )
    collection_id: str = Field(pattern=r"^aqse-knowledge-train-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    partition: Literal["TRAIN"] = "TRAIN"
    task: KnowledgeTask
    task_id: TaskId
    profile_id: State8ProfileId
    profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    class_order: tuple[str, ...] = Field(min_length=2)
    approved_by: str = Field(min_length=1, max_length=128)
    approved_at_utc: str = Field(pattern=UTC_TIMESTAMP_PATTERN)
    approval_declaration: Literal["explicitly approved for bounded TRAIN-only retraining"] = (
        TRAIN_APPROVAL_DECLARATION
    )
    entries: tuple[KnowledgeCollectionEntry, ...] = Field(
        min_length=1,
        max_length=4096,
    )

    @model_validator(mode="after")
    def validate_collection_contract(self) -> KnowledgeTrainCollectionManifest:
        if self.task_id != task_id_for(self.task):
            raise ValueError("knowledge collection task identity is incompatible")
        if self.profile_id != profile_id_for(self.task):
            raise ValueError("knowledge collection profile is incompatible")
        if self.class_order != class_order_for(self.task):
            raise ValueError("knowledge collection class order is incompatible")
        identifiers = tuple(item.observation_id for item in self.entries)
        if identifiers != tuple(sorted(identifiers)) or len(set(identifiers)) != len(identifiers):
            raise ValueError("knowledge collection entries must be sorted and distinct")
        return self


class KnowledgeObservationWriteResponse(FrozenModel):
    observation: KnowledgeObservationEpisode
    reused: bool


class KnowledgeLabelWriteResponse(FrozenModel):
    label: KnowledgeReviewedLabel
    reused: bool


class KnowledgeCollectionWriteResponse(FrozenModel):
    collection: KnowledgeTrainCollectionManifest
    reused: bool


class KnowledgeRegistryView(FrozenModel):
    """Bounded registry view; observations and labels stay distinct records."""

    schema_version: Literal["aqse.network-demo.knowledge-registry.v1"] = (
        "aqse.network-demo.knowledge-registry.v1"
    )
    observations: tuple[KnowledgeObservationEpisode, ...]
    reviewed_labels: tuple[KnowledgeReviewedLabel, ...]
    approved_train_collections: tuple[KnowledgeTrainCollectionManifest, ...]


def train_collection_identity_payload(
    manifest: KnowledgeTrainCollectionManifest,
) -> dict[str, Any]:
    return manifest.model_dump(
        mode="json",
        exclude={"collection_id", "content_digest"},
    )
