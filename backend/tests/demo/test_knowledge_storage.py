from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.demo.knowledge_models import (
    KnowledgeTask,
    KnowledgeTrainApprovalRequest,
)
from app.demo.knowledge_storage import (
    KnowledgeApprovalError,
    KnowledgeConflictError,
    KnowledgeStoreError,
    approve_train_collection,
    capture_observation_episode,
    create_reviewed_label,
    knowledge_root,
    load_approved_train_partition,
    write_observation_episode,
    write_reviewed_label,
)
from app.features.state8 import state8_profile, state8_profile_fingerprint
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
    State8FeatureQuality,
    State8FeatureRecord,
)


def _feature(
    ordinal: int,
    *,
    network: bool = False,
    fingerprint: str | None = None,
) -> State8FeatureRecord:
    profile_id = NETWORK_STATE8_PROFILE_ID if network else LOCAL_STATE8_PROFILE_ID
    profile = state8_profile(profile_id)
    sensor_id = "S1"
    return State8FeatureRecord(
        window_id=f"knowledge-fixture-{ordinal}",
        session_id=f"knowledge-session-{ordinal}",
        sensor_id=sensor_id,
        profile_id=profile_id,
        profile_fingerprint=(fingerprint or state8_profile_fingerprint(profile)),
        reference_id=f"aqse-state8-reference-{ordinal:016x}",
        start_time_s=8.0 + ordinal,
        end_exclusive_time_s=12.0 + ordinal,
        source_frame_ids=tuple(range(1, 401)),
        peer_sensor_ids=("S2", "S3") if network else (),
        values=(1.0, 0.5, 0.1, -2.0, 0.0, 0.25, 0.75, 1.0),
        quality=State8FeatureQuality(
            valid_for_quantum=True,
            flags=(),
            per_feature_valid=(True, True, True, True, True, True, True, True),
            received_sample_count=400,
            usable_sample_count=400,
        ),
    )


def _capture_and_label(
    root: Path,
    ordinal: int,
    label: str,
) -> str:
    observation = capture_observation_episode(
        _feature(ordinal),
        task=KnowledgeTask.LOCAL,
        node_count=1,
    )
    write_observation_episode(observation, root=root)
    reviewed = create_reviewed_label(
        observation,
        label=label,
        reviewer_id="reviewer-fixture",
        reviewed_at_utc=f"2026-09-10T10:00:{ordinal:02d}Z",
    )
    write_reviewed_label(reviewed, root=root)
    return observation.observation_id


def _approval(*observation_ids: str) -> KnowledgeTrainApprovalRequest:
    return KnowledgeTrainApprovalRequest(
        task=KnowledgeTask.LOCAL,
        observation_ids=tuple(sorted(observation_ids)),
        approved_by="principal-investigator",
        approved_at_utc="2026-09-10T11:00:00Z",
    )


def test_observation_and_human_label_are_structurally_separate(
    tmp_path: Path,
) -> None:
    observation = capture_observation_episode(
        _feature(1),
        task=KnowledgeTask.LOCAL,
        node_count=1,
    )
    reviewed = create_reviewed_label(
        observation,
        label="NORMAL",
        reviewer_id="human-reviewer",
        reviewed_at_utc="2026-09-10T10:00:01Z",
    )
    observation_path, _ = write_observation_episode(observation, root=tmp_path)
    label_path, _ = write_reviewed_label(reviewed, root=tmp_path)

    observation_keys = observation.model_dump(mode="json").keys()
    assert (
        not {
            "truth",
            "seed",
            "scenario",
            "prediction",
            "predictions",
            "predicted_class",
            "label",
        }
        & observation_keys
    )
    assert "feature" not in reviewed.model_dump(mode="json")
    assert observation_path.parent.name == "observations"
    assert label_path.parent.name == "labels"
    assert observation_path.parent != label_path.parent


def test_immutable_writes_are_idempotent_and_conflicting_review_is_rejected(
    tmp_path: Path,
) -> None:
    observation = capture_observation_episode(
        _feature(2),
        task=KnowledgeTask.LOCAL,
        node_count=1,
    )
    first_path, first_reused = write_observation_episode(observation, root=tmp_path)
    second_path, second_reused = write_observation_episode(observation, root=tmp_path)
    assert first_path == second_path
    assert not first_reused
    assert second_reused

    first_label = create_reviewed_label(
        observation,
        label="NORMAL",
        reviewer_id="reviewer-a",
        reviewed_at_utc="2026-09-10T10:00:02Z",
    )
    _, first_label_reused = write_reviewed_label(first_label, root=tmp_path)
    _, second_label_reused = write_reviewed_label(first_label, root=tmp_path)
    assert not first_label_reused
    assert second_label_reused

    conflicting_label = create_reviewed_label(
        observation,
        label="CHANGE_DETECTED",
        reviewer_id="reviewer-b",
        reviewed_at_utc="2026-09-10T10:01:02Z",
    )
    with pytest.raises(KnowledgeConflictError, match="different content"):
        write_reviewed_label(conflicting_label, root=tmp_path)


def test_incomplete_train_approval_is_rejected(tmp_path: Path) -> None:
    observation = capture_observation_episode(
        _feature(3),
        task=KnowledgeTask.LOCAL,
        node_count=1,
    )
    write_observation_episode(observation, root=tmp_path)

    with pytest.raises(KnowledgeApprovalError, match="incomplete"):
        approve_train_collection(_approval(observation.observation_id), root=tmp_path)


def test_approved_collection_is_content_addressed_idempotent_and_reloadable(
    tmp_path: Path,
) -> None:
    normal_id = _capture_and_label(tmp_path, 4, "NORMAL")
    change_id = _capture_and_label(tmp_path, 5, "CHANGE_DETECTED")
    request = _approval(normal_id, change_id)

    path, manifest, reused = approve_train_collection(request, root=tmp_path)
    repeated_path, repeated_manifest, repeated = approve_train_collection(
        request,
        root=tmp_path,
    )
    partition = load_approved_train_partition(manifest.collection_id, root=tmp_path)

    assert not reused
    assert repeated
    assert repeated_path == path
    assert repeated_manifest == manifest
    assert path.name == f"{manifest.collection_id}.json"
    assert manifest.collection_id.endswith(manifest.content_digest[:16])
    assert partition.partition == "train"
    assert partition.study_artifact_id == manifest.collection_id
    assert partition.study_content_digest == manifest.content_digest
    assert partition.task_id == "aqse.local-change.v1"
    assert {item.label for item in partition.examples} == {
        "NORMAL",
        "CHANGE_DETECTED",
    }
    assert len({item.generative_lineage_id for item in partition.examples}) == 2


def test_profile_mismatch_is_rejected_before_persistence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incompatible state8 profile"):
        capture_observation_episode(
            _feature(6),
            task=KnowledgeTask.NETWORK,
            node_count=3,
        )
    with pytest.raises(ValueError, match="unknown profile fingerprint"):
        capture_observation_episode(
            _feature(7, fingerprint="0" * 64),
            task=KnowledgeTask.LOCAL,
            node_count=1,
        )
    assert not (tmp_path / "network-demo" / "knowledge").exists()


def test_registry_is_train_only_and_never_creates_test_channel(tmp_path: Path) -> None:
    observation_id = _capture_and_label(tmp_path, 8, "NORMAL")
    _, manifest, _ = approve_train_collection(
        _approval(observation_id),
        root=tmp_path,
    )
    partition = load_approved_train_partition(manifest.collection_id, root=tmp_path)
    names = {path.name.lower() for path in knowledge_root(tmp_path).rglob("*") if path.is_file()}

    assert manifest.partition == "TRAIN"
    assert partition.partition == "train"
    assert not any("test" in name for name in names)
    with pytest.raises(ValidationError):
        KnowledgeTrainApprovalRequest.model_validate(
            {
                "partition": "TEST",
                "task": "local",
                "observation_ids": [observation_id],
                "approved_by": "reviewer",
                "approved_at_utc": "2026-09-10T12:00:00Z",
            }
        )
    with pytest.raises(ValidationError):
        KnowledgeTrainApprovalRequest.model_validate(
            {
                "task": "local",
                "observation_ids": ["../../test-access-ledger"],
                "approved_by": "reviewer",
                "approved_at_utc": "2026-09-10T12:00:00Z",
            }
        )


def test_symlinked_storage_channel_is_rejected(tmp_path: Path) -> None:
    storage = knowledge_root(tmp_path)
    labels = storage / "labels"
    labels.rmdir()
    labels.symlink_to(storage / "observations", target_is_directory=True)

    with pytest.raises(KnowledgeStoreError, match="symlink"):
        knowledge_root(tmp_path)
