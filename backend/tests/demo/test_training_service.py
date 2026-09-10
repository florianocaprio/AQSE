from __future__ import annotations

from time import monotonic, sleep
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.demo.bundle_models import (
    LOCAL_TASK_ID,
    NETWORK_TASK_ID,
    DemoTaskExample,
    DemoTaskPartition,
)
from app.demo.evaluation import DemoTrainingCancelled
from app.demo.runtime_models import DemoTrainingIntent
from app.demo.training_service import (
    DemoTrainingBusyError,
    DemoTrainingJobService,
    _knowledge_training_partitions,
)
from app.features.state8 import state8_profile, state8_profile_fingerprint


def _intent(suffix: str) -> DemoTrainingIntent:
    return DemoTrainingIntent(
        intent_id=f"aqse-demo-intent-{suffix}",
        study_artifact_id="aqse-network-study-fixture",
    )


def _partition(
    task_id: str,
    partition: str,
    *,
    study_id: str,
    digest_character: str,
) -> DemoTaskPartition:
    local = task_id == LOCAL_TASK_ID
    profile_id = "aqse.local-state8.v1" if local else "aqse.network-state8.v1"
    class_order = (
        ("NORMAL", "CHANGE_DETECTED")
        if local
        else (
            "NORMAL",
            "ENVIRONMENT_COMPATIBLE",
            "DEVICE_COMPATIBLE",
            "MIXED_OR_AMBIGUOUS",
        )
    )
    profile = state8_profile(profile_id)
    return DemoTaskPartition(
        study_artifact_id=study_id,
        study_content_digest=digest_character * 64,
        partition=partition,
        task_id=task_id,
        profile_id=profile_id,
        profile_fingerprint=state8_profile_fingerprint(profile),
        class_order=class_order,
        examples=(
            DemoTaskExample(
                sample_id=f"{study_id}-{task_id}-{partition}",
                episode_id=f"episode-{study_id}-{task_id}-{partition}",
                generative_lineage_id=(
                    f"lineage-{study_id}-{task_id}-{partition}"
                ),
                node_count=1 if local else 3,
                label="NORMAL",
                feature_values=(1.0, 0.5, 0.1, -2.0, 0.0, 0.25, 0.75, 1.0),
                eligible=True,
            ),
        ),
    )


def test_training_intent_is_idempotent_busy_guarded_and_cancellable(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AQSE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "app.demo.training_service.find_current_study",
        lambda: (
            tmp_path / "study",
            SimpleNamespace(artifact_id="aqse-network-study-fixture"),
        ),
    )
    monkeypatch.setattr(
        "app.demo.training_service.load_task_partitions",
        lambda *_args, **_kwargs: (object(), object()),
    )

    def cancellable_fit(
        _train,
        _validation,
        *,
        cancellation,
        progress,
    ):
        del progress
        while not cancellation.is_set():
            sleep(0.01)
        raise DemoTrainingCancelled("fixture cancellation")

    monkeypatch.setattr(
        "app.demo.training_service.fit_demo_task_candidates",
        cancellable_fit,
    )
    service = DemoTrainingJobService()
    intent = _intent("same-click")

    first, created = service.start(intent)
    duplicate, duplicate_created = service.start(intent)

    assert created
    assert not duplicate_created
    assert duplicate.job_id == first.job_id
    with pytest.raises(DemoTrainingBusyError):
        service.start(_intent("different-click"))

    cancelled = service.cancel(first.job_id)
    assert cancelled is not None
    deadline = monotonic() + 2.0
    terminal = service.get(first.job_id)
    while terminal is not None and terminal.state != "cancelled" and monotonic() < deadline:
        sleep(0.01)
        terminal = service.get(first.job_id)
    assert terminal is not None
    assert terminal.state == "cancelled"
    assert terminal.selection_freeze_id is None

    restarted = DemoTrainingJobService()
    loaded, loaded_created = restarted.start(intent)
    assert not loaded_created
    assert loaded == terminal
    service.clear_for_tests()


def test_training_intent_requires_an_atomic_local_and_network_collection_pair() -> None:
    common = {
        "intent_id": "aqse-demo-intent-knowledge-pair",
        "study_artifact_id": "aqse-network-study-fixture",
    }
    local_id = f"aqse-knowledge-train-{'1' * 16}"
    network_id = f"aqse-knowledge-train-{'2' * 16}"

    with pytest.raises(ValidationError, match="atomic bundle pair"):
        DemoTrainingIntent(**common, local_train_collection_id=local_id)
    with pytest.raises(ValidationError, match="atomic bundle pair"):
        DemoTrainingIntent(**common, network_train_collection_id=network_id)

    paired = DemoTrainingIntent(
        **common,
        local_train_collection_id=local_id,
        network_train_collection_id=network_id,
    )
    assert paired.local_train_collection_id == local_id
    assert paired.network_train_collection_id == network_id


def test_knowledge_retraining_loader_uses_approved_train_and_frozen_validation_only(
    monkeypatch,
) -> None:
    canonical_study_id = "aqse-network-study-fixture"
    local_collection_id = f"aqse-knowledge-train-{'1' * 16}"
    network_collection_id = f"aqse-knowledge-train-{'2' * 16}"
    approved_local = _partition(
        LOCAL_TASK_ID,
        "train",
        study_id=local_collection_id,
        digest_character="b",
    )
    approved_network = _partition(
        NETWORK_TASK_ID,
        "train",
        study_id=network_collection_id,
        digest_character="c",
    )
    frozen_local_validation = _partition(
        LOCAL_TASK_ID,
        "validation",
        study_id=canonical_study_id,
        digest_character="a",
    )
    frozen_network_validation = _partition(
        NETWORK_TASK_ID,
        "validation",
        study_id=canonical_study_id,
        digest_character="a",
    )
    loaded_collection_ids: list[str] = []

    def load_collection(collection_id: str) -> DemoTaskPartition:
        loaded_collection_ids.append(collection_id)
        return {
            local_collection_id: approved_local,
            network_collection_id: approved_network,
        }[collection_id]

    monkeypatch.setattr(
        "app.demo.training_service.load_approved_train_partition",
        load_collection,
    )
    intent = DemoTrainingIntent(
        intent_id="aqse-demo-intent-approved-knowledge",
        study_artifact_id=canonical_study_id,
        local_train_collection_id=local_collection_id,
        network_train_collection_id=network_collection_id,
    )

    loaded = _knowledge_training_partitions(
        intent,
        frozen_local_validation,
        frozen_network_validation,
    )

    assert loaded is not None
    local_train, network_train, local_validation, network_validation = loaded
    assert loaded_collection_ids == [local_collection_id, network_collection_id]
    assert local_train.partition == network_train.partition == "train"
    assert local_train.examples == approved_local.examples
    assert network_train.examples == approved_network.examples
    assert local_validation.partition == network_validation.partition == "validation"
    assert local_validation.examples == frozen_local_validation.examples
    assert network_validation.examples == frozen_network_validation.examples
    assert len({item.study_artifact_id for item in loaded}) == 1
    assert len({item.study_content_digest for item in loaded}) == 1
    assert all("test" not in item.partition.lower() for item in loaded)
