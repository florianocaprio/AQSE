from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.four_sensor_study import storage


def _fake_build(*, count: int, partition: str) -> dict[str, Any]:
    episode_ids = tuple(f"fixture-{partition}-episode-{index:03d}" for index in range(count))
    return {
        "episode_plans": [
            {"episode_id": episode_id, "partition": partition}
            for episode_id in episode_ids
        ],
        "observations": [
            {
                "episode_id": episode_id,
                "partition": partition,
                "observed_fixture_value": float(index),
            }
            for index, episode_id in enumerate(episode_ids)
        ],
        "labels": [
            {
                "episode_id": episode_id,
                "partition": partition,
                "scenario": "NORMAL",
            }
            for episode_id in episode_ids
        ],
    }


def _ledger_context() -> tuple[dict[str, object], ...]:
    protocol = {
        "artifact_id": "protocol-freeze-fixture",
        "content_digest": "1" * 64,
    }
    dataset = {
        "artifact_id": "final-test-fixture",
        "content_digest": "2" * 64,
        "bindings": {
            "protocol_freeze_artifact_id": protocol["artifact_id"],
            "protocol_freeze_content_digest": protocol["content_digest"],
        },
    }
    model = {
        "artifact_id": "model-binding-fixture",
        "content_digest": "3" * 64,
    }
    authorization = {
        "artifact_id": "authorization-fixture",
        "content_digest": "4" * 64,
    }
    return dataset, protocol, model, authorization


def test_storage_exposes_only_final_test_semantic_loaders() -> None:
    assert storage.FINAL_TEST_EPISODE_COUNT == 100
    assert not hasattr(storage, "load_development_partition")
    assert not hasattr(storage, "load_train_partition")
    assert not hasattr(storage, "load_validation_partition")


def test_pilot_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    pilot_path, manifest, _reused = storage.write_pilot(
        _fake_build(count=20, partition="pilot"),
        root=tmp_path,
    )
    manifest_path = pilot_path / "manifest.json"
    tampered = json.loads(manifest_path.read_text())
    tampered["episode_count"] = 21
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(tampered))

    assert manifest["episode_count"] == 20
    with pytest.raises(ValueError, match="invalid episode count"):
        storage.verify_pilot(pilot_path)


def test_protocol_bound_final_dataset_starts_sealed_and_rejects_early_labels(
    tmp_path: Path,
) -> None:
    pilot_path, _pilot_manifest, _pilot_reused = storage.write_pilot(
        _fake_build(count=20, partition="pilot"),
        root=tmp_path,
    )
    protocol_path, protocol_manifest, _protocol_reused = storage.write_protocol_freeze(
        {
            "study_id": "aqse-four-sensor-study-v1",
            "final_test_episode_count": 100,
        },
        pilot_path=pilot_path,
        acceptance={"accepted": True, "scope": "structural-feasibility-only"},
        root=tmp_path,
    )
    final_build = _fake_build(count=100, partition="test")
    for plan in final_build["episode_plans"]:
        plan["protocol_freeze_digest"] = protocol_manifest["content_digest"]
    dataset_path, dataset_manifest, _dataset_reused = (
        storage.write_final_test_dataset(
            final_build,
            protocol_freeze_path=protocol_path,
            root=tmp_path,
        )
    )
    model_path, _model_manifest, _model_reused = storage.write_model_binding(
        {"model_id": "already-frozen-fixture-model", "fit_performed": False},
        protocol_freeze_path=protocol_path,
        dataset_path=dataset_path,
        root=tmp_path,
    )
    authorization_path, _authorization_manifest, _authorization_reused = (
        storage.write_test_authorization(
            {"authorized": True, "purpose": "pytest-ordering-check"},
            protocol_freeze_path=protocol_path,
            dataset_path=dataset_path,
            model_binding_path=model_path,
            root=tmp_path,
        )
    )

    assert dataset_manifest["episode_count"] == 100
    assert dataset_manifest["bindings"]["protocol_freeze_artifact_id"] == (
        protocol_manifest["artifact_id"]
    )
    initial = storage.read_test_ledger(dataset_path)
    assert tuple(entry["event"] for entry in initial.entries) == ("sealed",)

    with pytest.raises(PermissionError, match="authorized order"):
        storage.load_test_labels(
            dataset_path,
            protocol_freeze_path=protocol_path,
            model_binding_path=model_path,
            authorization_path=authorization_path,
        )
    assert storage.read_test_ledger(dataset_path) == initial


def test_test_ledger_accepts_only_the_declared_event_order() -> None:
    dataset, protocol, model, authorization = _ledger_context()
    genesis = storage._ledger_entry(
        dataset_manifest=dataset,
        sequence=0,
        event="sealed",
        protocol_freeze=protocol,
        model_binding=None,
        authorization=None,
        evaluation=None,
        previous_entry_sha256=None,
    )
    observations_opened = storage._ledger_entry(
        dataset_manifest=dataset,
        sequence=1,
        event="observations_opened",
        protocol_freeze=protocol,
        model_binding=model,
        authorization=authorization,
        evaluation=None,
        previous_entry_sha256=genesis["entry_sha256"],
    )
    labels_opened = storage._ledger_entry(
        dataset_manifest=dataset,
        sequence=2,
        event="labels_opened",
        protocol_freeze=protocol,
        model_binding=model,
        authorization=authorization,
        evaluation=None,
        previous_entry_sha256=observations_opened["entry_sha256"],
    )
    valid_payload = b"".join(
        storage._ledger_payload(entry)
        for entry in (genesis, observations_opened, labels_opened)
    )

    decoded = storage._decode_ledger(valid_payload, dataset_manifest=dataset)
    assert tuple(entry["event"] for entry in decoded) == (
        "sealed",
        "observations_opened",
        "labels_opened",
    )

    labels_before_observations = storage._ledger_entry(
        dataset_manifest=dataset,
        sequence=1,
        event="labels_opened",
        protocol_freeze=protocol,
        model_binding=model,
        authorization=authorization,
        evaluation=None,
        previous_entry_sha256=genesis["entry_sha256"],
    )
    invalid_payload = b"".join(
        storage._ledger_payload(entry)
        for entry in (genesis, labels_before_observations)
    )
    with pytest.raises(ValueError, match="identity or hash chain"):
        storage._decode_ledger(invalid_payload, dataset_manifest=dataset)
