from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from app.training.canonical import file_sha256
from app.training.models import (
    DatasetPartition,
)
from app.training.models import (
    TestAccessAuthorization as AccessAuthorization,
)
from app.training.storage import (
    LEDGER_NAME,
    load_partition,
    resign_partition_windows,
    verify_archive,
    write_dataset,
)


def test_two_archives_share_scientific_identity_but_not_execution_metadata(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    first_path, first_manifest, first_execution = write_dataset(
        small_pilot_build, root=tmp_path / "one"
    )
    second_path, second_manifest, second_execution = write_dataset(
        small_pilot_build, root=tmp_path / "two"
    )
    assert first_manifest.scientific_digest == second_manifest.scientific_digest
    assert first_manifest.dataset_id == second_manifest.dataset_id
    assert first_execution.artifact_path != second_execution.artifact_path
    assert first_path.name == second_path.name


def test_labels_are_separate_and_no_normalization_is_fitted(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    path, manifest, _ = write_dataset(small_pilot_build, root=tmp_path)
    loaded = load_partition(path, DatasetPartition.PILOT)
    assert "target" not in loaded["arrays"]
    assert {item["target"] for item in loaded["labels"]["labels"]} == {-1, 1}
    assert manifest.fit_state == "not-fitted"
    assert manifest.scaler_id is None
    assert manifest.theta_id is None
    assert manifest.reference_bank_id is None
    assert manifest.model_id is None


def test_archived_windows_receive_fresh_valid_live_hmac(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    metadata = json.loads((path / "partitions" / "pilot" / "metadata.json").read_text())
    assert "provenance_token" not in metadata["windows"][0][0]
    records = resign_partition_windows(path, DatasetPartition.PILOT)
    assert records
    assert all(record.provenance_token.startswith("feature-window-v1.") for record in records)


def test_development_test_is_sealed_and_fixture_open_is_logged(
    tmp_path: Path,
    small_development_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, manifest, _ = write_dataset(small_development_build, root=tmp_path)
    assert manifest.test_state == "sealed"
    with pytest.raises(PermissionError, match="sealed by default"):
        load_partition(path, DatasetPartition.TEST)

    monkeypatch.setenv("AQSE_TEST_FIXTURE_DATASET_ID", manifest.dataset_id)
    loaded = load_partition(
        path,
        DatasetPartition.TEST,
        authorization=AccessAuthorization(
            actor="pytest",
            reason="software sealing fixture",
            fixture_only=True,
        ),
    )
    assert loaded["arrays"]["features"].shape[-1] == 8
    ledger = [json.loads(line) for line in (path / LEDGER_NAME).read_text().splitlines()]
    assert [entry["event"] for entry in ledger] == ["sealed", "opened"]
    assert ledger[-1]["fixture_only"] is True


def test_real_test_cannot_be_opened_with_fixture_authorization(
    tmp_path: Path,
    small_development_build,
) -> None:
    path, _, _ = write_dataset(small_development_build, root=tmp_path)
    with pytest.raises(PermissionError, match="Milestone 1D.4"):
        load_partition(
            path,
            DatasetPartition.TEST,
            authorization=AccessAuthorization(
                actor="pytest",
                reason="must remain rejected",
                fixture_only=True,
            ),
        )


def test_manifest_and_payloads_are_read_only(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    assert os.stat(path / "manifest.json").st_mode & 0o222 == 0
    assert os.stat(path / "partitions" / "pilot" / "features.npy").st_mode & 0o222 == 0


def test_immutable_identity_cannot_be_overwritten(
    tmp_path: Path,
    small_pilot_build,
) -> None:
    write_dataset(small_pilot_build, root=tmp_path)
    with pytest.raises(FileExistsError, match="immutable dataset already exists"):
        write_dataset(small_pilot_build, root=tmp_path)


@pytest.mark.parametrize("mutation", ["version", "path", "truncated", "object", "labels"])
def test_archive_rejects_unsafe_or_corrupt_content(
    tmp_path: Path,
    small_pilot_build,
    mutation: str,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    manifest_path = path / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    if mutation == "version":
        manifest["schema_version"] = "unsupported"
    elif mutation == "path":
        manifest["files"][0]["relative_path"] = "../escape.npy"
    elif mutation in {"truncated", "object"}:
        record = next(
            item for item in manifest["files"] if item["relative_path"].endswith("features.npy")
        )
        target = path / record["relative_path"]
        target.chmod(0o644)
        if mutation == "truncated":
            target.write_bytes(b"\x93NUMPY")
        else:
            with target.open("wb") as handle:
                np.save(handle, np.asarray([{"unsafe": True}], dtype=object), allow_pickle=True)
        record["byte_count"] = target.stat().st_size
        record["file_sha256"] = file_sha256(target)
    else:
        record = next(
            item for item in manifest["files"] if item["relative_path"].startswith("labels/")
        )
        target = path / record["relative_path"]
        target.chmod(0o644)
        labels = json.loads(target.read_text())
        labels["labels"][0]["target"] *= -1
        target.write_text(json.dumps(labels))
        record["byte_count"] = target.stat().st_size
        record["file_sha256"] = file_sha256(target)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, TypeError)):
        verify_archive(path)


def test_archive_rejects_oversized_files_before_loading(
    tmp_path: Path,
    small_pilot_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _, _ = write_dataset(small_pilot_build, root=tmp_path)
    monkeypatch.setattr("app.training.storage.MAX_SINGLE_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="safe load limit"):
        verify_archive(path)


def test_artifact_root_limit_fails_before_overflow(
    tmp_path: Path,
    small_pilot_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AQSE_ARTIFACT_LIMIT_BYTES", "128")
    with pytest.raises(OSError, match="configured byte limit"):
        write_dataset(small_pilot_build, root=tmp_path)
