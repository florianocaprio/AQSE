from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from app.features.models import FeatureProfile, WindowFeatureRecord
from app.features.provenance import sign_window_record, verify_window_record
from app.training.canonical import (
    array_identity,
    canonical_array,
    canonical_json_bytes,
    file_sha256,
    scientific_digest,
)
from app.training.generation import DatasetBuild, dataset_identity_metadata
from app.training.models import (
    DatasetManifest,
    DatasetPartition,
    ExecutionMetadata,
    FileRecord,
    ObservationInterval,
    PartitionSummary,
    TestAccessAuthorization,
    TestAccessLedgerEntry,
)
from app.training.splits import (
    validate_no_cross_partition_duplicates,
    validate_no_cross_partition_raw_overlaps,
)

DEFAULT_LIMIT_BYTES = 1_073_741_824
MAX_SINGLE_FILE_BYTES = 268_435_456
MANIFEST_NAME = "manifest.json"
EXECUTION_NAME = "execution.json"
LEDGER_NAME = "test-access-ledger.jsonl"
IDENTITY_NAME = "scientific-identity.json"


def artifact_root() -> Path:
    configured = os.environ.get("AQSE_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[3]
    return (repository_root.parent / "AQSE-artifacts").resolve()


def artifact_limit_bytes() -> int:
    raw = os.environ.get("AQSE_ARTIFACT_LIMIT_BYTES", str(DEFAULT_LIMIT_BYTES))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("AQSE_ARTIFACT_LIMIT_BYTES must be an integer") from exc
    if value <= 0:
        raise ValueError("AQSE_ARTIFACT_LIMIT_BYTES must be positive")
    return value


def _safe_child(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {relative_path}")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"artifact path escapes root: {relative_path}")
    return candidate


def _root_usage(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _partition_payload(build: DatasetBuild, partition: DatasetPartition) -> dict[str, Any]:
    episode_ids = {
        assignment.episode_id
        for assignment in build.assignments
        if assignment.partition is partition
    }
    episodes = tuple(
        episode for episode in build.episodes if episode.plan.episode_id in episode_ids
    )
    if not episodes:
        raise ValueError(f"partition {partition.value} contains no episodes")
    arrays: dict[str, NDArray[Any]] = {
        "time_s": np.stack([episode.time_s for episode in episodes]),
        "measured_field_T": np.stack([episode.measured_field_T for episode in episodes]),
        "temperature_K": np.stack([episode.temperature_K for episode in episodes]),
        "saturation_mask": np.stack([episode.saturation_mask for episode in episodes]),
        "features": np.stack([episode.features for episode in episodes]),
        "valid_mask": np.stack([episode.valid_mask for episode in episodes]),
    }
    metadata = {
        "schema_version": "aqse.partition-metadata.v1",
        "partition": partition.value,
        "episode_ids": [episode.plan.episode_id for episode in episodes],
        "lineage_ids": [episode.plan.lineage_id for episode in episodes],
        "profile": episodes[0].feature_response.profile.model_dump(mode="json"),
        "plans": [episode.plan.model_dump(mode="json") for episode in episodes],
        "coverage": [episode.coverage.model_dump(mode="json") for episode in episodes],
        "windows": [
            [
                record.model_dump(mode="json", exclude={"provenance_token"})
                for record in episode.feature_response.windows
            ]
            for episode in episodes
        ],
    }
    labels = {
        "schema_version": "aqse.partition-labels.v1",
        "partition": partition.value,
        "labels": [episode.label.model_dump(mode="json") for episode in episodes],
    }
    return {"episodes": episodes, "arrays": arrays, "metadata": metadata, "labels": labels}


def write_dataset(
    build: DatasetBuild, *, root: Path | None = None
) -> tuple[Path, DatasetManifest, ExecutionMetadata]:
    """Atomically archive a build without overwriting an existing scientific identity."""

    started = time.perf_counter()
    tracemalloc.start()
    resolved_root = (root or artifact_root()).resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    limit = artifact_limit_bytes()
    if _root_usage(resolved_root) >= limit:
        raise OSError("AQSE artifact root has reached its configured byte limit")

    partition_order = (
        (DatasetPartition.PILOT,)
        if build.kind == "pilot"
        else (
            DatasetPartition.TRAIN,
            DatasetPartition.VALIDATION,
            DatasetPartition.TEST,
        )
    )
    payloads = {partition: _partition_payload(build, partition) for partition in partition_order}
    identity_metadata = dataset_identity_metadata(build)
    identity_arrays = {
        f"{partition.value}/{name}": array
        for partition, payload in payloads.items()
        for name, array in payload["arrays"].items()
    }
    identity_digest = scientific_digest(identity_metadata, identity_arrays)
    dataset_id = f"aqse-{build.kind}-{identity_digest[:16]}"
    final_path = _safe_child(resolved_root, dataset_id)
    if final_path.exists():
        raise FileExistsError(f"immutable dataset already exists: {final_path}")
    estimated_bytes = _estimated_archive_bytes(payloads, identity_metadata)
    if _root_usage(resolved_root) + estimated_bytes > limit:
        raise OSError("AQSE artifact write would exceed the configured byte limit")
    staging = _safe_child(resolved_root, f".{dataset_id}.tmp-{uuid4().hex}")
    staging.mkdir()
    file_records: list[FileRecord] = []
    try:
        identity_path = staging / IDENTITY_NAME
        _write_json(identity_path, identity_metadata)
        file_records.append(_json_file_record(identity_path, staging))
        for partition, payload in payloads.items():
            partition_dir = staging / "partitions" / partition.value
            partition_dir.mkdir(parents=True)
            for name, source in payload["arrays"].items():
                array = canonical_array(source)
                target = partition_dir / f"{name}.npy"
                with target.open("wb") as handle:
                    np.save(handle, array, allow_pickle=False)
                identity = array_identity(array)
                file_records.append(
                    FileRecord(
                        relative_path=target.relative_to(staging).as_posix(),
                        byte_count=target.stat().st_size,
                        file_sha256=file_sha256(target),
                        content_sha256=identity["content_sha256"],
                        dtype=identity["dtype"],
                        shape=tuple(identity["shape"]),
                        order="C",
                        endianness=identity["endianness"],
                    )
                )
            metadata_path = partition_dir / "metadata.json"
            _write_json(metadata_path, payload["metadata"])
            file_records.append(_json_file_record(metadata_path, staging))
            labels_path = staging / "labels" / f"{partition.value}.json"
            _write_json(labels_path, payload["labels"])
            file_records.append(_json_file_record(labels_path, staging))
            if _root_usage(resolved_root) > limit:
                raise OSError("AQSE artifact write would exceed the configured byte limit")

        validate_no_cross_partition_duplicates(
            build.assignments,
            {episode.plan.episode_id: episode.observation_digest for episode in build.episodes},
        )
        validate_no_cross_partition_raw_overlaps(
            build.assignments,
            tuple(
                ObservationInterval(
                    episode_id=episode.plan.episode_id,
                    lineage_id=episode.plan.lineage_id,
                    sensor_id="M1",
                    start_index=0,
                    end_index=len(episode.time_s),
                )
                for episode in build.episodes
            ),
        )
        summaries = tuple(
            _partition_summary(
                partition, payloads[partition], hide_quality=(partition is DatasetPartition.TEST)
            )
            for partition in partition_order
        )
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            kind=build.kind,  # type: ignore[arg-type]
            scientific_digest=identity_digest,
            feature_profile_fingerprint=build.feature_profile_fingerprint,
            split_policy=("pilot-only" if build.kind == "pilot" else "stratified-lineage-60-20-20"),
            partition_summaries=summaries,
            files=tuple(sorted(file_records, key=lambda item: item.relative_path)),
            test_state="not-applicable" if build.kind == "pilot" else "sealed",
        )
        manifest_path = staging / MANIFEST_NAME
        _write_json(manifest_path, manifest.model_dump(mode="json"))
        if build.kind == "development":
            _initialize_ledger(staging, dataset_id)

        _, peak_memory = tracemalloc.get_traced_memory()
        duration_ms = (time.perf_counter() - started) * 1_000.0
        execution = ExecutionMetadata(
            dataset_id=dataset_id,
            artifact_path=str(final_path),
            written_at_utc=_utc_now(),
            duration_ms=duration_ms,
            peak_memory_bytes=peak_memory,
            total_bytes=_root_usage(staging),
        )
        _write_json(staging / EXECUTION_NAME, execution.model_dump(mode="json"))
        if _root_usage(resolved_root) > limit:
            raise OSError("AQSE artifact write would exceed the configured byte limit")

        for path in staging.rglob("*"):
            if path.is_file() and path.name not in {LEDGER_NAME, EXECUTION_NAME}:
                path.chmod(0o444)
        os.replace(staging, final_path)
        verify_archive(final_path)
        return final_path, manifest, execution
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        tracemalloc.stop()


def _json_file_record(path: Path, staging: Path) -> FileRecord:
    content = path.read_bytes().rstrip(b"\n")
    return FileRecord(
        relative_path=path.relative_to(staging).as_posix(),
        byte_count=path.stat().st_size,
        file_sha256=file_sha256(path),
        content_sha256=hashlib.sha256(content).hexdigest(),
    )


def _estimated_archive_bytes(
    payloads: dict[DatasetPartition, dict[str, Any]],
    identity_metadata: dict[str, Any],
) -> int:
    numeric_bytes = sum(
        canonical_array(array).nbytes + 4_096
        for payload in payloads.values()
        for array in payload["arrays"].values()
    )
    json_bytes = len(canonical_json_bytes(identity_metadata))
    for payload in payloads.values():
        json_bytes += len(canonical_json_bytes(payload["metadata"]))
        json_bytes += len(canonical_json_bytes(payload["labels"]))
    # Reserve one MiB for manifests, ledger, NPY headers and execution metadata.
    return numeric_bytes + json_bytes + 1_048_576


def _partition_summary(
    partition: DatasetPartition,
    payload: dict[str, Any],
    *,
    hide_quality: bool,
) -> PartitionSummary:
    episodes = payload["episodes"]
    proposed = sum(episode.coverage.proposed_windows for episode in episodes)
    accepted = sum(episode.coverage.accepted_windows for episode in episodes)
    return PartitionSummary(
        partition=partition,
        episode_count=len(episodes),
        lineage_count=len({episode.plan.lineage_id for episode in episodes}),
        proposed_window_count=proposed,
        accepted_window_count=None if hide_quality else accepted,
        rejected_window_count=None if hide_quality else proposed - accepted,
    )


def verify_archive(dataset_path: Path) -> DatasetManifest:
    path = dataset_path.resolve()
    raw_manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate(raw_manifest)
    if manifest.dataset_id != path.name:
        raise ValueError("dataset directory and manifest identifiers do not match")
    identity_arrays: dict[str, NDArray[Any]] = {}
    for record in manifest.files:
        target = _safe_child(path, record.relative_path)
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"artifact file is missing or unsafe: {record.relative_path}")
        if target.stat().st_size != record.byte_count:
            raise ValueError(f"artifact file size mismatch: {record.relative_path}")
        if target.stat().st_size > MAX_SINGLE_FILE_BYTES:
            raise ValueError(f"artifact file exceeds the safe load limit: {record.relative_path}")
        if file_sha256(target) != record.file_sha256:
            raise ValueError(f"artifact file digest mismatch: {record.relative_path}")
        if target.suffix == ".npy":
            array = _safe_load_array(target)
            identity = array_identity(array)
            if identity["content_sha256"] != record.content_sha256:
                raise ValueError(f"numeric content digest mismatch: {record.relative_path}")
            if identity["dtype"] != record.dtype or tuple(identity["shape"]) != record.shape:
                raise ValueError(f"numeric schema mismatch: {record.relative_path}")
            relative = Path(record.relative_path)
            if len(relative.parts) == 3 and relative.parts[0] == "partitions":
                identity_arrays[f"{relative.parts[1]}/{relative.stem}"] = array
    identity_metadata = json.loads((path / IDENTITY_NAME).read_text(encoding="utf-8"))
    if scientific_digest(identity_metadata, identity_arrays) != manifest.scientific_digest:
        raise ValueError("scientific identity digest mismatch")
    labels_by_episode: dict[str, dict[str, Any]] = {}
    for summary in manifest.partition_summaries:
        labels_path = _safe_child(path, f"labels/{summary.partition.value}.json")
        label_payload = json.loads(labels_path.read_text(encoding="utf-8"))
        for label in label_payload["labels"]:
            labels_by_episode[label["episode_id"]] = label
    ordered_labels = [labels_by_episode[plan["episode_id"]] for plan in identity_metadata["plans"]]
    label_digest = hashlib.sha256(canonical_json_bytes(ordered_labels)).hexdigest()
    if label_digest != identity_metadata["label_artifact_sha256"]:
        raise ValueError("label artifact digest mismatch")
    return manifest


def _safe_load_array(path: Path) -> NDArray[Any]:
    if path.stat().st_size > MAX_SINGLE_FILE_BYTES:
        raise ValueError("numeric artifact exceeds the safe load limit")
    try:
        with path.open("rb") as handle:
            value = np.load(handle, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"invalid or truncated numeric artifact: {path.name}") from exc
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ValueError("numeric artifacts must be non-object NumPy arrays")
    return value


def load_partition(
    dataset_path: Path,
    partition: DatasetPartition,
    *,
    authorization: TestAccessAuthorization | None = None,
) -> dict[str, Any]:
    path = dataset_path.resolve()
    manifest = verify_archive(path)
    if partition is DatasetPartition.TEST:
        _authorize_test_open(path, manifest, authorization)
    partition_dir = _safe_child(path, f"partitions/{partition.value}")
    arrays = {item.stem: _safe_load_array(item) for item in sorted(partition_dir.glob("*.npy"))}
    metadata = json.loads((partition_dir / "metadata.json").read_text(encoding="utf-8"))
    labels = json.loads(
        _safe_child(path, f"labels/{partition.value}.json").read_text(encoding="utf-8")
    )
    return {"arrays": arrays, "metadata": metadata, "labels": labels}


def _authorize_test_open(
    path: Path,
    manifest: DatasetManifest,
    authorization: TestAccessAuthorization | None,
) -> None:
    if authorization is None:
        raise PermissionError("the AQSE test partition is sealed by default")
    fixture_dataset = os.environ.get("AQSE_TEST_FIXTURE_DATASET_ID") == manifest.dataset_id
    real_authorized = os.environ.get("AQSE_ALLOW_TEST_OPEN") == "1"
    if fixture_dataset:
        if not authorization.fixture_only:
            raise PermissionError("test fixtures require fixture_only authorization")
    elif authorization.fixture_only or not real_authorized:
        raise PermissionError(
            "opening the real test partition requires explicit Milestone 1D.4 authorization"
        )
    _append_ledger(path, manifest.dataset_id, authorization)


def resign_partition_windows(
    dataset_path: Path, partition: DatasetPartition
) -> tuple[WindowFeatureRecord, ...]:
    """Reload archived safe records and issue fresh process-local HMAC tokens."""

    loaded = load_partition(dataset_path, partition)
    profile = FeatureProfile.model_validate(loaded["metadata"]["profile"])
    records: list[WindowFeatureRecord] = []
    for episode_windows in loaded["metadata"]["windows"]:
        for payload in episode_windows:
            unsigned = WindowFeatureRecord.model_validate({**payload, "provenance_token": ""})
            signed = unsigned.model_copy(
                update={"provenance_token": sign_window_record(profile, unsigned)}
            )
            if not verify_window_record(profile, signed):
                raise ValueError("failed to re-sign an archived feature window")
            records.append(signed)
    return tuple(records)


def _initialize_ledger(path: Path, dataset_id: str) -> None:
    payload = {
        "schema_version": "aqse.test-access-ledger.v1",
        "sequence": 0,
        "event": "sealed",
        "dataset_id": dataset_id,
        "occurred_at_utc": _utc_now(),
        "actor": "AQSE 1D.1 archiver",
        "reason": "test partition sealed at dataset creation",
        "fixture_only": False,
        "previous_entry_sha256": None,
    }
    payload["entry_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    entry = TestAccessLedgerEntry.model_validate(payload)
    (path / LEDGER_NAME).write_bytes(canonical_json_bytes(entry.model_dump(mode="json")) + b"\n")


def _append_ledger(
    path: Path,
    dataset_id: str,
    authorization: TestAccessAuthorization,
) -> None:
    ledger_path = path / LEDGER_NAME
    entries = [
        TestAccessLedgerEntry.model_validate_json(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    _verify_ledger(entries, dataset_id)
    previous = entries[-1].entry_sha256
    payload = {
        "schema_version": "aqse.test-access-ledger.v1",
        "sequence": len(entries),
        "event": "opened",
        "dataset_id": dataset_id,
        "occurred_at_utc": _utc_now(),
        "actor": authorization.actor,
        "reason": authorization.reason,
        "fixture_only": authorization.fixture_only,
        "previous_entry_sha256": previous,
    }
    payload["entry_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    entry = TestAccessLedgerEntry.model_validate(payload)
    temporary = ledger_path.with_name(f".{LEDGER_NAME}.{uuid4().hex}.tmp")
    temporary.write_bytes(
        b"".join(
            canonical_json_bytes(item.model_dump(mode="json")) + b"\n" for item in (*entries, entry)
        )
    )
    os.replace(temporary, ledger_path)


def _verify_ledger(entries: list[TestAccessLedgerEntry], dataset_id: str) -> None:
    if not entries or entries[0].event != "sealed":
        raise ValueError("test access ledger has no sealing genesis entry")
    previous: str | None = None
    for sequence, entry in enumerate(entries):
        if entry.sequence != sequence or entry.dataset_id != dataset_id:
            raise ValueError("test access ledger sequence or dataset identity is invalid")
        if entry.previous_entry_sha256 != previous:
            raise ValueError("test access ledger chain is invalid")
        payload = entry.model_dump(mode="json", exclude={"entry_sha256"})
        expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if entry.entry_sha256 != expected:
            raise ValueError("test access ledger entry digest is invalid")
        previous = entry.entry_sha256


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
