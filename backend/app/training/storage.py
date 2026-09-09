from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from app.features.models import (
    FeatureExtractionRequest,
    FeatureProfile,
    FeatureWindowConfiguration,
    MeasuredVectorSeries,
    WindowFeatureRecord,
)
from app.features.provenance import sign_window_record, verify_window_record
from app.features.windowed import extract_windowed_magnetometer_features
from app.training.canonical import (
    array_identity,
    canonical_array,
    canonical_json_bytes,
    file_sha256,
    scientific_digest,
)
from app.training.generation import DatasetBuild, dataset_identity_metadata
from app.training.models import (
    ArchivedWindowRecord,
    DatasetManifest,
    DatasetPartition,
    EpisodeLabel,
    EpisodePlan,
    ExecutionMetadata,
    FileRecord,
    LegacyDatasetManifest,
    ObservableProvenance,
    ObservationInterval,
    PartitionMetadataV2,
    PartitionSummary,
    SoftwareProvenance,
    SplitAssignment,
    TestAccessAuthorization,
    TestAccessLedgerEntry,
)
from app.training.software_provenance import (
    capture_software_provenance,
    validate_software_provenance,
)
from app.training.splits import (
    validate_no_cross_partition_duplicates,
    validate_no_cross_partition_raw_overlaps,
)

DEFAULT_LIMIT_BYTES = 1_073_741_824
MAX_SINGLE_FILE_BYTES = 268_435_456
MAX_MANIFEST_BYTES = 4_194_304
MAX_JSON_BYTES = 33_554_432
MAX_ARRAY_ELEMENTS = 50_000_000
FEATURE_RTOL = 1.0e-10
FEATURE_ATOL = 1.0e-12
MANIFEST_NAME = "manifest.json"
EXECUTION_NAME = "execution.json"
LEDGER_NAME = "test-access-ledger.jsonl"
LEDGER_LOCK_NAME = ".test-access-ledger.lock"
IDENTITY_NAME = "scientific-identity.json"
SOFTWARE_PROVENANCE_NAME = "provenance/software.json"
ARRAY_NAMES = (
    "features",
    "measured_field_T",
    "saturation_mask",
    "temperature_K",
    "time_s",
    "valid_mask",
)


@dataclass(frozen=True)
class ObservablePartition:
    """Truth-free DTO intended for later encoding consumers."""

    dataset_id: str
    partition: DatasetPartition
    episode_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    profile: FeatureProfile
    provenance: ObservableProvenance
    time_s: NDArray[np.float64]
    measured_field_T: NDArray[np.float64]
    temperature_K: NDArray[np.float64]
    saturation_mask: NDArray[np.bool_]
    features: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    windows: tuple[tuple[WindowFeatureRecord, ...], ...]


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
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {relative_path}")
    candidate = root.joinpath(relative)
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
        raise ValueError(f"artifact path escapes root: {relative_path}")
    return candidate


def _validated_file(root: Path, relative_path: str) -> Path:
    target = _safe_child(root, relative_path)
    cursor = root
    for part in Path(relative_path).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"artifact path contains a symlink: {relative_path}")
    if not target.is_file():
        raise ValueError(f"artifact file is missing or unsafe: {relative_path}")
    return target


def _root_usage(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _read_json(path: Path, *, limit: int | None = None) -> Any:
    if limit is None:
        limit = MAX_JSON_BYTES
    if path.stat().st_size > limit:
        raise ValueError(f"JSON artifact exceeds the bounded load limit: {path.name}")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path.name}") from exc


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
    profile = episodes[0].feature_response.profile.model_dump(mode="json")
    profile_fingerprint = hashlib.sha256(canonical_json_bytes(profile)).hexdigest()
    metadata = PartitionMetadataV2(
        partition=partition,
        episode_ids=tuple(episode.plan.episode_id for episode in episodes),
        lineage_ids=tuple(episode.plan.lineage_id for episode in episodes),
        profile=profile,
        observable_provenance=ObservableProvenance(feature_profile_fingerprint=profile_fingerprint),
        raw_sources=tuple(episode.raw_source for episode in episodes),
        coverage=tuple(episode.coverage for episode in episodes),
        windows=tuple(
            tuple(
                ArchivedWindowRecord.model_validate(
                    record.model_dump(mode="json", exclude={"features", "provenance_token"})
                )
                for record in episode.feature_response.windows
            )
            for episode in episodes
        ),
    ).model_dump(mode="json")
    labels = {
        "schema_version": "aqse.partition-labels.v2",
        "partition": partition.value,
        "labels": [episode.label.model_dump(mode="json") for episode in episodes],
    }
    generation = {
        "schema_version": "aqse.generation-channel.v1",
        "partition": partition.value,
        "plans": [episode.plan.model_dump(mode="json") for episode in episodes],
        "assignments": [
            assignment.model_dump(mode="json")
            for assignment in build.assignments
            if assignment.partition is partition
        ],
        "observation_bindings": [
            {
                "episode_id": episode.plan.episode_id,
                "lineage_id": episode.plan.lineage_id,
                "observation_content_digest": episode.observation_content_digest,
                "observation_binding_digest": episode.observation_binding_digest,
                "raw_source": episode.raw_source.model_dump(mode="json"),
            }
            for episode in episodes
        ],
    }
    return {
        "episodes": episodes,
        "arrays": arrays,
        "metadata": metadata,
        "labels": labels,
        "generation": generation,
    }


def _validate_build_references(build: DatasetBuild) -> None:
    episode_ids = [episode.plan.episode_id for episode in build.episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("dataset build contains duplicate episode identifiers")
    assignments = {item.episode_id: item for item in build.assignments}
    if len(assignments) != len(build.assignments) or set(assignments) != set(episode_ids):
        raise ValueError("every episode must have exactly one split assignment")
    lineage_partitions: dict[str, DatasetPartition] = {}
    for episode in build.episodes:
        if episode.plan.episode_id != episode.label.episode_id:
            raise ValueError("plan and label episode identifiers must match")
        if episode.plan.lineage_id != episode.label.lineage_id:
            raise ValueError("plan and label lineage identifiers must match")
        assignment = assignments[episode.plan.episode_id]
        if assignment.lineage_id != episode.plan.lineage_id:
            raise ValueError("assignment lineage does not match plan and label")
        previous = lineage_partitions.setdefault(assignment.lineage_id, assignment.partition)
        if previous is not assignment.partition:
            raise ValueError("one lineage cannot cross dataset partitions")
        if episode.raw_source.end_index > len(episode.time_s):
            raise ValueError("raw source interval exceeds the observable episode")
    validate_no_cross_partition_duplicates(
        build.assignments,
        {episode.plan.episode_id: episode.observation_content_digest for episode in build.episodes},
    )
    validate_no_cross_partition_raw_overlaps(
        build.assignments,
        tuple(
            ObservationInterval(
                episode_id=episode.plan.episode_id,
                lineage_id=episode.plan.lineage_id,
                sensor_id=episode.raw_source.sensor_id,
                start_index=episode.raw_source.start_index,
                end_index=episode.raw_source.end_index,
                source_id=episode.raw_source.source_id,
                relation=episode.raw_source.relation,
            )
            for episode in build.episodes
        ),
    )


def write_dataset(
    build: DatasetBuild, *, root: Path | None = None
) -> tuple[Path, DatasetManifest, ExecutionMetadata]:
    """Validate a complete v2 candidate before atomically publishing it."""

    started = time.perf_counter()
    tracemalloc.start()
    resolved_root = (root or artifact_root()).resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    limit = artifact_limit_bytes()
    staging: Path | None = None
    try:
        if _root_usage(resolved_root) >= limit:
            raise OSError("AQSE artifact root has reached its configured byte limit")
        _validate_build_references(build)
        partition_order = (
            (DatasetPartition.PILOT,)
            if build.kind == "pilot"
            else (
                DatasetPartition.TRAIN,
                DatasetPartition.VALIDATION,
                DatasetPartition.TEST,
            )
        )
        payloads = {
            partition: _partition_payload(build, partition) for partition in partition_order
        }
        software = capture_software_provenance()
        software_payload = software.model_dump(mode="json")
        validate_software_provenance(software, verify_effective_sources=True)
        numeric_metadata = dataset_identity_metadata(build)
        numeric_digest = hashlib.sha256(canonical_json_bytes(numeric_metadata)).hexdigest()
        scientific_identity = {
            "schema_version": "aqse.dataset-scientific-identity.v2",
            "numeric_content_digest": numeric_digest,
            "feature_profile_fingerprint": build.feature_profile_fingerprint,
            "partition_metadata_sha256": {
                partition.value: _canonical_digest(payloads[partition]["metadata"])
                for partition in partition_order
            },
            "partition_label_sha256": {
                partition.value: _canonical_digest(payloads[partition]["labels"])
                for partition in partition_order
            },
            "generation_channel_sha256": {
                partition.value: _canonical_digest(payloads[partition]["generation"])
                for partition in partition_order
            },
            "array_content_sha256": {
                f"partitions/{partition.value}/{name}.npy": array_identity(array)["content_sha256"]
                for partition in partition_order
                for name, array in payloads[partition]["arrays"].items()
            },
            "software_provenance_sha256": _canonical_digest(software_payload),
            "label_policy": "aqse.white-noise-regime.v1",
            "expected_encoding": "aqse.tqk8.encoding.phase-direct.v1",
            "fit_state": "not-fitted",
        }
        identity_digest = _canonical_digest(scientific_identity)
        dataset_id = f"aqse-{build.kind}-{identity_digest[:16]}"
        final_path = _safe_child(resolved_root, dataset_id)
        if final_path.exists():
            raise FileExistsError(f"immutable dataset already exists: {final_path}")
        estimated_bytes = _estimated_archive_bytes(payloads, scientific_identity)
        if _root_usage(resolved_root) + estimated_bytes > limit:
            raise OSError("AQSE artifact write would exceed the configured byte limit")
        staging = _safe_child(resolved_root, f".{dataset_id}.tmp-{uuid4().hex}")
        staging.mkdir()
        file_records: list[FileRecord] = []

        for relative, payload in (
            (IDENTITY_NAME, scientific_identity),
            (SOFTWARE_PROVENANCE_NAME, software_payload),
        ):
            target = staging / relative
            _write_json(target, payload)
            file_records.append(_json_file_record(target, staging))
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
            for relative, value in (
                (f"partitions/{partition.value}/metadata.json", payload["metadata"]),
                (f"labels/{partition.value}.json", payload["labels"]),
                (f"generation/{partition.value}.json", payload["generation"]),
            ):
                target = staging / relative
                _write_json(target, value)
                file_records.append(_json_file_record(target, staging))
            if _root_usage(resolved_root) > limit:
                raise OSError("AQSE artifact write would exceed the configured byte limit")

        summaries = tuple(
            _partition_summary(
                partition, payloads[partition], hide_quality=partition is DatasetPartition.TEST
            )
            for partition in partition_order
        )
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            kind=build.kind,  # type: ignore[arg-type]
            scientific_digest=identity_digest,
            numeric_content_digest=numeric_digest,
            feature_profile_fingerprint=build.feature_profile_fingerprint,
            software_provenance_sha256=_canonical_digest(software_payload),
            split_policy=("pilot-only" if build.kind == "pilot" else "stratified-lineage-60-20-20"),
            partition_summaries=summaries,
            files=tuple(sorted(file_records, key=lambda item: item.relative_path)),
            test_state="not-applicable" if build.kind == "pilot" else "sealed",
        )
        _write_json(staging / MANIFEST_NAME, manifest.model_dump(mode="json"))
        _validate_archive_candidate(staging, manifest)

        if build.kind == "development":
            _initialize_ledger(staging, manifest)
        _, peak_memory = tracemalloc.get_traced_memory()
        execution = ExecutionMetadata(
            dataset_id=dataset_id,
            artifact_path=str(final_path),
            written_at_utc=_utc_now(),
            duration_ms=(time.perf_counter() - started) * 1_000.0,
            peak_memory_bytes=peak_memory,
            total_bytes=_root_usage(staging),
        )
        _write_json(staging / EXECUTION_NAME, execution.model_dump(mode="json"))
        if _root_usage(resolved_root) > limit:
            raise OSError("AQSE artifact write would exceed the configured byte limit")
        for item in staging.rglob("*"):
            if item.is_file() and item.name not in {LEDGER_NAME, EXECUTION_NAME}:
                item.chmod(0o444)
        os.replace(staging, final_path)
        staging = None
        verify_archive_opaque(final_path)
        return final_path, manifest, execution
    except Exception:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        tracemalloc.stop()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _observation_content_digest(
    time_s: NDArray[Any],
    measured_field_t: NDArray[Any],
    temperature_k: NDArray[Any],
    saturation_mask: NDArray[Any],
) -> str:
    return scientific_digest(
        {
            "schema_version": "aqse.observation-content.v1",
            "sensor_type": "magnetometer_vector3",
            "arrays": {
                "time_s": {"unit": "s"},
                "measured_field_T": {"unit": "T"},
                "temperature_K": {"unit": "K"},
                "saturation_mask": {"unit": "boolean"},
            },
        },
        {
            "time_s": time_s,
            "measured_field_T": measured_field_t,
            "temperature_K": temperature_k,
            "saturation_mask": saturation_mask,
        },
    )


def _json_file_record(path: Path, staging: Path) -> FileRecord:
    content = path.read_bytes().rstrip(b"\n")
    return FileRecord(
        relative_path=path.relative_to(staging).as_posix(),
        byte_count=path.stat().st_size,
        file_sha256=file_sha256(path),
        content_sha256=hashlib.sha256(content).hexdigest(),
    )


def _estimated_archive_bytes(
    payloads: dict[DatasetPartition, dict[str, Any]], identity_metadata: dict[str, Any]
) -> int:
    numeric_bytes = sum(
        canonical_array(array).nbytes + 4_096
        for payload in payloads.values()
        for array in payload["arrays"].values()
    )
    json_bytes = len(canonical_json_bytes(identity_metadata))
    for payload in payloads.values():
        json_bytes += sum(
            len(canonical_json_bytes(payload[name]))
            for name in ("metadata", "labels", "generation")
        )
    return numeric_bytes + json_bytes + 1_048_576


def _partition_summary(
    partition: DatasetPartition, payload: dict[str, Any], *, hide_quality: bool
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


def _manifest_for_path(path: Path) -> DatasetManifest | LegacyDatasetManifest:
    target = _validated_file(path, MANIFEST_NAME)
    raw = _read_json(target, limit=MAX_MANIFEST_BYTES)
    if not isinstance(raw, dict):
        raise ValueError("dataset manifest must be a JSON object")
    if raw.get("schema_version") == "aqse.dataset-manifest.v1":
        return LegacyDatasetManifest.model_validate(raw)
    return DatasetManifest.model_validate(raw)


def verify_archive_opaque(dataset_path: Path) -> DatasetManifest | LegacyDatasetManifest:
    """Verify paths, sizes and streaming SHA-256 without NumPy or label decoding."""

    path = dataset_path.resolve()
    manifest = _manifest_for_path(path)
    if manifest.dataset_id != path.name:
        raise ValueError("dataset directory and manifest identifiers do not match")
    records = {record.relative_path: record for record in manifest.files}
    if len(records) != len(manifest.files):
        raise ValueError("dataset manifest contains duplicate file paths")
    for record in manifest.files:
        target = _validated_file(path, record.relative_path)
        size = target.stat().st_size
        if size != record.byte_count:
            raise ValueError(f"artifact file size mismatch: {record.relative_path}")
        if size > MAX_SINGLE_FILE_BYTES:
            raise ValueError(f"artifact file exceeds the safe load limit: {record.relative_path}")
        if file_sha256(target) != record.file_sha256:
            raise ValueError(f"artifact file digest mismatch: {record.relative_path}")
    if isinstance(manifest, DatasetManifest):
        actual = {
            item.relative_to(path).as_posix()
            for item in path.rglob("*")
            if (item.is_file() or item.is_symlink())
            and item.name not in {MANIFEST_NAME, EXECUTION_NAME, LEDGER_NAME, LEDGER_LOCK_NAME}
        }
        expected = set(records)
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        if unexpected:
            raise ValueError(f"unexpected scientific file: {unexpected[0]}")
        if missing:
            raise ValueError(f"missing scientific file: {missing[0]}")
        identity = _read_json(_validated_file(path, IDENTITY_NAME))
        if _canonical_digest(identity) != manifest.scientific_digest:
            raise ValueError("scientific identity digest mismatch")
        if identity.get("numeric_content_digest") != manifest.numeric_content_digest:
            raise ValueError("numeric content identity mismatch")
        if identity.get("software_provenance_sha256") != manifest.software_provenance_sha256:
            raise ValueError("software provenance identity mismatch")
    return manifest


def verify_archive(dataset_path: Path) -> DatasetManifest | LegacyDatasetManifest:
    """Compatibility name for the intentionally opaque archive verification."""

    return verify_archive_opaque(dataset_path)


def _record_map(manifest: DatasetManifest) -> dict[str, FileRecord]:
    return {record.relative_path: record for record in manifest.files}


def _inspect_npy(path: Path) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    try:
        with path.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError("unsupported NPY header version")
            payload_offset = handle.tell()
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"invalid or truncated numeric artifact: {path.name}") from exc
    if (
        not isinstance(shape, tuple)
        or not shape
        or any(not isinstance(item, int) or item < 0 for item in shape)
    ):
        raise ValueError("numeric artifact contains an invalid shape")
    element_count = math.prod(shape)
    if element_count > MAX_ARRAY_ELEMENTS:
        raise ValueError("numeric artifact shape exceeds the element limit")
    if dtype.hasobject or dtype.fields is not None or dtype.kind in "OSUV":
        raise ValueError("numeric artifacts must use an approved primitive dtype")
    if payload_offset + element_count * dtype.itemsize != path.stat().st_size:
        raise ValueError("numeric artifact header and physical payload size disagree")
    return shape, fortran, dtype, payload_offset


def _safe_load_array(
    path: Path,
    record: FileRecord | None = None,
    array_name: str | None = None,
) -> NDArray[Any]:
    if path.stat().st_size > MAX_SINGLE_FILE_BYTES:
        raise ValueError("numeric artifact exceeds the safe load limit")
    shape, fortran, dtype, _ = _inspect_npy(path)
    if fortran:
        raise ValueError("numeric artifacts must use C order")
    allowed = (
        np.dtype("bool") if array_name in {"saturation_mask", "valid_mask"} else np.dtype("<f8")
    )
    if array_name is not None and dtype != allowed:
        raise ValueError(f"numeric artifact {array_name} has an unsupported dtype")
    if record is not None and (tuple(shape) != record.shape or dtype.str != record.dtype):
        raise ValueError(f"numeric schema mismatch: {record.relative_path}")
    try:
        with path.open("rb") as handle:
            value = np.load(handle, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"invalid or truncated numeric artifact: {path.name}") from exc
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ValueError("numeric artifacts must be non-object NumPy arrays")
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ValueError(f"numeric artifact contains non-finite values: {path.name}")
    if record is not None and array_identity(value)["content_sha256"] != record.content_sha256:
        raise ValueError(f"numeric content digest mismatch: {record.relative_path}")
    value.setflags(write=False)
    return value


def _authorize_if_test(
    path: Path,
    manifest: DatasetManifest,
    partition: DatasetPartition,
    authorization: TestAccessAuthorization | None,
) -> None:
    if partition is DatasetPartition.TEST:
        _authorize_test_open(path, manifest, authorization)


def load_observations(
    dataset_path: Path,
    partition: DatasetPartition,
    *,
    authorization: TestAccessAuthorization | None = None,
) -> ObservablePartition:
    path = dataset_path.resolve()
    manifest = verify_archive_opaque(path)
    if isinstance(manifest, LegacyDatasetManifest):
        raise ValueError(
            "legacy v1 archives support opaque verification only; migration/rebuild requires "
            "separate authorization"
        )
    _authorize_if_test(path, manifest, partition, authorization)
    return _load_observations_semantic(path, manifest, partition)


def load_partition(
    dataset_path: Path,
    partition: DatasetPartition,
    *,
    authorization: TestAccessAuthorization | None = None,
) -> ObservablePartition:
    """Compatibility alias; labels and generation plans are intentionally excluded."""

    return load_observations(dataset_path, partition, authorization=authorization)


def _load_observations_semantic(
    path: Path, manifest: DatasetManifest, partition: DatasetPartition
) -> ObservablePartition:
    records = _record_map(manifest)
    metadata_relative = f"partitions/{partition.value}/metadata.json"
    if metadata_relative not in records:
        raise ValueError(f"partition is not present in the manifest: {partition.value}")
    metadata = PartitionMetadataV2.model_validate(
        _read_json(_validated_file(path, metadata_relative))
    )
    if metadata.partition is not partition:
        raise ValueError("partition metadata identity mismatch")
    identity = _read_json(_validated_file(path, IDENTITY_NAME))
    if identity["partition_metadata_sha256"].get(partition.value) != _canonical_digest(
        metadata.model_dump(mode="json")
    ):
        raise ValueError("partition window ledger is not bound to scientific identity")
    profile = FeatureProfile.model_validate(metadata.profile)
    fingerprint = _canonical_digest(profile.model_dump(mode="json"))
    if fingerprint != manifest.feature_profile_fingerprint:
        raise ValueError("feature profile fingerprint differs from the manifest")
    if fingerprint != metadata.observable_provenance.feature_profile_fingerprint:
        raise ValueError("observable provenance contains the wrong profile fingerprint")
    software = SoftwareProvenance.model_validate(
        _read_json(_validated_file(path, SOFTWARE_PROVENANCE_NAME))
    )
    validate_software_provenance(software)
    if _canonical_digest(software.model_dump(mode="json")) != manifest.software_provenance_sha256:
        raise ValueError("software provenance content differs from the manifest")

    arrays: dict[str, NDArray[Any]] = {}
    partition_prefix = f"partitions/{partition.value}/"
    declared_names = {
        Path(relative).stem
        for relative in records
        if relative.startswith(partition_prefix) and relative.endswith(".npy")
    }
    if declared_names != set(ARRAY_NAMES):
        raise ValueError("partition numeric allowlist does not match the v2 contract")
    for name in ARRAY_NAMES:
        relative = f"partitions/{partition.value}/{name}.npy"
        record = records[relative]
        arrays[name] = _safe_load_array(_validated_file(path, relative), record, name)
        if identity["array_content_sha256"].get(relative) != record.content_sha256:
            raise ValueError("numeric payload is not bound to scientific identity")
    _validate_array_shapes(arrays, metadata)
    windows = _recompute_and_validate_windows(arrays, metadata, profile)
    return ObservablePartition(
        dataset_id=manifest.dataset_id,
        partition=partition,
        episode_ids=metadata.episode_ids,
        lineage_ids=metadata.lineage_ids,
        profile=profile,
        provenance=metadata.observable_provenance,
        time_s=arrays["time_s"],
        measured_field_T=arrays["measured_field_T"],
        temperature_K=arrays["temperature_K"],
        saturation_mask=arrays["saturation_mask"],
        features=arrays["features"],
        valid_mask=arrays["valid_mask"],
        windows=windows,
    )


def _validate_array_shapes(arrays: dict[str, NDArray[Any]], metadata: PartitionMetadataV2) -> None:
    episode_count = len(metadata.episode_ids)
    if episode_count == 0 or len(metadata.lineage_ids) != episode_count:
        raise ValueError("partition episode and lineage identifiers are incomplete")
    if len(set(metadata.episode_ids)) != episode_count:
        raise ValueError("partition contains duplicate episode identifiers")
    sample_count = arrays["time_s"].shape[1] if arrays["time_s"].ndim == 2 else -1
    expected_windows = len(metadata.windows[0]) if metadata.windows else -1
    expected = {
        "time_s": (episode_count, sample_count),
        "measured_field_T": (episode_count, sample_count, 3),
        "temperature_K": (episode_count, sample_count),
        "saturation_mask": (episode_count, sample_count, 3),
        "features": (episode_count, expected_windows, 8),
        "valid_mask": (episode_count, expected_windows),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"numeric array shape violates the partition contract: {name}")
    if len(metadata.windows) != episode_count or len(metadata.coverage) != episode_count:
        raise ValueError("window and coverage ledgers must align with episodes")
    if len(metadata.raw_sources) != episode_count:
        raise ValueError("raw-source provenance must align with episodes")
    if any(len(items) != expected_windows for items in metadata.windows):
        raise ValueError("window ledgers have inconsistent shapes")


def _recompute_and_validate_windows(
    arrays: dict[str, NDArray[Any]],
    metadata: PartitionMetadataV2,
    profile: FeatureProfile,
) -> tuple[tuple[WindowFeatureRecord, ...], ...]:
    result: list[tuple[WindowFeatureRecord, ...]] = []
    for index, (episode_id, lineage_id) in enumerate(
        zip(metadata.episode_ids, metadata.lineage_ids, strict=True)
    ):
        raw_source = metadata.raw_sources[index]
        if raw_source.sensor_id != "M1" or raw_source.end_index > arrays["time_s"].shape[1]:
            raise ValueError("raw-source provenance is inconsistent with observable arrays")
        series = MeasuredVectorSeries(
            acquisition_id=episode_id,
            sensor_id=raw_source.sensor_id,
            sampling_rate_hz=profile.sampling_rate_hz,
            time_s=arrays["time_s"][index].tolist(),
            measured_field=[tuple(row) for row in arrays["measured_field_T"][index]],
            field_unit=metadata.observable_provenance.field_unit,
            temperature_k=arrays["temperature_K"][index].tolist(),
            saturation_mask=[tuple(row) for row in arrays["saturation_mask"][index]],
        )
        response = extract_windowed_magnetometer_features(
            FeatureExtractionRequest(
                series=series,
                channel=profile.channel,
                window=FeatureWindowConfiguration(
                    duration_s=profile.window_duration_s,
                    overlap_fraction=profile.overlap_fraction,
                ),
            )
        )
        if response.profile.model_dump(mode="json") != profile.model_dump(mode="json"):
            raise ValueError("versioned extractor profile differs from the archive")
        calculated = np.asarray([item.features.values for item in response.windows])
        if not np.allclose(
            calculated, arrays["features"][index], rtol=FEATURE_RTOL, atol=FEATURE_ATOL
        ):
            raise ValueError("archived features differ from raw-observable recalculation")
        calculated_valid = np.asarray(
            [item.quality.valid_for_quantum for item in response.windows], dtype=np.bool_
        )
        if not np.array_equal(calculated_valid, arrays["valid_mask"][index]):
            raise ValueError("valid_mask differs from recalculated window quality")
        archived = metadata.windows[index]
        if len(archived) != len(response.windows):
            raise ValueError("window ledger length differs from recalculated windows")
        rebuilt: list[WindowFeatureRecord] = []
        for window_index, (stored, current) in enumerate(
            zip(archived, response.windows, strict=True)
        ):
            current_metadata = current.model_dump(
                mode="json", exclude={"features", "provenance_token"}
            )
            if stored.model_dump(mode="json") != current_metadata:
                raise ValueError("window indices, times, identity or quality are inconsistent")
            rebuilt.append(
                current.model_copy(
                    update={
                        "features": current.features.model_copy(
                            update={"values": tuple(arrays["features"][index, window_index])}
                        ),
                        "provenance_token": "",
                    }
                )
            )
        coverage = metadata.coverage[index]
        if coverage.episode_id != episode_id or coverage.lineage_id != lineage_id:
            raise ValueError("coverage identity differs from partition identity")
        accepted = int(np.count_nonzero(calculated_valid))
        if (
            coverage.proposed_windows != len(response.windows)
            or coverage.accepted_windows != accepted
            or coverage.rejected_windows != len(response.windows) - accepted
        ):
            raise ValueError("coverage accounting differs from recalculated quality")
        result.append(tuple(rebuilt))
    return tuple(result)


def load_labels(
    dataset_path: Path,
    partition: DatasetPartition,
    *,
    authorization: TestAccessAuthorization | None = None,
) -> tuple[EpisodeLabel, ...]:
    path = dataset_path.resolve()
    manifest = verify_archive_opaque(path)
    if isinstance(manifest, LegacyDatasetManifest):
        raise ValueError("legacy v1 labels require an explicitly authorized migration")
    _authorize_if_test(path, manifest, partition, authorization)
    return _load_labels_semantic(path, manifest, partition)


def _load_labels_semantic(
    path: Path, manifest: DatasetManifest, partition: DatasetPartition
) -> tuple[EpisodeLabel, ...]:
    relative = f"labels/{partition.value}.json"
    payload = _read_json(_validated_file(path, relative))
    if payload.get("schema_version") != "aqse.partition-labels.v2":
        raise ValueError("unsupported partition-label schema")
    if payload.get("partition") != partition.value:
        raise ValueError("label partition identity mismatch")
    identity = _read_json(_validated_file(path, IDENTITY_NAME))
    if identity["partition_label_sha256"].get(partition.value) != _canonical_digest(payload):
        raise ValueError("labels are not bound to scientific identity")
    return tuple(EpisodeLabel.model_validate(item) for item in payload.get("labels", ()))


def load_generation_channel(
    dataset_path: Path,
    partition: DatasetPartition,
    *,
    authorization: TestAccessAuthorization | None = None,
) -> tuple[tuple[EpisodePlan, ...], tuple[SplitAssignment, ...]]:
    path = dataset_path.resolve()
    manifest = verify_archive_opaque(path)
    if isinstance(manifest, LegacyDatasetManifest):
        raise ValueError("legacy v1 generation plans require an authorized migration")
    _authorize_if_test(path, manifest, partition, authorization)
    return _load_generation_semantic(path, manifest, partition)


def _load_generation_semantic(
    path: Path, manifest: DatasetManifest, partition: DatasetPartition
) -> tuple[tuple[EpisodePlan, ...], tuple[SplitAssignment, ...]]:
    relative = f"generation/{partition.value}.json"
    payload = _read_json(_validated_file(path, relative))
    if payload.get("schema_version") != "aqse.generation-channel.v1":
        raise ValueError("unsupported generation-channel schema")
    if payload.get("partition") != partition.value:
        raise ValueError("generation partition identity mismatch")
    identity = _read_json(_validated_file(path, IDENTITY_NAME))
    if identity["generation_channel_sha256"].get(partition.value) != _canonical_digest(payload):
        raise ValueError("generation channel is not bound to scientific identity")
    plans = tuple(EpisodePlan.model_validate(item) for item in payload.get("plans", ()))
    assignments = tuple(
        SplitAssignment.model_validate(item) for item in payload.get("assignments", ())
    )
    bindings = payload.get("observation_bindings", ())
    if len(plans) != len(assignments) or len(plans) != len(bindings):
        raise ValueError("generation plans, assignments and bindings are incomplete")
    for plan, assignment, binding in zip(plans, assignments, bindings, strict=True):
        if (
            plan.episode_id != assignment.episode_id
            or plan.lineage_id != assignment.lineage_id
            or assignment.partition is not partition
            or binding.get("episode_id") != plan.episode_id
            or binding.get("lineage_id") != plan.lineage_id
        ):
            raise ValueError("generation identity references are inconsistent")
    return plans, assignments


def _validate_archive_candidate(path: Path, manifest: DatasetManifest) -> None:
    observed = verify_archive_opaque_at_staging(path, manifest)
    software = SoftwareProvenance.model_validate(
        _read_json(_validated_file(path, SOFTWARE_PROVENANCE_NAME))
    )
    validate_software_provenance(software, verify_effective_sources=True)
    for summary in observed.partition_summaries:
        partition = summary.partition
        observations = _load_observations_semantic(path, observed, partition)
        labels = _load_labels_semantic(path, observed, partition)
        plans, assignments = _load_generation_semantic(path, observed, partition)
        if observations.episode_ids != tuple(item.episode_id for item in labels):
            raise ValueError("observable and label episode order differs")
        if observations.episode_ids != tuple(item.episode_id for item in plans):
            raise ValueError("observable and generation episode order differs")
        if observations.lineage_ids != tuple(item.lineage_id for item in labels):
            raise ValueError("observable and label lineage order differs")
        if observations.episode_ids != tuple(item.episode_id for item in assignments):
            raise ValueError("observable and assignment episode order differs")
        generation_payload = _read_json(_validated_file(path, f"generation/{partition.value}.json"))
        metadata = PartitionMetadataV2.model_validate(
            _read_json(_validated_file(path, f"partitions/{partition.value}/metadata.json"))
        )
        for index, (plan, binding, raw_source) in enumerate(
            zip(
                plans,
                generation_payload["observation_bindings"],
                metadata.raw_sources,
                strict=True,
            )
        ):
            content_digest = _observation_content_digest(
                observations.time_s[index],
                observations.measured_field_T[index],
                observations.temperature_K[index],
                observations.saturation_mask[index],
            )
            expected_binding = _canonical_digest(
                {
                    "schema_version": "aqse.observation-binding.v1",
                    "episode_id": plan.episode_id,
                    "lineage_id": plan.lineage_id,
                    "content_digest": content_digest,
                    "raw_source": raw_source.model_dump(mode="json"),
                }
            )
            if binding.get("raw_source") != raw_source.model_dump(mode="json"):
                raise ValueError("generation raw-source binding differs from observable metadata")
            if binding.get("observation_content_digest") != content_digest:
                raise ValueError("generation content digest differs from observable arrays")
            if binding.get("observation_binding_digest") != expected_binding:
                raise ValueError("generation identity/provenance binding is invalid")


def verify_archive_opaque_at_staging(path: Path, manifest: DatasetManifest) -> DatasetManifest:
    """Opaque candidate verification before the final dataset-id directory exists."""

    parsed = _manifest_for_path(path)
    if not isinstance(parsed, DatasetManifest) or parsed != manifest:
        raise ValueError("candidate manifest differs from the writer manifest")
    records = {record.relative_path: record for record in manifest.files}
    if len(records) != len(manifest.files):
        raise ValueError("candidate manifest contains duplicate paths")
    for record in manifest.files:
        target = _validated_file(path, record.relative_path)
        if target.stat().st_size != record.byte_count or file_sha256(target) != record.file_sha256:
            raise ValueError(f"candidate file integrity failed: {record.relative_path}")
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if (item.is_file() or item.is_symlink()) and item.name != MANIFEST_NAME
    }
    if actual != set(records):
        raise ValueError("candidate contains missing or unexpected scientific files")
    identity = _read_json(_validated_file(path, IDENTITY_NAME))
    if _canonical_digest(identity) != manifest.scientific_digest:
        raise ValueError("candidate scientific identity digest mismatch")
    return parsed


def resign_partition_windows(
    dataset_path: Path,
    partition: DatasetPartition,
    *,
    authorization: TestAccessAuthorization | None = None,
) -> tuple[WindowFeatureRecord, ...]:
    """Recalculate from raw observables, then issue fresh process-local HMAC tokens."""

    loaded = load_observations(dataset_path, partition, authorization=authorization)
    records: list[WindowFeatureRecord] = []
    for episode_windows in loaded.windows:
        for unsigned in episode_windows:
            signed = unsigned.model_copy(
                update={"provenance_token": sign_window_record(loaded.profile, unsigned)}
            )
            if not verify_window_record(loaded.profile, signed):
                raise ValueError("failed to re-sign a recalculated feature window")
            records.append(signed)
    return tuple(records)


def _initialize_ledger(path: Path, manifest: DatasetManifest) -> None:
    payload = {
        "schema_version": "aqse.test-access-ledger.v2",
        "sequence": 0,
        "event": "sealed",
        "dataset_id": manifest.dataset_id,
        "scientific_digest": manifest.scientific_digest,
        "occurred_at_utc": _utc_now(),
        "actor": "AQSE 1D.1 archiver",
        "reason": "test partition sealed at dataset creation",
        "fixture_only": False,
        "previous_entry_sha256": None,
    }
    payload["entry_sha256"] = _canonical_digest(payload)
    entry = TestAccessLedgerEntry.model_validate(payload)
    (path / LEDGER_NAME).write_bytes(canonical_json_bytes(entry.model_dump(mode="json")) + b"\n")


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
    _append_ledger(path, manifest, authorization)


def _append_ledger(
    path: Path, manifest: DatasetManifest, authorization: TestAccessAuthorization
) -> None:
    ledger_path = _validated_file(path, LEDGER_NAME)
    lock_path = path / LEDGER_LOCK_NAME
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if ledger_path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("test access ledger exceeds the bounded load limit")
        entries = [
            TestAccessLedgerEntry.model_validate_json(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        _verify_ledger(entries, manifest)
        payload = {
            "schema_version": "aqse.test-access-ledger.v2",
            "sequence": len(entries),
            "event": "opened",
            "dataset_id": manifest.dataset_id,
            "scientific_digest": manifest.scientific_digest,
            "occurred_at_utc": _utc_now(),
            "actor": authorization.actor,
            "reason": authorization.reason,
            "fixture_only": authorization.fixture_only,
            "previous_entry_sha256": entries[-1].entry_sha256,
        }
        payload["entry_sha256"] = _canonical_digest(payload)
        entry = TestAccessLedgerEntry.model_validate(payload)
        temporary = ledger_path.with_name(f".{LEDGER_NAME}.{uuid4().hex}.tmp")
        temporary.write_bytes(
            b"".join(
                canonical_json_bytes(item.model_dump(mode="json")) + b"\n"
                for item in (*entries, entry)
            )
        )
        os.replace(temporary, ledger_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _verify_ledger(entries: list[TestAccessLedgerEntry], manifest: DatasetManifest) -> None:
    if not entries or entries[0].event != "sealed":
        raise ValueError("test access ledger has no sealing genesis entry")
    previous: str | None = None
    for sequence, entry in enumerate(entries):
        if (
            entry.sequence != sequence
            or entry.dataset_id != manifest.dataset_id
            or entry.scientific_digest != manifest.scientific_digest
        ):
            raise ValueError("test access ledger sequence or dataset identity is invalid")
        if entry.previous_entry_sha256 != previous:
            raise ValueError("test access ledger chain is invalid")
        payload = entry.model_dump(mode="json", exclude={"entry_sha256"})
        if entry.entry_sha256 != _canonical_digest(payload):
            raise ValueError("test access ledger entry digest is invalid")
        previous = entry.entry_sha256


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
