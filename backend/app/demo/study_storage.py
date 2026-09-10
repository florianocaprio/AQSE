from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from app.demo.protocol import FROZEN_NETWORK_DEMO_PROTOCOL
from app.demo.study_models import (
    DemoTestLedgerEntry,
    LoadedStudyPartition,
    NetworkStudyBuild,
    NetworkStudyLabel,
    NetworkStudyManifest,
    NetworkStudyObservation,
    StudyFileRecord,
    StudyPartition,
)
from app.features.state8 import state8_profile, state8_profile_fingerprint
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
)
from app.training.canonical import canonical_json_bytes, file_sha256
from app.training.storage import artifact_limit_bytes, artifact_root

HISTORICAL_DATASET_ID = "aqse-development-064acca20fc788c6"
HISTORICAL_TEST_LEDGER_SHA256 = (
    "e0d4898141d8be07f4a4f1af7582791eae61994ced52bb7b3dba30a7ddda4b70"
)
MAX_DEMO_JSON_BYTES = 64 * 1024 * 1024
PARTITION_COUNTS: dict[StudyPartition, int] = {
    "train": 96,
    "validation": 32,
    "test": 32,
}
OBSERVATION_FILES: dict[StudyPartition, str] = {
    partition: f"observations-{partition}.json" for partition in PARTITION_COUNTS
}
LABEL_FILES: dict[StudyPartition, str] = {
    partition: f"labels-{partition}.json" for partition in PARTITION_COUNTS
}
SCIENTIFIC_FILES = {
    "protocol.json",
    "generation.json",
    *OBSERVATION_FILES.values(),
    *LABEL_FILES.values(),
}
STUDY_FILES = {
    *SCIENTIFIC_FILES,
    "execution.json",
    "manifest.json",
    "test-access-ledger.jsonl",
}


def _demo_root(root: Path | None = None) -> Path:
    resolved = (root or artifact_root()).resolve()
    target = resolved / "network-demo"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _studies_root(root: Path | None = None) -> Path:
    target = _demo_root(root) / "studies"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_bytes(path: Path, payload: bytes, mode: int = 0o444) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)


def _json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _scientific_payloads(build: NetworkStudyBuild) -> dict[str, bytes]:
    payloads = {
        "protocol.json": _json_bytes(
            FROZEN_NETWORK_DEMO_PROTOCOL.model_dump(mode="json")
        ),
        "generation.json": _json_bytes(
            [item.model_dump(mode="json") for item in build.episode_plans]
        ),
    }
    for partition in PARTITION_COUNTS:
        payloads[OBSERVATION_FILES[partition]] = _json_bytes(
            [
                item.model_dump(mode="json")
                for item in build.observations
                if item.partition == partition
            ]
        )
        payloads[LABEL_FILES[partition]] = _json_bytes(
            [
                item.model_dump(mode="json")
                for item in build.labels
                if item.partition == partition
            ]
        )
    return payloads


def _scientific_digest(payloads: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payloads):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(payloads[name])
        digest.update(b"\x00")
    return digest.hexdigest()


def _ledger_entry(
    *,
    artifact_id: str,
    sequence: int,
    event: str,
    reason: str,
    selection_freeze_id: str | None,
    previous_entry_sha256: str | None,
) -> DemoTestLedgerEntry:
    payload = {
        "schema_version": "aqse.network-demo.test-ledger.v1",
        "study_artifact_id": artifact_id,
        "sequence": sequence,
        "event": event,
        "reason": reason,
        "selection_freeze_id": selection_freeze_id,
        "previous_entry_sha256": previous_entry_sha256,
    }
    return DemoTestLedgerEntry(
        **payload,
        entry_sha256=hashlib.sha256(_json_bytes(payload)).hexdigest(),
    )


def verify_historical_test_ledger(root: Path | None = None) -> str:
    resolved = (root or artifact_root()).resolve()
    ledger = resolved / HISTORICAL_DATASET_ID / "test-access-ledger.jsonl"
    if ledger.is_symlink() or not ledger.is_file():
        raise RuntimeError("historical Milestone 1D TEST ledger is unavailable")
    digest = file_sha256(ledger)
    if digest != HISTORICAL_TEST_LEDGER_SHA256:
        raise RuntimeError("historical Milestone 1D TEST ledger changed")
    return digest


def _root_usage(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def write_network_study(
    build: NetworkStudyBuild,
    *,
    root: Path | None = None,
) -> tuple[Path, NetworkStudyManifest, bool]:
    resolved_artifact_root = (root or artifact_root()).resolve()
    verify_historical_test_ledger(resolved_artifact_root)
    payloads = _scientific_payloads(build)
    content_digest = _scientific_digest(payloads)
    artifact_id = f"aqse-network-study-{content_digest[:16]}"
    parent = _studies_root(resolved_artifact_root)
    target = parent / artifact_id
    lock_path = parent / ".network-study-write.lock"
    with lock_path.open("a+b") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists():
            manifest = verify_network_study(target)
            if manifest.content_digest != content_digest:
                raise RuntimeError(
                    "existing study identity has different scientific content"
                )
            return target, manifest, True

        execution = {
            "schema_version": "aqse.network-demo.execution.v1",
            "artifact_id": artifact_id,
            "created_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "python_version": platform.python_version(),
            "artifact_root": str(resolved_artifact_root),
        }
        payloads_with_execution = {**payloads, "execution.json": _json_bytes(execution)}
        records = tuple(
            StudyFileRecord(
                relative_path=name,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            for name, payload in sorted(payloads_with_execution.items())
        )
        manifest = NetworkStudyManifest(
            artifact_id=artifact_id,
            content_digest=content_digest,
            protocol_digest=build.protocol_digest,
            feature_profile_fingerprints={
                profile_id: state8_profile_fingerprint(state8_profile(profile_id))
                for profile_id in (LOCAL_STATE8_PROFILE_ID, NETWORK_STATE8_PROFILE_ID)
            },
            partition_counts=dict(PARTITION_COUNTS),
            files=records,
        )
        manifest_payload = _json_bytes(manifest.model_dump(mode="json"))
        genesis = _ledger_entry(
            artifact_id=artifact_id,
            sequence=0,
            event="sealed",
            reason="new study TEST sealed before any model fitting",
            selection_freeze_id=None,
            previous_entry_sha256=None,
        )
        ledger_payload = _json_bytes(genesis.model_dump(mode="json")) + b"\n"
        required = sum(len(item) for item in payloads_with_execution.values())
        required += len(manifest_payload) + len(ledger_payload)
        if _root_usage(resolved_artifact_root) + required > artifact_limit_bytes():
            raise RuntimeError("AQSE artifact limit would be exceeded by the network study")

        staging = Path(mkdtemp(prefix=f".{artifact_id}.", dir=parent))
        try:
            for name, payload in payloads_with_execution.items():
                _write_bytes(staging / name, payload)
            _write_bytes(staging / "manifest.json", manifest_payload)
            _write_bytes(staging / "test-access-ledger.jsonl", ledger_payload, mode=0o600)
            verify_network_study(staging, expected_artifact_id=artifact_id)
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        verify_network_study(target)
        verify_historical_test_ledger(resolved_artifact_root)
        return target, manifest, False


def _read_bounded(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing study file: {path.name}")
    if path.stat().st_size > MAX_DEMO_JSON_BYTES:
        raise ValueError(f"study file exceeds the bounded reader: {path.name}")
    return path.read_bytes()


def verify_network_study(
    path: Path,
    *,
    expected_artifact_id: str | None = None,
) -> NetworkStudyManifest:
    resolved = path.resolve()
    if not resolved.is_dir() or path.is_symlink():
        raise ValueError("network study path must be a real directory")
    actual = {item.name for item in resolved.iterdir()}
    if actual != STUDY_FILES:
        raise ValueError("network study contains missing or unexpected files")
    manifest = NetworkStudyManifest.model_validate_json(
        _read_bounded(resolved / "manifest.json")
    )
    directory_identity = expected_artifact_id or resolved.name
    if directory_identity != manifest.artifact_id:
        raise ValueError("network study directory and manifest identities differ")
    if manifest.artifact_id != f"aqse-network-study-{manifest.content_digest[:16]}":
        raise ValueError("network study content and artifact identities differ")
    expected_record_names = SCIENTIFIC_FILES | {"execution.json"}
    record_names = [record.relative_path for record in manifest.files]
    if len(record_names) != len(set(record_names)) or set(record_names) != expected_record_names:
        raise ValueError("network study manifest file allowlist is invalid")
    for record in manifest.files:
        if record.relative_path not in expected_record_names:
            raise ValueError("network study manifest contains an unsafe file path")
        payload = _read_bounded(resolved / record.relative_path)
        if len(payload) != record.size_bytes:
            raise ValueError(f"study file size changed: {record.relative_path}")
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ValueError(f"study file digest changed: {record.relative_path}")
    payloads = {name: _read_bounded(resolved / name) for name in SCIENTIFIC_FILES}
    if _scientific_digest(payloads) != manifest.content_digest:
        raise ValueError("network study scientific digest is invalid")
    if manifest.protocol_digest != FROZEN_NETWORK_DEMO_PROTOCOL.digest:
        raise ValueError("network study protocol is incompatible")
    if payloads["protocol.json"] != _json_bytes(
        FROZEN_NETWORK_DEMO_PROTOCOL.model_dump(mode="json")
    ):
        raise ValueError("network study protocol payload is incompatible")
    expected_fingerprints = {
        profile_id: state8_profile_fingerprint(state8_profile(profile_id))
        for profile_id in (LOCAL_STATE8_PROFILE_ID, NETWORK_STATE8_PROFILE_ID)
    }
    if manifest.feature_profile_fingerprints != expected_fingerprints:
        raise ValueError("network study feature profiles are incompatible")
    if manifest.partition_counts != PARTITION_COUNTS:
        raise ValueError("network study partition counts are incompatible")
    execution = json.loads(_read_bounded(resolved / "execution.json"))
    if (
        not isinstance(execution, dict)
        or execution.get("schema_version") != "aqse.network-demo.execution.v1"
        or execution.get("artifact_id") != manifest.artifact_id
    ):
        raise ValueError("network study execution metadata is incompatible")
    return manifest


def _decode_observations(
    path: Path,
    partition: StudyPartition,
) -> tuple[NetworkStudyObservation, ...]:
    payload = json.loads(_read_bounded(path / OBSERVATION_FILES[partition]))
    if not isinstance(payload, list) or len(payload) != PARTITION_COUNTS[partition]:
        raise ValueError("network study observation partition has an invalid row count")
    if any(item.get("partition") != partition for item in payload if isinstance(item, dict)):
        raise ValueError("network study observation was stored in the wrong partition")
    return tuple(NetworkStudyObservation.model_validate(item) for item in payload)


def _decode_labels(
    path: Path,
    partition: StudyPartition,
) -> tuple[NetworkStudyLabel, ...]:
    payload = json.loads(_read_bounded(path / LABEL_FILES[partition]))
    if not isinstance(payload, list) or len(payload) != PARTITION_COUNTS[partition]:
        raise ValueError("network study label partition has an invalid row count")
    if any(item.get("partition") != partition for item in payload if isinstance(item, dict)):
        raise ValueError("network study label was stored in the wrong partition")
    return tuple(NetworkStudyLabel.model_validate(item) for item in payload)


def load_development_partition(
    path: Path,
    partition: StudyPartition,
    *,
    include_labels: bool,
) -> LoadedStudyPartition:
    if partition == "test":
        raise PermissionError("new-study TEST requires the one-time frozen access path")
    verify_network_study(path)
    observations = _decode_observations(path, partition)
    labels = None
    if include_labels:
        labels = _decode_labels(path, partition)
    return LoadedStudyPartition(
        partition=partition,
        observations=observations,
        labels=labels,
    )


def _decode_demo_test_ledger(
    payload: bytes,
    *,
    artifact_id: str,
) -> tuple[DemoTestLedgerEntry, ...]:
    rows = [line for line in payload.splitlines() if line]
    entries = tuple(DemoTestLedgerEntry.model_validate_json(line) for line in rows)
    if not 1 <= len(entries) <= 3:
        raise ValueError("new-study TEST ledger has an invalid event count")
    for index, entry in enumerate(entries):
        expected = _ledger_entry(
            artifact_id=entry.study_artifact_id,
            sequence=entry.sequence,
            event=entry.event,
            reason=entry.reason,
            selection_freeze_id=entry.selection_freeze_id,
            previous_entry_sha256=entry.previous_entry_sha256,
        )
        if (
            entry != expected
            or entry.sequence != index
            or entry.study_artifact_id != artifact_id
        ):
            raise ValueError("new-study TEST ledger identity is invalid")
        if index == 0:
            if (
                entry.event != "sealed"
                or entry.reason != "new study TEST sealed before any model fitting"
                or entry.selection_freeze_id is not None
                or entry.previous_entry_sha256 is not None
            ):
                raise ValueError("new-study TEST ledger genesis is invalid")
        else:
            expected_reason = (
                "frozen-network-demo-test-observations"
                if index == 1
                else "frozen-network-demo-final-labels"
            )
            if (
                entry.event != "opened"
                or entry.reason != expected_reason
                or not entry.selection_freeze_id
                or entry.previous_entry_sha256 != entries[index - 1].entry_sha256
            ):
                raise ValueError("new-study TEST ledger chain is invalid")
            if (
                index == 2
                and entry.selection_freeze_id != entries[1].selection_freeze_id
            ):
                raise ValueError("new-study TEST ledger selection identity changed")
    return entries


def read_demo_test_ledger(path: Path) -> tuple[DemoTestLedgerEntry, ...]:
    payload = _read_bounded(path / "test-access-ledger.jsonl")
    return _decode_demo_test_ledger(payload, artifact_id=path.resolve().name)


def _append_test_event(path: Path, *, freeze_id: str, reason: str, sequence: int) -> None:
    if not freeze_id or len(freeze_id) > 256:
        raise ValueError("a bounded selection freeze identity is required")
    ledger_path = path / "test-access-ledger.jsonl"
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise ValueError("new-study TEST ledger is missing or unsafe")
    with ledger_path.open("r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        payload = handle.read(MAX_DEMO_JSON_BYTES + 1)
        if len(payload) > MAX_DEMO_JSON_BYTES:
            raise ValueError("new-study TEST ledger exceeds the bounded reader")
        entries = _decode_demo_test_ledger(
            payload,
            artifact_id=path.resolve().name,
        )
        if len(entries) != sequence:
            raise PermissionError("new-study TEST access plan has already advanced")
        entry = _ledger_entry(
            artifact_id=entries[0].study_artifact_id,
            sequence=sequence,
            event="opened",
            reason=reason,
            selection_freeze_id=freeze_id,
            previous_entry_sha256=entries[-1].entry_sha256,
        )
        handle.seek(0, os.SEEK_END)
        handle.write(_json_bytes(entry.model_dump(mode="json")) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_new_test_observations(
    path: Path,
    *,
    selection_freeze_id: str,
) -> LoadedStudyPartition:
    verify_network_study(path)
    _append_test_event(
        path,
        freeze_id=selection_freeze_id,
        reason="frozen-network-demo-test-observations",
        sequence=1,
    )
    observations = _decode_observations(path, "test")
    return LoadedStudyPartition(partition="test", observations=observations)


def load_new_test_labels(
    path: Path,
    *,
    selection_freeze_id: str,
) -> tuple[NetworkStudyLabel, ...]:
    verify_network_study(path)
    entries = read_demo_test_ledger(path)
    if len(entries) != 2 or entries[1].selection_freeze_id != selection_freeze_id:
        raise PermissionError("TEST label access requires the matching observation event")
    _append_test_event(
        path,
        freeze_id=selection_freeze_id,
        reason="frozen-network-demo-final-labels",
        sequence=2,
    )
    return _decode_labels(path, "test")
