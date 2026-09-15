from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Mapping, Sequence
from uuid import uuid4

from pydantic import BaseModel

from app.training.canonical import canonical_json_bytes
from app.training.storage import artifact_limit_bytes, artifact_root

MAX_JSON_BYTES = 128 * 1024 * 1024
PILOT_EPISODE_COUNT = 20
FINAL_TEST_EPISODE_COUNT = 100
STUDY_DIRECTORY = "four-sensor-study-v1"
LEDGER_SCHEMA = "aqse.four-sensor-study.test-ledger.v1"

PILOT_FILES = {"generation.json", "observations.json", "labels.json"}
FINAL_DATASET_FILES = {"generation.json", "observations.json", "labels.json"}
TRUTH_KEYS = {
    "scenario",
    "label",
    "local_label",
    "network_label",
    "target",
    "truth",
    "truth_label",
    "generative_truth",
    "episode_seed",
    "generation_seed",
    "network_configuration",
    "events",
}


@dataclass(frozen=True)
class TestLedgerSnapshot:
    entries: tuple[dict[str, Any], ...]
    file_sha256: str


def _normalise(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _normalise(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("scientific JSON mappings require string keys")
        return {key: _normalise(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalise(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(_normalise(value)) + b"\n"


def _digest_payloads(payloads: Mapping[str, bytes], bindings: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(_normalise(bindings)))
    for name in sorted(payloads):
        digest.update(b"\x00")
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(payloads[name])
    return digest.hexdigest()


def _study_root(root: Path | None = None) -> Path:
    base = root or artifact_root()
    if base.exists() and base.is_symlink():
        raise ValueError("AQSE artifact root cannot be a symbolic link")
    base.mkdir(parents=True, exist_ok=True)
    target = base.resolve() / STUDY_DIRECTORY
    if target.exists() and target.is_symlink():
        raise ValueError("four-sensor study root cannot be a symbolic link")
    target.mkdir(parents=True, exist_ok=True)
    return target


def four_sensor_study_root(root: Path | None = None) -> Path:
    """Return the isolated root used by the four-sensor study."""

    return _study_root(root)


def _category(study_root: Path, name: str) -> Path:
    if name not in {
        "pilots",
        "protocol-freezes",
        "final-test-datasets",
        "model-bindings",
        "authorizations",
        "final-evaluations",
        "ledger-locks",
    }:
        raise ValueError("unknown four-sensor artifact category")
    path = study_root / name
    if path.exists() and path.is_symlink():
        raise ValueError("four-sensor artifact category cannot be a symbolic link")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_path(category: Path, artifact_id: str) -> Path:
    if not artifact_id or "/" in artifact_id or artifact_id in {".", ".."}:
        raise ValueError("unsafe four-sensor artifact identity")
    target = category / artifact_id
    if category.resolve() not in target.resolve(strict=False).parents:
        raise ValueError("four-sensor artifact path escapes its category")
    return target


def _read_bounded(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"four-sensor artifact file is missing or unsafe: {path.name}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_JSON_BYTES:
        raise ValueError(f"four-sensor artifact file exceeds its bound: {path.name}")
    return path.read_bytes()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(_read_bounded(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid four-sensor JSON artifact: {path.name}") from exc


def _write_bytes(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _root_usage(root: Path) -> int:
    return sum(
        item.stat().st_size
        for item in root.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def _manifest(
    *,
    artifact_id: str,
    artifact_kind: str,
    content_digest: str,
    payloads: Mapping[str, bytes],
    bindings: Mapping[str, Any],
    episode_count: int | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": f"aqse.four-sensor-study.{artifact_kind}-manifest.v1",
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind,
        "content_digest": content_digest,
        "bindings": _normalise(bindings),
        "files": {
            name: {
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in sorted(payloads.items())
        },
    }
    if episode_count is not None:
        value["episode_count"] = episode_count
    return value


def _verify_content_artifact(
    path: Path,
    *,
    artifact_kind: str,
    expected_files: set[str],
    extra_files: set[str] | None = None,
    expected_artifact_id: str | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_dir():
        raise ValueError("four-sensor artifact path must be a real directory")
    allowed = expected_files | {"manifest.json"} | (extra_files or set())
    if {item.name for item in resolved.iterdir()} != allowed:
        raise ValueError("four-sensor artifact contains missing or unexpected files")
    manifest = _read_json(resolved / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("four-sensor artifact manifest must be an object")
    if (
        manifest.get("schema_version")
        != f"aqse.four-sensor-study.{artifact_kind}-manifest.v1"
        or manifest.get("artifact_kind") != artifact_kind
        or manifest.get("artifact_id") != (expected_artifact_id or resolved.name)
        or set(manifest.get("files", {})) != expected_files
        or not isinstance(manifest.get("bindings"), dict)
    ):
        raise ValueError("four-sensor artifact manifest contract is invalid")
    artifact_id = manifest["artifact_id"]
    content_digest = manifest.get("content_digest")
    if (
        not isinstance(content_digest, str)
        or len(content_digest) != 64
        or not artifact_id.endswith(content_digest[:16])
    ):
        raise ValueError("four-sensor artifact content identity is invalid")
    payloads: dict[str, bytes] = {}
    for name in expected_files:
        if Path(name).name != name:
            raise ValueError("four-sensor manifest contains an unsafe filename")
        payload = _read_bounded(resolved / name)
        record = manifest["files"].get(name)
        if (
            not isinstance(record, dict)
            or record.get("byte_count") != len(payload)
            or record.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError(f"four-sensor artifact file identity changed: {name}")
        payloads[name] = payload
    if _digest_payloads(payloads, manifest["bindings"]) != content_digest:
        raise ValueError("four-sensor artifact content digest is invalid")
    return manifest


def _publish_content_artifact(
    *,
    root: Path | None,
    category_name: str,
    artifact_kind: str,
    artifact_prefix: str,
    values: Mapping[str, Any],
    bindings: Mapping[str, Any],
    episode_count: int | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    study_root = _study_root(root)
    category = _category(study_root, category_name)
    payloads = {name: _json_bytes(value) for name, value in values.items()}
    content_digest = _digest_payloads(payloads, bindings)
    artifact_id = f"{artifact_prefix}-{content_digest[:16]}"
    target = _artifact_path(category, artifact_id)
    lock_path = category / ".publish.lock"
    with lock_path.open("a+b") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists():
            manifest = _verify_content_artifact(
                target,
                artifact_kind=artifact_kind,
                expected_files=set(values),
            )
            if manifest["content_digest"] != content_digest:
                raise FileExistsError("immutable four-sensor artifact has different content")
            return target, manifest, True
        manifest = _manifest(
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            content_digest=content_digest,
            payloads=payloads,
            bindings=bindings,
            episode_count=episode_count,
        )
        staging = Path(mkdtemp(prefix=f".{artifact_id}.", dir=category))
        try:
            for name, payload in payloads.items():
                _write_bytes(staging / name, payload)
            _write_bytes(staging / "manifest.json", _json_bytes(manifest))
            _verify_content_artifact(
                staging,
                artifact_kind=artifact_kind,
                expected_files=set(values),
                expected_artifact_id=artifact_id,
            )
            if _root_usage(study_root.parent) > artifact_limit_bytes():
                raise OSError("four-sensor artifact exceeds the AQSE artifact byte limit")
            _fsync_directory(staging)
            os.replace(staging, target)
            _fsync_directory(category)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return target, _verify_content_artifact(
            target,
            artifact_kind=artifact_kind,
            expected_files=set(values),
        ), False


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise ValueError(f"four-sensor build is missing one of: {', '.join(names)}")


def _rows(value: Any, *names: str) -> list[Any]:
    rows = _field(value, *names)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise TypeError(f"four-sensor build channel {names[0]} must be a sequence")
    return list(rows)


def _row_field(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def _partition(row: Any) -> str | None:
    value = _row_field(row, "partition")
    return value.value if isinstance(value, Enum) else value


def _channel_ids(rows: Sequence[Any], *, channel: str) -> tuple[str, ...]:
    values = tuple(_row_field(row, "episode_id") for row in rows)
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{channel} channel requires a bounded episode_id per row")
    if len(set(values)) != len(values):
        raise ValueError(f"{channel} channel contains duplicate episode identifiers")
    return values  # type: ignore[return-value]


def _validate_truth_free_observations(observations: Sequence[Any]) -> None:
    for observation in observations:
        value = _normalise(observation)
        if not isinstance(value, dict):
            raise TypeError("four-sensor observations must serialize as objects")
        forbidden = TRUTH_KEYS.intersection(value)
        if forbidden:
            raise ValueError(
                "predictor observation channel contains forbidden truth fields: "
                + ", ".join(sorted(forbidden))
            )


def _validate_build_channels(
    build: Any,
    *,
    expected_count: int,
    expected_partition: str,
) -> tuple[list[Any], list[Any], list[Any]]:
    plans = _rows(build, "plans", "episode_plans")
    observations = _rows(build, "observations")
    labels = _rows(build, "labels")
    if not (len(plans) == len(observations) == len(labels) == expected_count):
        raise ValueError(
            f"four-sensor {expected_partition} build must contain exactly "
            f"{expected_count} rows in every channel"
        )
    plan_ids = _channel_ids(plans, channel="generation")
    observation_ids = _channel_ids(observations, channel="observation")
    label_ids = _channel_ids(labels, channel="label")
    if set(plan_ids) != set(observation_ids) or set(plan_ids) != set(label_ids):
        raise ValueError("four-sensor generation, observation and label channels differ")
    if any(_partition(plan) != expected_partition for plan in plans):
        raise ValueError(
            f"four-sensor {expected_partition} build contains another partition"
        )
    _validate_truth_free_observations(observations)
    return plans, observations, labels


def write_pilot(
    build: Any,
    *,
    root: Path | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    """Publish the non-canonical 20-episode pilot in its own namespace."""

    plans, observations, labels = _validate_build_channels(
        build,
        expected_count=PILOT_EPISODE_COUNT,
        expected_partition="pilot",
    )
    return _publish_content_artifact(
        root=root,
        category_name="pilots",
        artifact_kind="pilot",
        artifact_prefix="aqse-four-sensor-pilot",
        values={
            "generation.json": plans,
            "observations.json": observations,
            "labels.json": labels,
        },
        bindings={"study_id": "aqse-four-sensor-study-v1", "canonical": False},
        episode_count=PILOT_EPISODE_COUNT,
    )


def verify_pilot(path: Path) -> dict[str, Any]:
    manifest = _verify_content_artifact(
        path,
        artifact_kind="pilot",
        expected_files=PILOT_FILES,
    )
    if manifest.get("episode_count") != PILOT_EPISODE_COUNT:
        raise ValueError("four-sensor pilot manifest has an invalid episode count")
    return manifest


def write_protocol_freeze(
    protocol_payload: Any,
    *,
    pilot_path: Path,
    acceptance: Mapping[str, Any] | BaseModel,
    root: Path | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    """Freeze the protocol only after binding an accepted, separate pilot."""

    pilot_manifest = verify_pilot(pilot_path)
    acceptance_value = _normalise(acceptance)
    if not isinstance(acceptance_value, dict) or acceptance_value.get("accepted") is not True:
        raise ValueError("protocol freeze requires an explicit accepted pilot assessment")
    if acceptance_value.get("episode_count", PILOT_EPISODE_COUNT) != PILOT_EPISODE_COUNT:
        raise ValueError("pilot acceptance must describe exactly 20 noncanonical episodes")
    assessment_digest = acceptance_value.get("assessment_digest")
    if assessment_digest is not None:
        digest_payload = {
            key: value for key, value in acceptance_value.items() if key != "assessment_digest"
        }
        if assessment_digest != hashlib.sha256(
            canonical_json_bytes(digest_payload)
        ).hexdigest():
            raise ValueError("pilot assessment digest is invalid")
    protocol_value = _normalise(protocol_payload)
    if not isinstance(protocol_value, dict):
        raise TypeError("four-sensor protocol payload must serialize as an object")
    protocol_digest = hashlib.sha256(canonical_json_bytes(protocol_value)).hexdigest()
    binding = {
        "study_id": "aqse-four-sensor-study-v1",
        "pilot_artifact_id": pilot_manifest["artifact_id"],
        "pilot_content_digest": pilot_manifest["content_digest"],
        "protocol_digest": protocol_digest,
    }
    value = {
        "schema_version": "aqse.four-sensor-study.protocol-freeze.v1",
        "protocol": protocol_value,
        "protocol_digest": protocol_digest,
        "pilot_binding": {
            "artifact_id": pilot_manifest["artifact_id"],
            "content_digest": pilot_manifest["content_digest"],
        },
        "pilot_acceptance": acceptance_value,
    }
    effective_root = root or pilot_path.resolve().parents[2]
    if _study_root(effective_root) != pilot_path.resolve().parents[1]:
        raise ValueError("pilot and protocol freeze must share one isolated study root")
    return _publish_content_artifact(
        root=effective_root,
        category_name="protocol-freezes",
        artifact_kind="protocol-freeze",
        artifact_prefix="aqse-four-sensor-protocol-freeze",
        values={"protocol-freeze.json": value},
        bindings=binding,
    )


def verify_protocol_freeze(path: Path) -> dict[str, Any]:
    manifest = _verify_content_artifact(
        path,
        artifact_kind="protocol-freeze",
        expected_files={"protocol-freeze.json"},
    )
    payload = _read_json(path.resolve() / "protocol-freeze.json")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != "aqse.four-sensor-study.protocol-freeze.v1"
        or not isinstance(payload.get("protocol"), dict)
        or not isinstance(payload.get("pilot_binding"), dict)
        or not isinstance(payload.get("pilot_acceptance"), dict)
        or payload["pilot_acceptance"].get("accepted") is not True
    ):
        raise ValueError("four-sensor protocol freeze content is invalid")
    protocol_digest = hashlib.sha256(
        canonical_json_bytes(payload["protocol"])
    ).hexdigest()
    if (
        payload.get("protocol_digest") != protocol_digest
        or manifest["bindings"].get("protocol_digest") != protocol_digest
        or payload["pilot_binding"].get("artifact_id")
        != manifest["bindings"].get("pilot_artifact_id")
        or payload["pilot_binding"].get("content_digest")
        != manifest["bindings"].get("pilot_content_digest")
    ):
        raise ValueError("four-sensor protocol freeze bindings are invalid")
    return manifest


def load_protocol_freeze(path: Path) -> dict[str, Any]:
    """Load the public freeze envelope after full content verification."""

    verify_protocol_freeze(path)
    value = _read_json(path.resolve() / "protocol-freeze.json")
    assert isinstance(value, dict)
    return value


def _ledger_payload(entry: Mapping[str, Any]) -> bytes:
    return _json_bytes(entry)


def _ledger_entry(
    *,
    dataset_manifest: Mapping[str, Any],
    sequence: int,
    event: str,
    protocol_freeze: Mapping[str, Any],
    model_binding: Mapping[str, Any] | None,
    authorization: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
    previous_entry_sha256: str | None,
) -> dict[str, Any]:
    body = {
        "schema_version": LEDGER_SCHEMA,
        "dataset_id": dataset_manifest["artifact_id"],
        "dataset_content_digest": dataset_manifest["content_digest"],
        "sequence": sequence,
        "event": event,
        "protocol_freeze_id": protocol_freeze["artifact_id"],
        "protocol_freeze_content_digest": protocol_freeze["content_digest"],
        "model_binding_id": None if model_binding is None else model_binding["artifact_id"],
        "model_binding_content_digest": (
            None if model_binding is None else model_binding["content_digest"]
        ),
        "authorization_id": None if authorization is None else authorization["artifact_id"],
        "authorization_content_digest": (
            None if authorization is None else authorization["content_digest"]
        ),
        "evaluation_id": None if evaluation is None else evaluation["artifact_id"],
        "evaluation_content_digest": (
            None if evaluation is None else evaluation["content_digest"]
        ),
        "previous_entry_sha256": previous_entry_sha256,
    }
    return {
        **body,
        "entry_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def write_final_test_dataset(
    build: Any,
    *,
    protocol_freeze_path: Path,
    root: Path | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    """Publish exactly 100 TEST episodes with truth kept outside observations."""

    plans, observations, labels = _validate_build_channels(
        build,
        expected_count=FINAL_TEST_EPISODE_COUNT,
        expected_partition="test",
    )
    protocol_manifest = verify_protocol_freeze(protocol_freeze_path)
    expected_freeze_digest = protocol_manifest["content_digest"]
    for plan in plans:
        bound_digest = _row_field(plan, "protocol_freeze_digest")
        if bound_digest != expected_freeze_digest:
            raise ValueError("final TEST plan is not bound to the protocol freeze")
    pilot_ids = set(
        _channel_ids(
            _read_json(pilot_path(protocol_freeze_path) / "generation.json"),
            channel="pilot generation",
        )
    )
    if pilot_ids.intersection(_channel_ids(plans, channel="final TEST generation")):
        raise ValueError("pilot and final TEST episode identities overlap")
    values = {
        "generation.json": plans,
        "observations.json": observations,
        "labels.json": labels,
    }
    payloads = {name: _json_bytes(value) for name, value in values.items()}
    bindings = {
        "study_id": "aqse-four-sensor-study-v1",
        "protocol_freeze_artifact_id": protocol_manifest["artifact_id"],
        "protocol_freeze_content_digest": protocol_manifest["content_digest"],
        "pilot_artifact_id": protocol_manifest["bindings"]["pilot_artifact_id"],
        "pilot_content_digest": protocol_manifest["bindings"]["pilot_content_digest"],
    }
    content_digest = _digest_payloads(payloads, bindings)
    artifact_id = f"aqse-four-sensor-final-test-{content_digest[:16]}"
    effective_root = root or protocol_freeze_path.resolve().parents[2]
    study_root = _study_root(effective_root)
    if study_root != protocol_freeze_path.resolve().parents[1]:
        raise ValueError("protocol freeze and final TEST must share one study root")
    category = _category(study_root, "final-test-datasets")
    target = _artifact_path(category, artifact_id)
    lock_path = category / ".publish.lock"
    with lock_path.open("a+b") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists():
            manifest = verify_final_test_dataset(target)
            if manifest["content_digest"] != content_digest:
                raise FileExistsError("immutable final TEST identity has different content")
            return target, manifest, True
        manifest = _manifest(
            artifact_id=artifact_id,
            artifact_kind="final-test-dataset",
            content_digest=content_digest,
            payloads=payloads,
            bindings=bindings,
            episode_count=FINAL_TEST_EPISODE_COUNT,
        )
        genesis = _ledger_entry(
            dataset_manifest=manifest,
            sequence=0,
            event="sealed",
            protocol_freeze=protocol_manifest,
            model_binding=None,
            authorization=None,
            evaluation=None,
            previous_entry_sha256=None,
        )
        staging = Path(mkdtemp(prefix=f".{artifact_id}.", dir=category))
        try:
            for name, payload in payloads.items():
                _write_bytes(staging / name, payload)
            _write_bytes(staging / "manifest.json", _json_bytes(manifest))
            _write_bytes(
                staging / "test-access-ledger.jsonl",
                _ledger_payload(genesis),
                mode=0o600,
            )
            _verify_content_artifact(
                staging,
                artifact_kind="final-test-dataset",
                expected_files=FINAL_DATASET_FILES,
                extra_files={"test-access-ledger.jsonl"},
                expected_artifact_id=artifact_id,
            )
            _decode_ledger(
                _read_bounded(staging / "test-access-ledger.jsonl"),
                dataset_manifest=manifest,
            )
            if _root_usage(study_root.parent) > artifact_limit_bytes():
                raise OSError("final TEST dataset exceeds the AQSE artifact byte limit")
            _fsync_directory(staging)
            os.replace(staging, target)
            _fsync_directory(category)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return target, verify_final_test_dataset(target), False


def pilot_path(protocol_freeze_path: Path) -> Path:
    manifest = verify_protocol_freeze(protocol_freeze_path)
    study_root = protocol_freeze_path.resolve().parents[1]
    path = _artifact_path(
        _category(study_root, "pilots"),
        manifest["bindings"]["pilot_artifact_id"],
    )
    pilot_manifest = verify_pilot(path)
    if pilot_manifest["content_digest"] != manifest["bindings"]["pilot_content_digest"]:
        raise ValueError("protocol freeze refers to a different pilot payload")
    return path


def verify_final_test_dataset(path: Path) -> dict[str, Any]:
    manifest = _verify_content_artifact(
        path,
        artifact_kind="final-test-dataset",
        expected_files=FINAL_DATASET_FILES,
        extra_files={"test-access-ledger.jsonl"},
    )
    if manifest.get("episode_count") != FINAL_TEST_EPISODE_COUNT:
        raise ValueError("final TEST manifest must declare exactly 100 episodes")
    read_test_ledger(path)
    return manifest


def _write_bound_payload(
    payload: Any,
    *,
    root: Path | None,
    category_name: str,
    artifact_kind: str,
    artifact_prefix: str,
    content_name: str,
    protocol_freeze_path: Path,
    dataset_path: Path,
    model_binding_path: Path | None = None,
    authorization_path: Path | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    protocol = verify_protocol_freeze(protocol_freeze_path)
    dataset = verify_final_test_dataset(dataset_path)
    if (
        dataset["bindings"].get("protocol_freeze_artifact_id") != protocol["artifact_id"]
        or dataset["bindings"].get("protocol_freeze_content_digest")
        != protocol["content_digest"]
    ):
        raise ValueError("final TEST dataset and protocol freeze are not bound")
    payload_value = _normalise(payload)
    if not isinstance(payload_value, dict) or not payload_value:
        raise ValueError("bound scientific payload must be a non-empty object")
    bindings: dict[str, Any] = {
        "protocol_freeze_artifact_id": protocol["artifact_id"],
        "protocol_freeze_content_digest": protocol["content_digest"],
        "dataset_id": dataset["artifact_id"],
        "dataset_content_digest": dataset["content_digest"],
    }
    if model_binding_path is not None:
        model = verify_model_binding(model_binding_path)
        _validate_model_dataset_binding(model, protocol, dataset)
        bindings.update(
            {
                "model_binding_id": model["artifact_id"],
                "model_binding_content_digest": model["content_digest"],
            }
        )
    if authorization_path is not None:
        if model_binding_path is None:
            raise ValueError("an authorization binding requires a model binding")
        authorization = verify_test_authorization(authorization_path)
        if authorization["bindings"] != bindings:
            raise PermissionError("authorization differs from the bound scientific set")
        bindings.update(
            {
                "authorization_id": authorization["artifact_id"],
                "authorization_content_digest": authorization["content_digest"],
            }
        )
    wrapper = {
        "schema_version": f"aqse.four-sensor-study.{artifact_kind}.v1",
        "payload": payload_value,
        "payload_digest": hashlib.sha256(canonical_json_bytes(payload_value)).hexdigest(),
        "bindings": bindings,
    }
    effective_root = root or dataset_path.resolve().parents[2]
    if _study_root(effective_root) != dataset_path.resolve().parents[1]:
        raise ValueError("bound artifact and final TEST must share one study root")
    return _publish_content_artifact(
        root=effective_root,
        category_name=category_name,
        artifact_kind=artifact_kind,
        artifact_prefix=artifact_prefix,
        values={content_name: wrapper},
        bindings=bindings,
    )


def write_model_binding(
    payload: Any,
    *,
    protocol_freeze_path: Path,
    dataset_path: Path,
    root: Path | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    """Bind a caller-provided, already frozen model; this performs no fitting."""

    return _write_bound_payload(
        payload,
        root=root,
        category_name="model-bindings",
        artifact_kind="model-binding",
        artifact_prefix="aqse-four-sensor-model-binding",
        content_name="model-binding.json",
        protocol_freeze_path=protocol_freeze_path,
        dataset_path=dataset_path,
    )


def write_test_authorization(
    payload: Any,
    *,
    protocol_freeze_path: Path,
    dataset_path: Path,
    model_binding_path: Path,
    root: Path | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    payload_value = _normalise(payload)
    if not isinstance(payload_value, dict) or payload_value.get("authorized") is not True:
        raise ValueError("final TEST access requires explicit authorization")
    return _write_bound_payload(
        payload_value,
        root=root,
        category_name="authorizations",
        artifact_kind="test-authorization",
        artifact_prefix="aqse-four-sensor-test-authorization",
        content_name="authorization.json",
        protocol_freeze_path=protocol_freeze_path,
        dataset_path=dataset_path,
        model_binding_path=model_binding_path,
    )


write_authorization = write_test_authorization


def _verify_bound_payload(
    path: Path,
    *,
    artifact_kind: str,
    content_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _verify_content_artifact(
        path,
        artifact_kind=artifact_kind,
        expected_files={content_name},
    )
    value = _read_json(path.resolve() / content_name)
    if (
        not isinstance(value, dict)
        or value.get("schema_version")
        != f"aqse.four-sensor-study.{artifact_kind}.v1"
        or not isinstance(value.get("payload"), dict)
        or not isinstance(value.get("bindings"), dict)
        or value.get("bindings") != manifest["bindings"]
        or value.get("payload_digest")
        != hashlib.sha256(canonical_json_bytes(value["payload"])).hexdigest()
    ):
        raise ValueError(f"four-sensor {artifact_kind} payload is invalid")
    return manifest, value


def verify_model_binding(path: Path) -> dict[str, Any]:
    return _verify_bound_payload(
        path,
        artifact_kind="model-binding",
        content_name="model-binding.json",
    )[0]


def load_model_binding(path: Path) -> dict[str, Any]:
    """Return the caller-owned frozen model descriptor, never fitting a model."""

    _manifest_value, wrapper = _verify_bound_payload(
        path,
        artifact_kind="model-binding",
        content_name="model-binding.json",
    )
    return wrapper["payload"]


def verify_test_authorization(path: Path) -> dict[str, Any]:
    manifest, value = _verify_bound_payload(
        path,
        artifact_kind="test-authorization",
        content_name="authorization.json",
    )
    if value["payload"].get("authorized") is not True:
        raise ValueError("four-sensor final TEST authorization is not affirmative")
    return manifest


def load_test_authorization(path: Path) -> dict[str, Any]:
    _manifest_value, wrapper = _verify_bound_payload(
        path,
        artifact_kind="test-authorization",
        content_name="authorization.json",
    )
    if wrapper["payload"].get("authorized") is not True:
        raise ValueError("four-sensor final TEST authorization is not affirmative")
    return wrapper["payload"]


verify_authorization = verify_test_authorization


def _validate_model_dataset_binding(
    model: Mapping[str, Any],
    protocol: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> None:
    if (
        model["bindings"].get("protocol_freeze_artifact_id") != protocol["artifact_id"]
        or model["bindings"].get("protocol_freeze_content_digest")
        != protocol["content_digest"]
        or model["bindings"].get("dataset_id") != dataset["artifact_id"]
        or model["bindings"].get("dataset_content_digest") != dataset["content_digest"]
    ):
        raise PermissionError("model binding does not match protocol and final TEST dataset")


def _access_context(
    dataset_path: Path,
    *,
    protocol_freeze_path: Path,
    model_binding_path: Path,
    authorization_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset = verify_final_test_dataset(dataset_path)
    protocol = verify_protocol_freeze(protocol_freeze_path)
    model = verify_model_binding(model_binding_path)
    authorization = verify_test_authorization(authorization_path)
    _validate_model_dataset_binding(model, protocol, dataset)
    expected = {
        "protocol_freeze_artifact_id": protocol["artifact_id"],
        "protocol_freeze_content_digest": protocol["content_digest"],
        "dataset_id": dataset["artifact_id"],
        "dataset_content_digest": dataset["content_digest"],
        "model_binding_id": model["artifact_id"],
        "model_binding_content_digest": model["content_digest"],
    }
    if authorization["bindings"] != expected:
        raise PermissionError("TEST authorization does not match the bound scientific set")
    return dataset, protocol, model, authorization


def _decode_ledger(
    payload: bytes,
    *,
    dataset_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    rows = [line for line in payload.splitlines() if line]
    if not 1 <= len(rows) <= 4:
        raise ValueError("four-sensor TEST ledger has an invalid event count")
    try:
        entries = tuple(json.loads(line) for line in rows)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("four-sensor TEST ledger contains invalid JSON") from exc
    expected_events = (
        "sealed",
        "observations_opened",
        "labels_opened",
        "evaluation_published",
    )
    previous: str | None = None
    access_identity: tuple[Any, ...] | None = None
    for sequence, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("four-sensor TEST ledger entry must be an object")
        body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        if (
            entry.get("schema_version") != LEDGER_SCHEMA
            or entry.get("dataset_id") != dataset_manifest["artifact_id"]
            or entry.get("dataset_content_digest") != dataset_manifest["content_digest"]
            or entry.get("sequence") != sequence
            or entry.get("event") != expected_events[sequence]
            or entry.get("previous_entry_sha256") != previous
            or entry.get("entry_sha256")
            != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        ):
            raise ValueError("four-sensor TEST ledger identity or hash chain is invalid")
        if (
            entry.get("protocol_freeze_id")
            != dataset_manifest["bindings"].get("protocol_freeze_artifact_id")
            or entry.get("protocol_freeze_content_digest")
            != dataset_manifest["bindings"].get("protocol_freeze_content_digest")
        ):
            raise ValueError("four-sensor TEST ledger protocol binding is invalid")
        if sequence == 0:
            if any(
                entry.get(key) is not None
                for key in (
                    "model_binding_id",
                    "model_binding_content_digest",
                    "authorization_id",
                    "authorization_content_digest",
                    "evaluation_id",
                    "evaluation_content_digest",
                )
            ):
                raise ValueError("four-sensor TEST ledger genesis is not sealed")
        else:
            identity = (
                entry.get("model_binding_id"),
                entry.get("model_binding_content_digest"),
                entry.get("authorization_id"),
                entry.get("authorization_content_digest"),
            )
            if any(item is None for item in identity):
                raise ValueError("four-sensor TEST ledger access identity is incomplete")
            if access_identity is None:
                access_identity = identity
            elif identity != access_identity:
                raise ValueError("four-sensor TEST ledger access identity changed")
            has_evaluation = sequence == 3
            if has_evaluation != (entry.get("evaluation_id") is not None):
                raise ValueError("four-sensor TEST ledger evaluation order is invalid")
            if has_evaluation != (entry.get("evaluation_content_digest") is not None):
                raise ValueError("four-sensor TEST ledger evaluation digest is invalid")
        previous = entry["entry_sha256"]
    return entries


def read_test_ledger(path: Path) -> TestLedgerSnapshot:
    manifest = _verify_content_artifact(
        path,
        artifact_kind="final-test-dataset",
        expected_files=FINAL_DATASET_FILES,
        extra_files={"test-access-ledger.jsonl"},
    )
    ledger_path = path.resolve() / "test-access-ledger.jsonl"
    payload = _read_bounded(ledger_path)
    return TestLedgerSnapshot(
        entries=_decode_ledger(payload, dataset_manifest=manifest),
        file_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _same_access_identity(
    entry: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> bool:
    return (
        entry.get("model_binding_id") == model["artifact_id"]
        and entry.get("model_binding_content_digest") == model["content_digest"]
        and entry.get("authorization_id") == authorization["artifact_id"]
        and entry.get("authorization_content_digest") == authorization["content_digest"]
    )


def _advance_ledger(
    dataset_path: Path,
    *,
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    model: Mapping[str, Any],
    authorization: Mapping[str, Any],
    sequence: int,
    event: str,
    evaluation: Mapping[str, Any] | None = None,
    allow_existing: bool = False,
) -> TestLedgerSnapshot:
    resolved = dataset_path.resolve()
    study_root = resolved.parents[1]
    locks = _category(study_root, "ledger-locks")
    lock_path = locks / f"{dataset['artifact_id']}.lock"
    with lock_path.open("a+b") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        snapshot = read_test_ledger(resolved)
        if len(snapshot.entries) > sequence:
            existing = snapshot.entries[sequence]
            if existing.get("event") != event or not _same_access_identity(
                existing,
                model=model,
                authorization=authorization,
            ):
                raise PermissionError("TEST ledger already advanced under another authorization")
            if evaluation is not None and (
                existing.get("evaluation_id") != evaluation["artifact_id"]
                or existing.get("evaluation_content_digest") != evaluation["content_digest"]
            ):
                raise PermissionError("TEST evaluation was already published with another identity")
            if not allow_existing:
                raise PermissionError(
                    "this semantic TEST channel was already opened; repeated decoding is refused"
                )
            return snapshot
        if len(snapshot.entries) != sequence:
            raise PermissionError("TEST ledger events must follow the authorized order")
        entry = _ledger_entry(
            dataset_manifest=dataset,
            sequence=sequence,
            event=event,
            protocol_freeze=protocol,
            model_binding=model,
            authorization=authorization,
            evaluation=evaluation,
            previous_entry_sha256=snapshot.entries[-1]["entry_sha256"],
        )
        entries = (*snapshot.entries, entry)
        payload = b"".join(_ledger_payload(item) for item in entries)
        ledger_path = resolved / "test-access-ledger.jsonl"
        temporary = resolved / f".test-access-ledger.tmp-{uuid4().hex}"
        try:
            _write_bytes(temporary, payload, mode=0o600)
            _decode_ledger(payload, dataset_manifest=dataset)
            os.replace(temporary, ledger_path)
            _fsync_directory(resolved)
        finally:
            if temporary.exists():
                temporary.unlink()
        return read_test_ledger(resolved)


def _decode_channel(path: Path, filename: str, model_name: str) -> tuple[Any, ...]:
    value = _read_json(path.resolve() / filename)
    if not isinstance(value, list) or len(value) != FINAL_TEST_EPISODE_COUNT:
        raise ValueError("final TEST semantic channel has an invalid row count")
    try:
        from app.four_sensor_study import models as study_models

        model = getattr(study_models, model_name)
    except (ImportError, AttributeError):
        return tuple(value)
    return tuple(model.model_validate(item) for item in value)


def load_test_observations(
    dataset_path: Path,
    *,
    protocol_freeze_path: Path,
    model_binding_path: Path,
    authorization_path: Path,
) -> tuple[Any, ...]:
    dataset, protocol, model, authorization = _access_context(
        dataset_path,
        protocol_freeze_path=protocol_freeze_path,
        model_binding_path=model_binding_path,
        authorization_path=authorization_path,
    )
    _advance_ledger(
        dataset_path,
        dataset=dataset,
        protocol=protocol,
        model=model,
        authorization=authorization,
        sequence=1,
        event="observations_opened",
    )
    return _decode_channel(dataset_path, "observations.json", "FourSensorStudyObservation")


def load_test_labels(
    dataset_path: Path,
    *,
    protocol_freeze_path: Path,
    model_binding_path: Path,
    authorization_path: Path,
) -> tuple[Any, ...]:
    dataset, protocol, model, authorization = _access_context(
        dataset_path,
        protocol_freeze_path=protocol_freeze_path,
        model_binding_path=model_binding_path,
        authorization_path=authorization_path,
    )
    _advance_ledger(
        dataset_path,
        dataset=dataset,
        protocol=protocol,
        model=model,
        authorization=authorization,
        sequence=2,
        event="labels_opened",
    )
    return _decode_channel(dataset_path, "labels.json", "FourSensorStudyLabel")


def write_final_evaluation(
    payload: Any,
    *,
    protocol_freeze_path: Path,
    dataset_path: Path,
    model_binding_path: Path,
    authorization_path: Path,
    root: Path | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    dataset, protocol, model, authorization = _access_context(
        dataset_path,
        protocol_freeze_path=protocol_freeze_path,
        model_binding_path=model_binding_path,
        authorization_path=authorization_path,
    )
    entries = read_test_ledger(dataset_path).entries
    if len(entries) < 3 or not _same_access_identity(
        entries[2],
        model=model,
        authorization=authorization,
    ):
        raise PermissionError("final evaluation requires ordered observation and label access")
    return _write_bound_payload(
        payload,
        root=root,
        category_name="final-evaluations",
        artifact_kind="final-evaluation",
        artifact_prefix="aqse-four-sensor-final-evaluation",
        content_name="final-evaluation.json",
        protocol_freeze_path=protocol_freeze_path,
        dataset_path=dataset_path,
        model_binding_path=model_binding_path,
        authorization_path=authorization_path,
    )


def verify_final_evaluation(path: Path) -> dict[str, Any]:
    return _verify_bound_payload(
        path,
        artifact_kind="final-evaluation",
        content_name="final-evaluation.json",
    )[0]


def load_final_evaluation(path: Path) -> dict[str, Any]:
    _manifest_value, wrapper = _verify_bound_payload(
        path,
        artifact_kind="final-evaluation",
        content_name="final-evaluation.json",
    )
    return wrapper["payload"]


def mark_evaluation_published(
    dataset_path: Path,
    *,
    evaluation_path: Path,
    protocol_freeze_path: Path,
    model_binding_path: Path,
    authorization_path: Path,
) -> TestLedgerSnapshot:
    dataset, protocol, model, authorization = _access_context(
        dataset_path,
        protocol_freeze_path=protocol_freeze_path,
        model_binding_path=model_binding_path,
        authorization_path=authorization_path,
    )
    evaluation = verify_final_evaluation(evaluation_path)
    expected = {
        "protocol_freeze_artifact_id": protocol["artifact_id"],
        "protocol_freeze_content_digest": protocol["content_digest"],
        "dataset_id": dataset["artifact_id"],
        "dataset_content_digest": dataset["content_digest"],
        "model_binding_id": model["artifact_id"],
        "model_binding_content_digest": model["content_digest"],
        "authorization_id": authorization["artifact_id"],
        "authorization_content_digest": authorization["content_digest"],
    }
    if evaluation["bindings"] != expected:
        raise PermissionError("final evaluation does not match the authorized scientific set")
    return _advance_ledger(
        dataset_path,
        dataset=dataset,
        protocol=protocol,
        model=model,
        authorization=authorization,
        sequence=3,
        event="evaluation_published",
        evaluation=evaluation,
        allow_existing=True,
    )
