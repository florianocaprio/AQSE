from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.training.canonical import canonical_json_bytes, file_sha256
from app.training.run_models import TrainingRunArtifact, TrainingRunExecutionMetadata
from app.training.runner import validate_training_run_artifact
from app.training.software_provenance import capture_software_provenance
from app.training.storage import artifact_limit_bytes, artifact_root

MAX_RUN_JSON_BYTES = 8_388_608


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"training-run artifact file is missing or unsafe: {path.name}")
    if path.stat().st_size > MAX_RUN_JSON_BYTES:
        raise ValueError(f"training-run JSON exceeds the safe limit: {path.name}")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid training-run JSON: {path.name}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _root_usage(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _target(root: Path, run_id: str) -> Path:
    target = root / "training-runs" / run_id
    resolved_root = root.resolve()
    resolved = target.resolve(strict=False)
    if resolved_root not in resolved.parents:
        raise ValueError("training-run artifact path escapes the configured root")
    return target


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_training_run_artifact(
    artifact: TrainingRunArtifact,
    *,
    root: Path | None = None,
) -> tuple[Path, TrainingRunExecutionMetadata]:
    validate_training_run_artifact(artifact)
    resolved_root = (root or artifact_root()).resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    if _root_usage(resolved_root) >= artifact_limit_bytes():
        raise OSError("AQSE artifact root has reached its configured byte limit")
    final = _target(resolved_root, artifact.run_id)
    if final.exists():
        raise FileExistsError(f"immutable training-run artifact already exists: {final}")
    staging = final.with_name(f".{final.name}.tmp-{uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        _write_json(staging / "run.json", artifact.model_dump(mode="json"))
        provenance = capture_software_provenance()
        execution = TrainingRunExecutionMetadata(
            run_id=artifact.run_id,
            artifact_path=str(final),
            created_at_utc=_utc_now(),
            repository_base_sha=provenance.repository_base_sha,
            repository_dirty=provenance.repository_dirty,
            runtime_versions=provenance.runtime_versions,
        )
        _write_json(staging / "execution.json", execution.model_dump(mode="json"))
        files = {
            name: {
                "byte_count": (staging / name).stat().st_size,
                "sha256": file_sha256(staging / name),
            }
            for name in ("run.json", "execution.json")
        }
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": "aqse.qng-training-run-manifest.v1",
                "run_id": artifact.run_id,
                "content_digest": artifact.content_digest,
                "files": files,
            },
        )
        load_training_run_artifact(
            staging,
            expected_run_id=artifact.run_id,
            expected_training_input_fingerprint=(
                artifact.training_input.fingerprint_id
            ),
            allow_staging_name=True,
        )
        if _root_usage(resolved_root) > artifact_limit_bytes():
            raise OSError("AQSE training-run write would exceed the configured byte limit")
        for path in staging.iterdir():
            path.chmod(0o444)
        os.replace(staging, final)
        return final, execution
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_training_run_artifact(
    path: Path,
    *,
    expected_run_id: str,
    expected_training_input_fingerprint: str,
    allow_staging_name: bool = False,
) -> TrainingRunArtifact:
    resolved = path.resolve()
    manifest = _read_json(resolved / "manifest.json")
    if manifest.get("schema_version") != "aqse.qng-training-run-manifest.v1":
        raise ValueError("unsupported training-run manifest")
    if manifest.get("run_id") != expected_run_id:
        raise ValueError("training-run manifest identity mismatch")
    if not allow_staging_name and resolved.name != expected_run_id:
        raise ValueError("training-run directory and manifest identities differ")
    expected_files = {"run.json", "execution.json"}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("training-run manifest file allowlist is invalid")
    actual_entries = {item.name for item in resolved.iterdir()}
    if actual_entries != expected_files | {"manifest.json"}:
        raise ValueError("training-run artifact contains unexpected entries")
    for name in expected_files:
        target = resolved / name
        record = manifest["files"][name]
        if target.is_symlink() or target.stat().st_size != record.get("byte_count"):
            raise ValueError("training-run artifact file size or path is invalid")
        if file_sha256(target) != record.get("sha256"):
            raise ValueError("training-run artifact file digest is invalid")
    artifact = TrainingRunArtifact.model_validate(_read_json(resolved / "run.json"))
    execution = TrainingRunExecutionMetadata.model_validate(
        _read_json(resolved / "execution.json")
    )
    validate_training_run_artifact(artifact)
    if artifact.run_id != expected_run_id or execution.run_id != expected_run_id:
        raise ValueError("training-run artifact/execution identity mismatch")
    if artifact.content_digest != manifest.get("content_digest"):
        raise ValueError("training-run content digest differs from manifest")
    if artifact.training_input.fingerprint_id != expected_training_input_fingerprint:
        raise ValueError("training-run input fingerprint is incompatible")
    return artifact
