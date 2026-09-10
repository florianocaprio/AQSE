from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.training.canonical import canonical_json_bytes, file_sha256
from app.training.storage import artifact_limit_bytes, artifact_root
from app.training.trajectory import (
    TrajectoryAuditMapping,
    validate_trajectory_audit_mapping,
)

MAX_TRAJECTORY_AUDIT_BYTES = 8_388_608


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError("trajectory audit file is missing or unsafe")
    if path.stat().st_size > MAX_TRAJECTORY_AUDIT_BYTES:
        raise ValueError("trajectory audit file exceeds the bounded size")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("trajectory audit JSON is invalid") from exc


def _root_usage(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _target(root: Path, trajectory_id: str) -> Path:
    target = root / "trajectory-audits" / trajectory_id
    resolved_root = root.resolve()
    resolved = target.resolve(strict=False)
    if resolved_root not in resolved.parents:
        raise ValueError("trajectory audit path escapes the artifact root")
    return target


def load_trajectory_audit_mapping(
    path: Path,
    *,
    expected_trajectory_id: str,
    allow_staging_name: bool = False,
) -> TrajectoryAuditMapping:
    resolved = path.resolve()
    if not allow_staging_name and resolved.name != expected_trajectory_id:
        raise ValueError("trajectory audit directory identity is invalid")
    if {item.name for item in resolved.iterdir()} != {"mapping.json", "manifest.json"}:
        raise ValueError("trajectory audit contains unexpected entries")
    manifest = _read_json(resolved / "manifest.json")
    if manifest.get("schema_version") != "aqse.qng-trajectory-audit-manifest.v1":
        raise ValueError("trajectory audit manifest schema is invalid")
    if manifest.get("trajectory_id") != expected_trajectory_id:
        raise ValueError("trajectory audit manifest identity is invalid")
    mapping_path = resolved / "mapping.json"
    if mapping_path.stat().st_size != manifest.get("mapping_byte_count"):
        raise ValueError("trajectory audit mapping size is invalid")
    if file_sha256(mapping_path) != manifest.get("mapping_sha256"):
        raise ValueError("trajectory audit mapping file digest is invalid")
    mapping = TrajectoryAuditMapping.model_validate(_read_json(mapping_path))
    validate_trajectory_audit_mapping(mapping)
    if mapping.trajectory.trajectory_id != expected_trajectory_id:
        raise ValueError("trajectory audit content identity is invalid")
    if mapping.mapping_digest != manifest.get("mapping_digest"):
        raise ValueError("trajectory audit mapping/manifest digests differ")
    return mapping


def write_trajectory_audit_mapping(
    mapping: TrajectoryAuditMapping,
    *,
    root: Path | None = None,
) -> Path:
    validate_trajectory_audit_mapping(mapping)
    resolved_root = (root or artifact_root()).resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    if _root_usage(resolved_root) >= artifact_limit_bytes():
        raise OSError("AQSE artifact root has reached its configured byte limit")
    final = _target(resolved_root, mapping.trajectory.trajectory_id)
    if final.exists():
        existing = load_trajectory_audit_mapping(
            final,
            expected_trajectory_id=mapping.trajectory.trajectory_id,
        )
        if existing != mapping:
            raise FileExistsError("immutable trajectory audit already exists with other content")
        return final
    staging = final.with_name(f".{final.name}.tmp-{uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        mapping_path = staging / "mapping.json"
        mapping_path.write_bytes(
            canonical_json_bytes(mapping.model_dump(mode="json")) + b"\n"
        )
        manifest = {
            "schema_version": "aqse.qng-trajectory-audit-manifest.v1",
            "trajectory_id": mapping.trajectory.trajectory_id,
            "mapping_digest": mapping.mapping_digest,
            "mapping_byte_count": mapping_path.stat().st_size,
            "mapping_sha256": file_sha256(mapping_path),
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        load_trajectory_audit_mapping(
            staging,
            expected_trajectory_id=mapping.trajectory.trajectory_id,
            allow_staging_name=True,
        )
        if _root_usage(resolved_root) > artifact_limit_bytes():
            raise OSError("trajectory audit write would exceed the artifact limit")
        for path in staging.iterdir():
            path.chmod(0o444)
        os.replace(staging, final)
        return final
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
