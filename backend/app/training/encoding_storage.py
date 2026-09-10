from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.quantum.user_pipeline.tqk8 import AngleScaler
from app.training.canonical import canonical_json_bytes, file_sha256
from app.training.encoding import PhaseDirectEncoder, validate_encoder_artifact
from app.training.encoding_models import (
    EncoderExecutionMetadata,
    EncoderScientificArtifact,
    QuantumBankArtifact,
)
from app.training.software_provenance import capture_software_provenance
from app.training.storage import artifact_limit_bytes, artifact_root

MAX_ENCODING_JSON_BYTES = 4_194_304


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"encoding artifact file is missing or unsafe: {path.name}")
    if path.stat().st_size > MAX_ENCODING_JSON_BYTES:
        raise ValueError(f"encoding artifact JSON exceeds the safe limit: {path.name}")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid encoding artifact JSON: {path.name}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _root_usage(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )


def _safe_target(root: Path, category: str, artifact_id: str) -> Path:
    target = root / category / artifact_id
    resolved_root = root.resolve()
    resolved = target.resolve(strict=False)
    if resolved_root not in resolved.parents:
        raise ValueError("encoding artifact path escapes the configured root")
    return target


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_encoder_artifact(
    encoder: PhaseDirectEncoder,
    *,
    root: Path | None = None,
) -> tuple[Path, EncoderExecutionMetadata]:
    validate_encoder_artifact(encoder.artifact)
    resolved_root = (root or artifact_root()).resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    if _root_usage(resolved_root) >= artifact_limit_bytes():
        raise OSError("AQSE artifact root has reached its configured byte limit")
    final = _safe_target(resolved_root, "encoders", encoder.artifact.artifact_id)
    if final.exists():
        raise FileExistsError(f"immutable encoder artifact already exists: {final}")
    staging = final.with_name(f".{final.name}.tmp-{uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        _write_json(staging / "encoder.json", encoder.artifact.model_dump(mode="json"))
        provenance = capture_software_provenance()
        execution = EncoderExecutionMetadata(
            artifact_id=encoder.artifact.artifact_id,
            artifact_path=str(final),
            created_at_utc=_utc_now(),
            repository_base_sha=provenance.repository_base_sha,
            repository_dirty=provenance.repository_dirty,
            runtime_versions=provenance.runtime_versions,
        )
        _write_json(staging / "execution.json", execution.model_dump(mode="json"))
        manifest = {
            "schema_version": "aqse.training-encoder-manifest.v1",
            "artifact_id": encoder.artifact.artifact_id,
            "content_digest": encoder.artifact.content_digest,
            "files": {
                "encoder.json": {
                    "byte_count": (staging / "encoder.json").stat().st_size,
                    "sha256": file_sha256(staging / "encoder.json"),
                },
                "execution.json": {
                    "byte_count": (staging / "execution.json").stat().st_size,
                    "sha256": file_sha256(staging / "execution.json"),
                },
            },
        }
        _write_json(staging / "manifest.json", manifest)
        load_encoder_artifact(
            staging,
            expected_artifact_id=encoder.artifact.artifact_id,
            expected_dataset_id=encoder.artifact.source_dataset_id,
            expected_dataset_digest=encoder.artifact.source_dataset_digest,
            expected_profile_fingerprint=encoder.artifact.feature_profile_fingerprint,
            allow_staging_name=True,
        )
        if _root_usage(resolved_root) > artifact_limit_bytes():
            raise OSError("AQSE encoder write would exceed the configured byte limit")
        for item in staging.iterdir():
            item.chmod(0o444)
        os.replace(staging, final)
        return final, execution
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_encoder_artifact(
    path: Path,
    *,
    expected_artifact_id: str,
    expected_dataset_id: str,
    expected_dataset_digest: str,
    expected_profile_fingerprint: str,
    allow_staging_name: bool = False,
) -> PhaseDirectEncoder:
    resolved = path.resolve()
    manifest = _read_json(resolved / "manifest.json")
    if manifest.get("schema_version") != "aqse.training-encoder-manifest.v1":
        raise ValueError("unsupported encoder manifest schema")
    if manifest.get("artifact_id") != expected_artifact_id:
        raise ValueError("encoder manifest artifact identity mismatch")
    if not allow_staging_name and resolved.name != expected_artifact_id:
        raise ValueError("encoder directory and manifest identities differ")
    expected_files = {"encoder.json", "execution.json"}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("encoder manifest file allowlist is invalid")
    actual_files = {item.name for item in resolved.iterdir() if item.is_file()}
    if actual_files != expected_files | {"manifest.json"}:
        raise ValueError("encoder artifact contains unexpected files")
    for name in expected_files:
        target = resolved / name
        record = manifest["files"][name]
        if target.is_symlink() or target.stat().st_size != record.get("byte_count"):
            raise ValueError("encoder artifact file size or path is invalid")
        if file_sha256(target) != record.get("sha256"):
            raise ValueError("encoder artifact file digest is invalid")
    artifact = EncoderScientificArtifact.model_validate(_read_json(resolved / "encoder.json"))
    execution = EncoderExecutionMetadata.model_validate(
        _read_json(resolved / "execution.json")
    )
    validate_encoder_artifact(artifact)
    if artifact.artifact_id != expected_artifact_id:
        raise ValueError("unexpected encoder artifact ID")
    if artifact.content_digest != manifest.get("content_digest"):
        raise ValueError("encoder content digest differs from the manifest")
    if artifact.source_dataset_id != expected_dataset_id:
        raise ValueError("encoder dataset identity is incompatible")
    if artifact.source_dataset_digest != expected_dataset_digest:
        raise ValueError("encoder dataset digest is incompatible")
    if artifact.feature_profile_fingerprint != expected_profile_fingerprint:
        raise ValueError("encoder feature-profile identity is incompatible")
    if execution.artifact_id != expected_artifact_id:
        raise ValueError("encoder execution metadata identity is incompatible")
    mean = np.asarray(artifact.fitted_mean, dtype=np.float64)
    scale = np.asarray(artifact.fitted_scale, dtype=np.float64)
    mean.setflags(write=False)
    scale.setflags(write=False)
    return PhaseDirectEncoder(artifact=artifact, scaler=AngleScaler(mean=mean, scale=scale))


def write_bank_artifact(
    bank: QuantumBankArtifact,
    *,
    root: Path | None = None,
) -> Path:
    from app.training.banks import validate_bank_artifact

    validate_bank_artifact(bank)
    resolved_root = (root or artifact_root()).resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    if _root_usage(resolved_root) >= artifact_limit_bytes():
        raise OSError("AQSE artifact root has reached its configured byte limit")
    final = _safe_target(resolved_root, "banks", bank.artifact_id)
    if final.exists():
        raise FileExistsError(f"immutable quantum-bank artifact already exists: {final}")
    staging = final.with_name(f".{final.name}.tmp-{uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        _write_json(staging / "bank.json", bank.model_dump(mode="json"))
        manifest = {
            "schema_version": "aqse.quantum-bank-manifest.v1",
            "artifact_id": bank.artifact_id,
            "content_digest": bank.content_digest,
            "files": {
                "bank.json": {
                    "byte_count": (staging / "bank.json").stat().st_size,
                    "sha256": file_sha256(staging / "bank.json"),
                }
            },
        }
        _write_json(staging / "manifest.json", manifest)
        load_bank_artifact(
            staging,
            expected_artifact_id=bank.artifact_id,
            expected_dataset_id=bank.source_dataset_id,
            expected_dataset_digest=bank.source_dataset_digest,
            expected_encoder_artifact_id=bank.encoder_artifact_id,
            allow_staging_name=True,
        )
        if _root_usage(resolved_root) > artifact_limit_bytes():
            raise OSError("AQSE quantum-bank write would exceed the configured byte limit")
        for item in staging.iterdir():
            item.chmod(0o444)
        os.replace(staging, final)
        return final
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_bank_artifact(
    path: Path,
    *,
    expected_artifact_id: str,
    expected_dataset_id: str,
    expected_dataset_digest: str,
    expected_encoder_artifact_id: str,
    allow_staging_name: bool = False,
) -> QuantumBankArtifact:
    from app.training.banks import validate_bank_artifact

    resolved = path.resolve()
    manifest = _read_json(resolved / "manifest.json")
    if manifest.get("schema_version") != "aqse.quantum-bank-manifest.v1":
        raise ValueError("unsupported quantum-bank manifest")
    if manifest.get("artifact_id") != expected_artifact_id:
        raise ValueError("quantum-bank manifest identity mismatch")
    if not allow_staging_name and resolved.name != expected_artifact_id:
        raise ValueError("quantum-bank directory and manifest identities differ")
    if set(manifest.get("files", {})) != {"bank.json"}:
        raise ValueError("quantum-bank file allowlist is invalid")
    if {item.name for item in resolved.iterdir() if item.is_file()} != {
        "bank.json",
        "manifest.json",
    }:
        raise ValueError("quantum-bank artifact contains unexpected files")
    target = resolved / "bank.json"
    record = manifest["files"]["bank.json"]
    if target.is_symlink() or target.stat().st_size != record.get("byte_count"):
        raise ValueError("quantum-bank file size or path is invalid")
    if file_sha256(target) != record.get("sha256"):
        raise ValueError("quantum-bank file digest is invalid")
    bank = QuantumBankArtifact.model_validate(_read_json(target))
    validate_bank_artifact(bank)
    if bank.artifact_id != expected_artifact_id:
        raise ValueError("unexpected quantum-bank artifact ID")
    if bank.content_digest != manifest.get("content_digest"):
        raise ValueError("quantum-bank content digest differs from manifest")
    if bank.source_dataset_id != expected_dataset_id:
        raise ValueError("quantum-bank dataset identity is incompatible")
    if bank.source_dataset_digest != expected_dataset_digest:
        raise ValueError("quantum-bank dataset digest is incompatible")
    if bank.encoder_artifact_id != expected_encoder_artifact_id:
        raise ValueError("quantum-bank encoder identity is incompatible")
    return bank
