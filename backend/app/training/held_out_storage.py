from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from app.training.canonical import canonical_json_bytes, file_sha256
from app.training.held_out import (
    validate_final_held_out,
    validate_g5_authorization,
    validate_test_bank,
)
from app.training.held_out_models import (
    ArtifactExecutionMetadata,
    FinalHeldOutArtifact,
    FinalHeldOutExecutionMetadata,
    G5AuthorizationArtifact,
    HeldOutMethodRuntime,
    TestBankArtifact,
)
from app.training.models import DatasetManifest, LegacyDatasetManifest, TestAccessLedgerEntry
from app.training.software_provenance import capture_software_provenance
from app.training.storage import artifact_limit_bytes, artifact_root, verify_archive_opaque

MAX_HELD_OUT_JSON_BYTES = 33_554_432
ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class TestLedgerSnapshot:
    entries: tuple[TestAccessLedgerEntry, ...]
    file_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError("held-out artifact file is missing or unsafe")
    if path.stat().st_size > MAX_HELD_OUT_JSON_BYTES:
        raise ValueError("held-out artifact file exceeds the bounded size")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("held-out artifact JSON is invalid") from exc


def _root_usage(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _target(root: Path, category: str, artifact_id: str) -> Path:
    target = root / category / artifact_id
    if root.resolve() not in target.resolve(strict=False).parents:
        raise ValueError("held-out artifact path escapes the configured root")
    return target


def _verify_files(path: Path, manifest: dict[str, Any], expected: set[str]) -> None:
    if set(manifest.get("files", {})) != expected:
        raise ValueError("held-out artifact manifest allowlist is invalid")
    if {item.name for item in path.iterdir()} != expected | {"manifest.json"}:
        raise ValueError("held-out artifact contains unexpected entries")
    for name in expected:
        target = path / name
        record = manifest["files"][name]
        if target.is_symlink() or target.stat().st_size != record.get("byte_count"):
            raise ValueError("held-out artifact file size or path is invalid")
        if file_sha256(target) != record.get("sha256"):
            raise ValueError("held-out artifact file digest is invalid")


def _write_artifact(
    *,
    root: Path,
    category: str,
    artifact_id: str,
    content_digest: str,
    manifest_schema: str,
    content_name: str,
    content: ModelT,
    execution: BaseModel,
    load_existing: Callable[[Path], ModelT],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if _root_usage(root) >= artifact_limit_bytes():
        raise OSError("AQSE artifact root has reached its configured byte limit")
    final = _target(root, category, artifact_id)
    if final.exists():
        if load_existing(final) != content:
            raise FileExistsError("immutable held-out artifact has different content")
        return final
    staging = final.with_name(f".{final.name}.tmp-{uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        payloads = {content_name: content, "execution.json": execution}
        for name, value in payloads.items():
            (staging / name).write_bytes(
                canonical_json_bytes(value.model_dump(mode="json")) + b"\n"
            )
        files = {
            name: {
                "byte_count": (staging / name).stat().st_size,
                "sha256": file_sha256(staging / name),
            }
            for name in payloads
        }
        manifest = {
            "schema_version": manifest_schema,
            "artifact_id": artifact_id,
            "content_digest": content_digest,
            "files": files,
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        _verify_files(staging, manifest, set(payloads))
        if type(content).model_validate(_read_json(staging / content_name)) != content:
            raise ValueError("staged held-out artifact differs from validated content")
        if _root_usage(root) > artifact_limit_bytes():
            raise OSError("held-out artifact write would exceed the configured limit")
        for item in staging.iterdir():
            item.chmod(0o444)
        os.replace(staging, final)
        return final
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def read_test_ledger_opaque(dataset_path: Path) -> TestLedgerSnapshot:
    resolved = dataset_path.resolve()
    manifest = verify_archive_opaque(resolved)
    if isinstance(manifest, LegacyDatasetManifest):
        raise ValueError("held-out evaluation requires the canonical archive-v2 dataset")
    ledger_path = resolved / "test-access-ledger.jsonl"
    if (
        ledger_path.is_symlink()
        or not ledger_path.is_file()
        or ledger_path.stat().st_size > MAX_HELD_OUT_JSON_BYTES
    ):
        raise ValueError("TEST ledger is missing, unsafe or oversized")
    entries = tuple(
        TestAccessLedgerEntry.model_validate_json(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    _validate_ledger(entries, manifest)
    return TestLedgerSnapshot(entries=entries, file_sha256=file_sha256(ledger_path))


def _validate_ledger(
    entries: tuple[TestAccessLedgerEntry, ...],
    manifest: DatasetManifest,
) -> None:
    if not entries or entries[0].event != "sealed":
        raise ValueError("TEST ledger has no sealing genesis entry")
    previous: str | None = None
    for sequence, entry in enumerate(entries):
        if (
            entry.sequence != sequence
            or entry.dataset_id != manifest.dataset_id
            or entry.scientific_digest != manifest.scientific_digest
            or entry.previous_entry_sha256 != previous
        ):
            raise ValueError("TEST ledger identity or chain is invalid")
        payload = entry.model_dump(mode="json", exclude={"entry_sha256"})
        if entry.entry_sha256 != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
            raise ValueError("TEST ledger entry digest is invalid")
        if sequence and entry.event != "opened":
            raise ValueError("TEST ledger may contain only opened events after genesis")
        previous = entry.entry_sha256


def _artifact_execution(
    *,
    artifact_kind: str,
    artifact_id: str,
    artifact_path: Path,
) -> ArtifactExecutionMetadata:
    provenance = capture_software_provenance()
    return ArtifactExecutionMetadata(
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        artifact_path=str(artifact_path),
        created_at_utc=_utc_now(),
        repository_base_sha=provenance.repository_base_sha,
        repository_dirty=provenance.repository_dirty,
        runtime_versions=provenance.runtime_versions,
    )


def write_g5_authorization(
    authorization: G5AuthorizationArtifact,
    *,
    root: Path | None = None,
) -> tuple[Path, ArtifactExecutionMetadata]:
    validate_g5_authorization(authorization)
    resolved_root = (root or artifact_root()).resolve()
    final = _target(
        resolved_root,
        "g5-authorizations",
        authorization.authorization_id,
    )

    def load(path: Path) -> G5AuthorizationArtifact:
        return load_g5_authorization(
            path,
            expected_authorization_id=authorization.authorization_id,
        )[0]

    if final.exists():
        existing, execution = load_g5_authorization(
            final,
            expected_authorization_id=authorization.authorization_id,
        )
        if existing != authorization:
            raise FileExistsError("immutable G5 authorization has different content")
        return final, execution
    execution = _artifact_execution(
        artifact_kind="g5_authorization",
        artifact_id=authorization.authorization_id,
        artifact_path=final,
    )
    path = _write_artifact(
        root=resolved_root,
        category="g5-authorizations",
        artifact_id=authorization.authorization_id,
        content_digest=authorization.content_digest,
        manifest_schema="aqse.g5-authorization-manifest.v1",
        content_name="authorization.json",
        content=authorization,
        execution=execution,
        load_existing=load,
    )
    return path, execution


def load_g5_authorization(
    path: Path,
    *,
    expected_authorization_id: str,
) -> tuple[G5AuthorizationArtifact, ArtifactExecutionMetadata]:
    resolved = path.resolve()
    if resolved.name != expected_authorization_id:
        raise ValueError("G5 authorization directory identity is invalid")
    manifest = _read_json(resolved / "manifest.json")
    if manifest.get("schema_version") != "aqse.g5-authorization-manifest.v1":
        raise ValueError("G5 authorization manifest schema is invalid")
    if manifest.get("artifact_id") != expected_authorization_id:
        raise ValueError("G5 authorization manifest identity is invalid")
    _verify_files(resolved, manifest, {"authorization.json", "execution.json"})
    authorization = G5AuthorizationArtifact.model_validate(
        _read_json(resolved / "authorization.json")
    )
    execution = ArtifactExecutionMetadata.model_validate(
        _read_json(resolved / "execution.json")
    )
    validate_g5_authorization(authorization)
    if authorization.authorization_id != expected_authorization_id:
        raise ValueError("G5 authorization content identity is invalid")
    if authorization.content_digest != manifest.get("content_digest"):
        raise ValueError("G5 authorization content/manifest digests differ")
    if execution.artifact_id != expected_authorization_id:
        raise ValueError("G5 authorization execution identity is invalid")
    return authorization, execution


def write_test_bank(
    bank: TestBankArtifact,
    *,
    root: Path | None = None,
) -> tuple[Path, ArtifactExecutionMetadata]:
    validate_test_bank(bank)
    resolved_root = (root or artifact_root()).resolve()
    final = _target(resolved_root, "test-banks", bank.artifact_id)

    def load(path: Path) -> TestBankArtifact:
        return load_test_bank(path, expected_bank_id=bank.artifact_id)[0]

    if final.exists():
        raise FileExistsError("canonical TEST bank already exists; do not rerun TEST access")
    execution = _artifact_execution(
        artifact_kind="test_bank",
        artifact_id=bank.artifact_id,
        artifact_path=final,
    )
    path = _write_artifact(
        root=resolved_root,
        category="test-banks",
        artifact_id=bank.artifact_id,
        content_digest=bank.content_digest,
        manifest_schema="aqse.test-bank-manifest.v1",
        content_name="bank.json",
        content=bank,
        execution=execution,
        load_existing=load,
    )
    return path, execution


def load_test_bank(
    path: Path,
    *,
    expected_bank_id: str,
) -> tuple[TestBankArtifact, ArtifactExecutionMetadata]:
    resolved = path.resolve()
    if resolved.name != expected_bank_id:
        raise ValueError("TEST bank directory identity is invalid")
    manifest = _read_json(resolved / "manifest.json")
    if manifest.get("schema_version") != "aqse.test-bank-manifest.v1":
        raise ValueError("TEST bank manifest schema is invalid")
    if manifest.get("artifact_id") != expected_bank_id:
        raise ValueError("TEST bank manifest identity is invalid")
    _verify_files(resolved, manifest, {"bank.json", "execution.json"})
    bank = TestBankArtifact.model_validate(_read_json(resolved / "bank.json"))
    execution = ArtifactExecutionMetadata.model_validate(
        _read_json(resolved / "execution.json")
    )
    validate_test_bank(bank)
    if bank.artifact_id != expected_bank_id:
        raise ValueError("TEST bank content identity is invalid")
    if bank.content_digest != manifest.get("content_digest"):
        raise ValueError("TEST bank content/manifest digests differ")
    if execution.artifact_id != expected_bank_id:
        raise ValueError("TEST bank execution identity is invalid")
    return bank, execution


def write_final_held_out(
    artifact: FinalHeldOutArtifact,
    *,
    method_runtimes: tuple[HeldOutMethodRuntime, ...],
    total_wall_time_ms: float,
    root: Path | None = None,
) -> tuple[Path, FinalHeldOutExecutionMetadata]:
    validate_final_held_out(artifact)
    resolved_root = (root or artifact_root()).resolve()
    final = _target(resolved_root, "final-held-out-evaluations", artifact.artifact_id)
    if final.exists():
        raise FileExistsError("final held-out artifact already exists; do not rerun TEST")
    provenance = capture_software_provenance()
    execution = FinalHeldOutExecutionMetadata(
        artifact_id=artifact.artifact_id,
        artifact_path=str(final),
        created_at_utc=_utc_now(),
        repository_base_sha=provenance.repository_base_sha,
        repository_dirty=provenance.repository_dirty,
        runtime_versions=provenance.runtime_versions,
        total_wall_time_ms=total_wall_time_ms,
        method_runtimes=method_runtimes,
    )

    def load(path: Path) -> FinalHeldOutArtifact:
        return load_final_held_out(path, expected_artifact_id=artifact.artifact_id)[0]

    path = _write_artifact(
        root=resolved_root,
        category="final-held-out-evaluations",
        artifact_id=artifact.artifact_id,
        content_digest=artifact.content_digest,
        manifest_schema="aqse.final-held-out-manifest.v1",
        content_name="evaluation.json",
        content=artifact,
        execution=execution,
        load_existing=load,
    )
    return path, execution


def load_final_held_out(
    path: Path,
    *,
    expected_artifact_id: str,
) -> tuple[FinalHeldOutArtifact, FinalHeldOutExecutionMetadata]:
    resolved = path.resolve()
    if resolved.name != expected_artifact_id:
        raise ValueError("final held-out directory identity is invalid")
    manifest = _read_json(resolved / "manifest.json")
    if manifest.get("schema_version") != "aqse.final-held-out-manifest.v1":
        raise ValueError("final held-out manifest schema is invalid")
    if manifest.get("artifact_id") != expected_artifact_id:
        raise ValueError("final held-out manifest identity is invalid")
    _verify_files(resolved, manifest, {"evaluation.json", "execution.json"})
    artifact = FinalHeldOutArtifact.model_validate(_read_json(resolved / "evaluation.json"))
    execution = FinalHeldOutExecutionMetadata.model_validate(
        _read_json(resolved / "execution.json")
    )
    validate_final_held_out(artifact)
    if artifact.artifact_id != expected_artifact_id:
        raise ValueError("final held-out content identity is invalid")
    if artifact.content_digest != manifest.get("content_digest"):
        raise ValueError("final held-out content/manifest digests differ")
    if execution.artifact_id != expected_artifact_id:
        raise ValueError("final held-out execution identity is invalid")
    return artifact, execution
