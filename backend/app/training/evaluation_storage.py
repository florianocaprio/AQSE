from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from app.training.canonical import canonical_json_bytes, file_sha256
from app.training.evaluation import (
    validate_comparative_evaluation,
    validate_model_selection_freeze,
)
from app.training.evaluation_models import (
    ComparativeEvaluationArtifact,
    EvaluationExecutionMetadata,
    EvaluationProtocol,
    MethodRuntime,
    ModelSelectionFreeze,
    ProtocolExecutionMetadata,
    SelectionFreezeExecutionMetadata,
)
from app.training.evaluation_protocol import validate_evaluation_protocol
from app.training.software_provenance import capture_software_provenance
from app.training.storage import artifact_limit_bytes, artifact_root

MAX_EVALUATION_JSON_BYTES = 33_554_432
ModelT = TypeVar("ModelT", bound=BaseModel)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError("evaluation artifact file is missing or unsafe")
    if path.stat().st_size > MAX_EVALUATION_JSON_BYTES:
        raise ValueError("evaluation artifact file exceeds the bounded size")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("evaluation artifact JSON is invalid") from exc


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
        raise ValueError("evaluation artifact path escapes the configured root")
    return target


def _verify_files(path: Path, manifest: dict[str, Any], expected: set[str]) -> None:
    if set(manifest.get("files", {})) != expected:
        raise ValueError("evaluation artifact manifest allowlist is invalid")
    if {item.name for item in path.iterdir()} != expected | {"manifest.json"}:
        raise ValueError("evaluation artifact contains unexpected entries")
    for name in expected:
        target = path / name
        record = manifest["files"][name]
        if target.is_symlink() or target.stat().st_size != record.get("byte_count"):
            raise ValueError("evaluation artifact file size or path is invalid")
        if file_sha256(target) != record.get("sha256"):
            raise ValueError("evaluation artifact file digest is invalid")


def _write_artifact(
    *,
    root: Path,
    category: str,
    artifact_id: str,
    content_digest: str,
    manifest_schema: str,
    content_name: str,
    payloads: dict[str, BaseModel],
    load_existing: Callable[[Path], ModelT],
    expected_content: ModelT,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if _root_usage(root) >= artifact_limit_bytes():
        raise OSError("AQSE artifact root has reached its configured byte limit")
    final = _target(root, category, artifact_id)
    if final.exists():
        if load_existing(final) != expected_content:
            raise FileExistsError("immutable evaluation artifact has different content")
        return final
    staging = final.with_name(f".{final.name}.tmp-{uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
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
        candidate = type(expected_content).model_validate(_read_json(staging / content_name))
        if candidate != expected_content:
            raise ValueError("staged evaluation artifact differs from validated content")
        if _root_usage(root) > artifact_limit_bytes():
            raise OSError("evaluation artifact write would exceed the configured limit")
        for item in staging.iterdir():
            item.chmod(0o444)
        os.replace(staging, final)
        return final
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def write_evaluation_protocol(
    protocol: EvaluationProtocol,
    *,
    root: Path | None = None,
) -> tuple[Path, ProtocolExecutionMetadata]:
    validate_evaluation_protocol(protocol)
    resolved_root = (root or artifact_root()).resolve()
    final = _target(resolved_root, "evaluation-protocols", protocol.protocol_id)
    provenance = capture_software_provenance()
    execution = ProtocolExecutionMetadata(
        protocol_id=protocol.protocol_id,
        artifact_path=str(final),
        created_at_utc=_utc_now(),
        repository_base_sha=provenance.repository_base_sha,
        repository_dirty=provenance.repository_dirty,
        runtime_versions=provenance.runtime_versions,
    )

    def load(path: Path) -> EvaluationProtocol:
        return load_evaluation_protocol(path, expected_protocol_id=protocol.protocol_id)[0]

    if final.exists():
        existing, metadata = load_evaluation_protocol(
            final,
            expected_protocol_id=protocol.protocol_id,
        )
        if existing != protocol:
            raise FileExistsError("immutable comparative protocol has different content")
        return final, metadata
    path = _write_artifact(
        root=resolved_root,
        category="evaluation-protocols",
        artifact_id=protocol.protocol_id,
        content_digest=protocol.content_digest,
        manifest_schema="aqse.comparative-protocol-manifest.v1",
        content_name="protocol.json",
        payloads={"protocol.json": protocol, "execution.json": execution},
        load_existing=load,
        expected_content=protocol,
    )
    return path, execution


def load_evaluation_protocol(
    path: Path,
    *,
    expected_protocol_id: str,
) -> tuple[EvaluationProtocol, ProtocolExecutionMetadata]:
    resolved = path.resolve()
    if resolved.name != expected_protocol_id:
        raise ValueError("comparative protocol directory identity is invalid")
    manifest = _read_json(resolved / "manifest.json")
    if manifest.get("schema_version") != "aqse.comparative-protocol-manifest.v1":
        raise ValueError("comparative protocol manifest schema is invalid")
    if manifest.get("artifact_id") != expected_protocol_id:
        raise ValueError("comparative protocol manifest identity is invalid")
    _verify_files(resolved, manifest, {"protocol.json", "execution.json"})
    protocol = EvaluationProtocol.model_validate(_read_json(resolved / "protocol.json"))
    execution = ProtocolExecutionMetadata.model_validate(
        _read_json(resolved / "execution.json")
    )
    validate_evaluation_protocol(protocol)
    if protocol.protocol_id != expected_protocol_id:
        raise ValueError("comparative protocol content identity is invalid")
    if protocol.content_digest != manifest.get("content_digest"):
        raise ValueError("comparative protocol content/manifest digests differ")
    if execution.protocol_id != expected_protocol_id:
        raise ValueError("comparative protocol execution identity is invalid")
    return protocol, execution


def write_comparative_evaluation(
    evaluation: ComparativeEvaluationArtifact,
    *,
    method_runtimes: tuple[MethodRuntime, ...],
    total_wall_time_ms: float,
    root: Path | None = None,
) -> tuple[Path, EvaluationExecutionMetadata]:
    validate_comparative_evaluation(evaluation)
    resolved_root = (root or artifact_root()).resolve()
    final = _target(resolved_root, "comparative-evaluations", evaluation.evaluation_id)
    provenance = capture_software_provenance()
    execution = EvaluationExecutionMetadata(
        evaluation_id=evaluation.evaluation_id,
        artifact_path=str(final),
        created_at_utc=_utc_now(),
        repository_base_sha=provenance.repository_base_sha,
        repository_dirty=provenance.repository_dirty,
        runtime_versions=provenance.runtime_versions,
        total_wall_time_ms=total_wall_time_ms,
        method_runtimes=method_runtimes,
    )

    def load(path: Path) -> ComparativeEvaluationArtifact:
        return load_comparative_evaluation(
            path,
            expected_evaluation_id=evaluation.evaluation_id,
        )[0]

    if final.exists():
        existing, metadata = load_comparative_evaluation(
            final,
            expected_evaluation_id=evaluation.evaluation_id,
        )
        if existing != evaluation:
            raise FileExistsError("immutable comparative evaluation has different content")
        return final, metadata
    path = _write_artifact(
        root=resolved_root,
        category="comparative-evaluations",
        artifact_id=evaluation.evaluation_id,
        content_digest=evaluation.content_digest,
        manifest_schema="aqse.comparative-evaluation-manifest.v1",
        content_name="evaluation.json",
        payloads={"evaluation.json": evaluation, "execution.json": execution},
        load_existing=load,
        expected_content=evaluation,
    )
    return path, execution


def load_comparative_evaluation(
    path: Path,
    *,
    expected_evaluation_id: str,
) -> tuple[ComparativeEvaluationArtifact, EvaluationExecutionMetadata]:
    resolved = path.resolve()
    if resolved.name != expected_evaluation_id:
        raise ValueError("comparative evaluation directory identity is invalid")
    manifest = _read_json(resolved / "manifest.json")
    if manifest.get("schema_version") != "aqse.comparative-evaluation-manifest.v1":
        raise ValueError("comparative evaluation manifest schema is invalid")
    if manifest.get("artifact_id") != expected_evaluation_id:
        raise ValueError("comparative evaluation manifest identity is invalid")
    _verify_files(resolved, manifest, {"evaluation.json", "execution.json"})
    evaluation = ComparativeEvaluationArtifact.model_validate(
        _read_json(resolved / "evaluation.json")
    )
    execution = EvaluationExecutionMetadata.model_validate(
        _read_json(resolved / "execution.json")
    )
    validate_comparative_evaluation(evaluation)
    if evaluation.evaluation_id != expected_evaluation_id:
        raise ValueError("comparative evaluation content identity is invalid")
    if evaluation.content_digest != manifest.get("content_digest"):
        raise ValueError("comparative evaluation content/manifest digests differ")
    if execution.evaluation_id != expected_evaluation_id:
        raise ValueError("comparative evaluation execution identity is invalid")
    return evaluation, execution


def write_model_selection_freeze(
    freeze: ModelSelectionFreeze,
    *,
    root: Path | None = None,
) -> tuple[Path, SelectionFreezeExecutionMetadata]:
    validate_model_selection_freeze(freeze)
    resolved_root = (root or artifact_root()).resolve()
    final = _target(resolved_root, "model-selection-freezes", freeze.freeze_id)
    provenance = capture_software_provenance()
    execution = SelectionFreezeExecutionMetadata(
        freeze_id=freeze.freeze_id,
        artifact_path=str(final),
        created_at_utc=_utc_now(),
        repository_base_sha=provenance.repository_base_sha,
        repository_dirty=provenance.repository_dirty,
        runtime_versions=provenance.runtime_versions,
    )

    def load(path: Path) -> ModelSelectionFreeze:
        return load_model_selection_freeze(path, expected_freeze_id=freeze.freeze_id)[0]

    if final.exists():
        existing, metadata = load_model_selection_freeze(
            final,
            expected_freeze_id=freeze.freeze_id,
        )
        if existing != freeze:
            raise FileExistsError("immutable model-selection freeze has different content")
        return final, metadata
    path = _write_artifact(
        root=resolved_root,
        category="model-selection-freezes",
        artifact_id=freeze.freeze_id,
        content_digest=freeze.content_digest,
        manifest_schema="aqse.model-selection-freeze-manifest.v1",
        content_name="freeze.json",
        payloads={"freeze.json": freeze, "execution.json": execution},
        load_existing=load,
        expected_content=freeze,
    )
    return path, execution


def load_model_selection_freeze(
    path: Path,
    *,
    expected_freeze_id: str,
) -> tuple[ModelSelectionFreeze, SelectionFreezeExecutionMetadata]:
    resolved = path.resolve()
    if resolved.name != expected_freeze_id:
        raise ValueError("model-selection freeze directory identity is invalid")
    manifest = _read_json(resolved / "manifest.json")
    if manifest.get("schema_version") != "aqse.model-selection-freeze-manifest.v1":
        raise ValueError("model-selection freeze manifest schema is invalid")
    if manifest.get("artifact_id") != expected_freeze_id:
        raise ValueError("model-selection freeze manifest identity is invalid")
    _verify_files(resolved, manifest, {"freeze.json", "execution.json"})
    freeze = ModelSelectionFreeze.model_validate(_read_json(resolved / "freeze.json"))
    execution = SelectionFreezeExecutionMetadata.model_validate(
        _read_json(resolved / "execution.json")
    )
    validate_model_selection_freeze(freeze)
    if freeze.freeze_id != expected_freeze_id:
        raise ValueError("model-selection freeze content identity is invalid")
    if freeze.content_digest != manifest.get("content_digest"):
        raise ValueError("model-selection freeze content/manifest digests differ")
    if execution.freeze_id != expected_freeze_id:
        raise ValueError("model-selection freeze execution identity is invalid")
    return freeze, execution
