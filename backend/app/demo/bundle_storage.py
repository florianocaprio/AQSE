from __future__ import annotations

import fcntl
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from app.training.canonical import canonical_json_bytes, file_sha256
from app.training.storage import artifact_limit_bytes, artifact_root

from .bundle_models import (
    LOCAL_TASK_ID,
    NETWORK_TASK_ID,
    ActiveBundlePointer,
    BundleRuntime,
    DemoFinalEvaluation,
    DemoModelSelectionFreeze,
    DemoResearchBundle,
    canonical_digest,
)

MAX_BUNDLE_JSON_BYTES = 32 * 1024 * 1024
ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class AppliedBundleSet:
    pointer: ActiveBundlePointer
    local: BundleRuntime
    network: BundleRuntime


def _base_root(root: Path | None = None) -> Path:
    base = root or artifact_root()
    if base.is_symlink():
        raise ValueError("AQSE artifact root cannot be a symbolic link")
    base.mkdir(parents=True, exist_ok=True)
    resolved = base.resolve()
    target = resolved / "network-demo" / "registry"
    for directory in (resolved / "network-demo", target):
        if directory.exists() and directory.is_symlink():
            raise ValueError("demo registry path cannot contain a symbolic link")
        directory.mkdir(parents=True, exist_ok=True)
    return target


def _category(root: Path, name: str) -> Path:
    target = root / name
    if target.exists() and target.is_symlink():
        raise ValueError("bundle registry category cannot be a symbolic link")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _read_bounded(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("bundle artifact file is missing or unsafe")
    size = path.stat().st_size
    if size <= 0 or size > MAX_BUNDLE_JSON_BYTES:
        raise ValueError("bundle artifact file exceeds the bounded reader")
    return path.read_bytes()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(_read_bounded(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("bundle artifact JSON is invalid") from exc


def _root_usage(root: Path) -> int:
    return sum(
        item.stat().st_size
        for item in root.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def _artifact_target(category: Path, artifact_id: str) -> Path:
    target = category / artifact_id
    if category.resolve() not in target.resolve(strict=False).parents:
        raise ValueError("bundle artifact path escapes the configured registry")
    return target


def _verify_directory(
    path: Path,
    *,
    artifact_id: str,
    content_digest: str,
    manifest_schema: str,
    content_name: str,
) -> None:
    if path.is_symlink() or not path.is_dir() or path.name != artifact_id:
        raise ValueError("bundle artifact directory identity is invalid or unsafe")
    if {item.name for item in path.iterdir()} != {content_name, "manifest.json"}:
        raise ValueError("bundle artifact contains unexpected entries")
    manifest = _read_json(path / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("bundle artifact manifest is invalid")
    if (
        manifest.get("schema_version") != manifest_schema
        or manifest.get("artifact_id") != artifact_id
        or manifest.get("content_digest") != content_digest
        or set(manifest.get("files", {})) != {content_name}
    ):
        raise ValueError("bundle artifact manifest contract is invalid")
    record = manifest["files"][content_name]
    content = path / content_name
    if (
        content.is_symlink()
        or content.stat().st_size != record.get("byte_count")
        or file_sha256(content) != record.get("sha256")
    ):
        raise ValueError("bundle artifact content identity is invalid")


def _write_immutable(
    value: ModelT,
    *,
    root: Path,
    category_name: str,
    artifact_id: str,
    content_digest: str,
    manifest_schema: str,
    content_name: str,
    loader: Callable[[Path], ModelT],
) -> tuple[Path, bool]:
    category = _category(root, category_name)
    target = _artifact_target(category, artifact_id)
    if target.is_symlink():
        raise ValueError("bundle artifact target cannot be a symbolic link")
    if target.exists():
        if target.is_symlink() or loader(target) != value:
            raise FileExistsError("immutable bundle artifact has different content")
        return target, True
    content = canonical_json_bytes(value.model_dump(mode="json")) + b"\n"
    if len(content) > MAX_BUNDLE_JSON_BYTES:
        raise OSError("bundle artifact exceeds the configured per-file limit")
    staging = Path(mkdtemp(prefix=f".{artifact_id}.", dir=category))
    try:
        content_path = staging / content_name
        content_path.write_bytes(content)
        manifest = {
            "schema_version": manifest_schema,
            "artifact_id": artifact_id,
            "content_digest": content_digest,
            "files": {
                content_name: {
                    "byte_count": len(content),
                    "sha256": file_sha256(content_path),
                }
            },
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        if file_sha256(content_path) != manifest["files"][content_name]["sha256"]:
            raise RuntimeError("staged bundle content digest changed during publication")
        artifact_base = root.parents[1]
        if _root_usage(artifact_base) > artifact_limit_bytes():
            raise OSError("bundle artifact write would exceed the AQSE artifact limit")
        content_path.chmod(0o444)
        (staging / "manifest.json").chmod(0o444)
        try:
            os.replace(staging, target)
        except OSError:
            if target.exists() and not target.is_symlink() and loader(target) == value:
                shutil.rmtree(staging)
                return target, True
            raise
        loaded = loader(target)
        if loaded != value:
            raise RuntimeError("published bundle artifact differs from staged content")
        return target, False
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def write_bundle(
    bundle: DemoResearchBundle,
    *,
    root: Path | None = None,
) -> tuple[Path, bool]:
    registry = _base_root(root)
    return _write_immutable(
        bundle,
        root=registry,
        category_name="bundles",
        artifact_id=bundle.bundle_id,
        content_digest=bundle.content_digest,
        manifest_schema="aqse.network-demo.bundle-manifest.v1",
        content_name="bundle.json",
        loader=load_bundle,
    )


def load_bundle(path: Path) -> DemoResearchBundle:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("bundle path must be a real directory")
    value = DemoResearchBundle.model_validate(_read_json(path / "bundle.json"))
    _verify_directory(
        path,
        artifact_id=value.bundle_id,
        content_digest=value.content_digest,
        manifest_schema="aqse.network-demo.bundle-manifest.v1",
        content_name="bundle.json",
    )
    return value


def list_bundles(*, root: Path | None = None) -> tuple[DemoResearchBundle, ...]:
    category = _category(_base_root(root), "bundles")
    values: list[DemoResearchBundle] = []
    for path in sorted(category.iterdir(), key=lambda item: item.name):
        if path.name.startswith("."):
            continue
        values.append(load_bundle(path))
    return tuple(values)


def load_bundle_by_id(
    bundle_id: str,
    *,
    root: Path | None = None,
) -> DemoResearchBundle:
    registry = _base_root(root)
    return _load_registered_bundle(registry, bundle_id)


def write_selection_freeze(
    freeze: DemoModelSelectionFreeze,
    *,
    root: Path | None = None,
) -> tuple[Path, bool]:
    registry = _base_root(root)
    return _write_immutable(
        freeze,
        root=registry,
        category_name="selection-freezes",
        artifact_id=freeze.freeze_id,
        content_digest=freeze.content_digest,
        manifest_schema="aqse.network-demo.selection-freeze-manifest.v1",
        content_name="selection-freeze.json",
        loader=load_selection_freeze,
    )


def load_selection_freeze(path: Path) -> DemoModelSelectionFreeze:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("selection-freeze path must be a real directory")
    value = DemoModelSelectionFreeze.model_validate(
        _read_json(path / "selection-freeze.json")
    )
    _verify_directory(
        path,
        artifact_id=value.freeze_id,
        content_digest=value.content_digest,
        manifest_schema="aqse.network-demo.selection-freeze-manifest.v1",
        content_name="selection-freeze.json",
    )
    return value


def load_selection_freeze_by_id(
    freeze_id: str,
    *,
    root: Path | None = None,
) -> DemoModelSelectionFreeze:
    registry = _base_root(root)
    category = _category(registry, "selection-freezes")
    return load_selection_freeze(_artifact_target(category, freeze_id))


def write_final_evaluation(
    evaluation: DemoFinalEvaluation,
    *,
    root: Path | None = None,
) -> tuple[Path, bool]:
    registry = _base_root(root)
    return _write_immutable(
        evaluation,
        root=registry,
        category_name="final-evaluations",
        artifact_id=evaluation.evaluation_id,
        content_digest=evaluation.content_digest,
        manifest_schema="aqse.network-demo.final-evaluation-manifest.v1",
        content_name="final-evaluation.json",
        loader=load_final_evaluation,
    )


def load_final_evaluation(path: Path) -> DemoFinalEvaluation:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("final-evaluation path must be a real directory")
    value = DemoFinalEvaluation.model_validate(_read_json(path / "final-evaluation.json"))
    _verify_directory(
        path,
        artifact_id=value.evaluation_id,
        content_digest=value.content_digest,
        manifest_schema="aqse.network-demo.final-evaluation-manifest.v1",
        content_name="final-evaluation.json",
    )
    return value


def _selected_ids(
    freeze: DemoModelSelectionFreeze,
) -> tuple[str, str]:
    selections = {item.task_id: item.selected_bundle_id for item in freeze.selections}
    return selections[LOCAL_TASK_ID], selections[NETWORK_TASK_ID]


def _load_registered_bundle(registry: Path, bundle_id: str) -> DemoResearchBundle:
    return load_bundle(_artifact_target(_category(registry, "bundles"), bundle_id))


def _validate_application_set(
    freeze: DemoModelSelectionFreeze,
    local: DemoResearchBundle,
    network: DemoResearchBundle,
) -> None:
    if local.task_id != LOCAL_TASK_ID or network.task_id != NETWORK_TASK_ID:
        raise ValueError("application requires one local and one network bundle")
    expected = _selected_ids(freeze)
    if (local.bundle_id, network.bundle_id) != expected:
        raise ValueError("application bundle set differs from the selection freeze")
    for bundle in (local, network):
        if (
            bundle.study_artifact_id != freeze.study_artifact_id
            or bundle.study_content_digest != freeze.study_content_digest
            or bundle.protocol_digest != freeze.protocol_digest
        ):
            raise ValueError("application rejects an old or incompatible bundle")


def _pointer_path(registry: Path) -> Path:
    return registry / "active-bundles.json"


def _read_pointer(registry: Path) -> ActiveBundlePointer | None:
    path = _pointer_path(registry)
    if path.is_symlink():
        raise ValueError("active-bundle pointer cannot be a symbolic link")
    if not path.exists():
        return None
    return ActiveBundlePointer.model_validate(_read_json(path))


def apply_selection_freeze(
    freeze: DemoModelSelectionFreeze,
    *,
    root: Path | None = None,
) -> ActiveBundlePointer:
    """Atomically apply both selected spaces; identical retries are idempotent."""

    registry = _base_root(root)
    stored_freeze = load_selection_freeze_by_id(freeze.freeze_id, root=root)
    if stored_freeze != freeze:
        raise ValueError("application requires the identical persisted selection freeze")
    local_id, network_id = _selected_ids(freeze)
    local = _load_registered_bundle(registry, local_id)
    network = _load_registered_bundle(registry, network_id)
    _validate_application_set(freeze, local, network)
    lock_path = registry / "active-bundles.lock"
    if lock_path.exists() and lock_path.is_symlink():
        raise ValueError("active-bundle lock cannot be a symbolic link")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        previous = _read_pointer(registry)
        if previous is not None and (
            previous.selection_freeze_id == freeze.freeze_id
            and previous.local_bundle_id == local_id
            and previous.network_bundle_id == network_id
        ):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return previous
        revision = 1 if previous is None else previous.revision + 1
        scientific = {
            "schema_version": "aqse.network-demo.active-bundles.v1",
            "generation": revision,
            "revision": revision,
            "selection_freeze_id": freeze.freeze_id,
            "local_bundle_id": local_id,
            "network_bundle_id": network_id,
            "previous_application_id": (
                None if previous is None else previous.application_id
            ),
        }
        digest = canonical_digest(scientific)
        pointer = ActiveBundlePointer(
            **scientific,
            application_id=f"aqse-demo-application-{digest[:16]}",
        )
        target = _pointer_path(registry)
        if target.exists() and target.is_symlink():
            raise ValueError("active-bundle pointer cannot be a symbolic link")
        temporary = registry / f".active-bundles.{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as output:
                output.write(canonical_json_bytes(pointer.model_dump(mode="json")) + b"\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return pointer


def load_active_bundle_set(*, root: Path | None = None) -> AppliedBundleSet | None:
    """Reload the applied model set and immutable runtime state after restart."""

    registry = _base_root(root)
    pointer = _read_pointer(registry)
    if pointer is None:
        return None
    freezes = _category(registry, "selection-freezes")
    freeze = load_selection_freeze(_artifact_target(freezes, pointer.selection_freeze_id))
    local = _load_registered_bundle(registry, pointer.local_bundle_id)
    network = _load_registered_bundle(registry, pointer.network_bundle_id)
    _validate_application_set(freeze, local, network)
    if _selected_ids(freeze) != (pointer.local_bundle_id, pointer.network_bundle_id):
        raise ValueError("active-bundle pointer differs from its selection freeze")
    return AppliedBundleSet(
        pointer=pointer,
        local=BundleRuntime.from_artifact(local),
        network=BundleRuntime.from_artifact(network),
    )
