from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.demo.bundle_models import DemoTaskExample, DemoTaskPartition
from app.demo.knowledge_models import (
    HUMAN_REVIEW_DECLARATION,
    TRAIN_APPROVAL_DECLARATION,
    KnowledgeCollectionEntry,
    KnowledgeObservationEpisode,
    KnowledgeRegistryView,
    KnowledgeReviewedLabel,
    KnowledgeTask,
    KnowledgeTrainApprovalRequest,
    KnowledgeTrainCollectionManifest,
    class_order_for,
    observation_identity_payload,
    profile_id_for,
    reviewed_label_identity_payload,
    task_id_for,
    train_collection_identity_payload,
)
from app.features.state8 import state8_profile, state8_profile_fingerprint
from app.features.state8_models import State8FeatureRecord
from app.training.canonical import canonical_json_bytes, file_sha256
from app.training.storage import artifact_limit_bytes, artifact_root

MAX_OBSERVATION_BYTES = 1 * 1024 * 1024
MAX_LABEL_BYTES = 64 * 1024
MAX_COLLECTION_BYTES = 4 * 1024 * 1024


class KnowledgeStoreError(RuntimeError):
    pass


class KnowledgeConflictError(KnowledgeStoreError):
    pass


class KnowledgeApprovalError(KnowledgeStoreError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise KnowledgeStoreError(f"knowledge path cannot be a symlink: {path.name}")
    if path.exists() and not path.is_dir():
        raise KnowledgeStoreError(f"knowledge path is not a directory: {path.name}")
    path.mkdir(parents=False, exist_ok=True)


def knowledge_root(root: Path | None = None) -> Path:
    base = root if root is not None else artifact_root()
    if base.is_symlink():
        raise KnowledgeStoreError("artifact root cannot be a symlink")
    if base.exists() and not base.is_dir():
        raise KnowledgeStoreError("artifact root is not a directory")
    base.mkdir(parents=True, exist_ok=True)
    network_demo = base / "network-demo"
    _ensure_directory(network_demo)
    knowledge = network_demo / "knowledge"
    _ensure_directory(knowledge)
    for name in ("observations", "labels", "collections"):
        _ensure_directory(knowledge / name)
    return knowledge


def _storage_usage(root: Path) -> int:
    usage = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise KnowledgeStoreError("knowledge storage cannot contain symlinks")
        if path.is_file():
            usage += path.stat().st_size
    return usage


def _read_bounded(path: Path, *, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise KnowledgeStoreError(f"knowledge file is missing or unsafe: {path.name}")
    size = path.stat().st_size
    if size <= 0 or size > limit:
        raise KnowledgeStoreError(f"knowledge file exceeds its bounded reader: {path.name}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise KnowledgeStoreError(f"cannot read knowledge file: {path.name}") from exc


def _lock(root: Path):
    path = root / ".knowledge.lock"
    if path.is_symlink():
        raise KnowledgeStoreError("knowledge lock cannot be a symlink")
    return path.open("a+b")


def _write_immutable_json(
    root: Path,
    path: Path,
    value: Any,
    *,
    limit: int,
) -> bool:
    payload = canonical_json_bytes(value) + b"\n"
    if len(payload) > limit:
        raise KnowledgeStoreError(f"knowledge payload exceeds its limit: {path.name}")
    with _lock(root) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_symlink():
            raise KnowledgeStoreError(f"knowledge target cannot be a symlink: {path.name}")
        if path.exists():
            existing = _read_bounded(path, limit=limit)
            if existing != payload:
                raise KnowledgeConflictError(
                    f"immutable knowledge identity already has different content: {path.name}"
                )
            return True
        if _storage_usage(root) + len(payload) > artifact_limit_bytes():
            raise KnowledgeStoreError("knowledge storage would exceed the artifact limit")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o444)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return False


def capture_observation_episode(
    feature: State8FeatureRecord,
    *,
    task: KnowledgeTask,
    node_count: int,
) -> KnowledgeObservationEpisode:
    """Build a deterministic truth-free observation; no prediction is accepted."""

    expected_profile = state8_profile(profile_id_for(task))
    expected_fingerprint = state8_profile_fingerprint(expected_profile)
    if feature.profile_id != expected_profile.profile_id:
        raise ValueError("captured observation has an incompatible state8 profile")
    if feature.profile_fingerprint != expected_fingerprint:
        raise ValueError("captured observation has an unknown profile fingerprint")
    payload = observation_identity_payload(
        task=task,
        node_count=node_count,
        feature=feature,
    )
    digest = _digest(payload)
    return KnowledgeObservationEpisode(
        observation_id=f"aqse-knowledge-observation-{digest[:16]}",
        acquisition_id=f"aqse-knowledge-acquisition-{digest[16:32]}",
        content_digest=digest,
        task=task,
        task_id=task_id_for(task),
        profile_id=feature.profile_id,
        profile_fingerprint=feature.profile_fingerprint,
        node_count=node_count,
        feature=feature,
    )


def write_observation_episode(
    observation: KnowledgeObservationEpisode,
    *,
    root: Path | None = None,
) -> tuple[Path, bool]:
    storage = knowledge_root(root)
    path = storage / "observations" / f"{observation.observation_id}.json"
    reused = _write_immutable_json(
        storage,
        path,
        observation.model_dump(mode="json"),
        limit=MAX_OBSERVATION_BYTES,
    )
    return path, reused


# Short aliases keep the persistence API convenient for HTTP/service adapters.
capture_observation = capture_observation_episode
write_observation = write_observation_episode


def create_reviewed_label(
    observation: KnowledgeObservationEpisode,
    *,
    label: str,
    reviewer_id: str,
    reviewed_at_utc: str,
) -> KnowledgeReviewedLabel:
    """Create only an explicit human-review record, never an inferred pseudo-label."""

    payload = reviewed_label_identity_payload(
        observation_id=observation.observation_id,
        task=observation.task,
        label=label,
        reviewer_id=reviewer_id,
        reviewed_at_utc=reviewed_at_utc,
    )
    digest = _digest(payload)
    return KnowledgeReviewedLabel(
        label_id=f"aqse-human-label-{digest[:16]}",
        content_digest=digest,
        observation_id=observation.observation_id,
        task=observation.task,
        task_id=observation.task_id,
        label=label,
        review_source="human-review",
        reviewer_id=reviewer_id,
        reviewed_at_utc=reviewed_at_utc,
        review_declaration=HUMAN_REVIEW_DECLARATION,
    )


def _load_observation(path: Path) -> KnowledgeObservationEpisode:
    try:
        payload = json.loads(_read_bounded(path, limit=MAX_OBSERVATION_BYTES))
        return KnowledgeObservationEpisode.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise KnowledgeStoreError("stored knowledge observation is invalid") from exc


def load_observation_episode(
    observation_id: str,
    *,
    root: Path | None = None,
) -> KnowledgeObservationEpisode:
    if not (
        observation_id.startswith("aqse-knowledge-observation-")
        and len(observation_id) == 43
        and all(character in "0123456789abcdef" for character in observation_id[-16:])
    ):
        raise ValueError("invalid knowledge observation identifier")
    storage = knowledge_root(root)
    return _load_observation(storage / "observations" / f"{observation_id}.json")


def _load_label(path: Path) -> KnowledgeReviewedLabel:
    try:
        payload = json.loads(_read_bounded(path, limit=MAX_LABEL_BYTES))
        return KnowledgeReviewedLabel.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise KnowledgeStoreError("stored human-reviewed label is invalid") from exc


def write_reviewed_label(
    label: KnowledgeReviewedLabel,
    *,
    root: Path | None = None,
) -> tuple[Path, bool]:
    storage = knowledge_root(root)
    observation_path = storage / "observations" / f"{label.observation_id}.json"
    observation = _load_observation(observation_path)
    if observation.task is not label.task or observation.task_id != label.task_id:
        raise KnowledgeConflictError("human-reviewed label task differs from observation")
    path = storage / "labels" / f"{label.observation_id}.json"
    reused = _write_immutable_json(
        storage,
        path,
        label.model_dump(mode="json"),
        limit=MAX_LABEL_BYTES,
    )
    return path, reused


def _collection_identity_payload(
    request: KnowledgeTrainApprovalRequest,
    *,
    profile_fingerprint: str,
    entries: tuple[KnowledgeCollectionEntry, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "aqse.network-demo.train-collection.v1",
        "partition": "TRAIN",
        "task": request.task.value,
        "task_id": task_id_for(request.task),
        "profile_id": profile_id_for(request.task),
        "profile_fingerprint": profile_fingerprint,
        "class_order": class_order_for(request.task),
        "approved_by": request.approved_by,
        "approved_at_utc": request.approved_at_utc,
        "approval_declaration": TRAIN_APPROVAL_DECLARATION,
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }


def approve_train_collection(
    request: KnowledgeTrainApprovalRequest,
    *,
    root: Path | None = None,
) -> tuple[Path, KnowledgeTrainCollectionManifest, bool]:
    """Freeze a complete, reviewed, TRAIN-only collection by content identity."""

    storage = knowledge_root(root)
    expected_profile = state8_profile(profile_id_for(request.task))
    expected_fingerprint = state8_profile_fingerprint(expected_profile)
    entries: list[KnowledgeCollectionEntry] = []
    for observation_id in request.observation_ids:
        observation_path = storage / "observations" / f"{observation_id}.json"
        label_path = storage / "labels" / f"{observation_id}.json"
        try:
            observation = _load_observation(observation_path)
            label = _load_label(label_path)
        except KnowledgeStoreError as exc:
            raise KnowledgeApprovalError(
                f"TRAIN approval is incomplete for observation {observation_id}"
            ) from exc
        if observation.observation_id != observation_id:
            raise KnowledgeApprovalError("observation identity differs from its filename")
        if observation.task is not request.task or label.task is not request.task:
            raise KnowledgeApprovalError("TRAIN approval mixes task identities")
        if label.observation_id != observation_id:
            raise KnowledgeApprovalError("reviewed label does not match its observation")
        if (
            observation.profile_id != expected_profile.profile_id
            or observation.profile_fingerprint != expected_fingerprint
        ):
            raise KnowledgeApprovalError("TRAIN approval contains an incompatible profile")
        entries.append(
            KnowledgeCollectionEntry(
                observation_id=observation_id,
                observation_content_digest=observation.content_digest,
                observation_file_sha256=file_sha256(observation_path),
                label_id=label.label_id,
                label_content_digest=label.content_digest,
                label_file_sha256=file_sha256(label_path),
            )
        )
    ordered_entries = tuple(entries)
    payload = _collection_identity_payload(
        request,
        profile_fingerprint=expected_fingerprint,
        entries=ordered_entries,
    )
    digest = _digest(payload)
    manifest = KnowledgeTrainCollectionManifest(
        collection_id=f"aqse-knowledge-train-{digest[:16]}",
        content_digest=digest,
        partition="TRAIN",
        task=request.task,
        task_id=task_id_for(request.task),
        profile_id=expected_profile.profile_id,
        profile_fingerprint=expected_fingerprint,
        class_order=class_order_for(request.task),
        approved_by=request.approved_by,
        approved_at_utc=request.approved_at_utc,
        approval_declaration=TRAIN_APPROVAL_DECLARATION,
        entries=ordered_entries,
    )
    path = storage / "collections" / f"{manifest.collection_id}.json"
    reused = _write_immutable_json(
        storage,
        path,
        manifest.model_dump(mode="json"),
        limit=MAX_COLLECTION_BYTES,
    )
    return path, manifest, reused


def _load_manifest(
    collection_id: str,
    *,
    root: Path | None = None,
) -> tuple[Path, KnowledgeTrainCollectionManifest]:
    if not (
        collection_id.startswith("aqse-knowledge-train-")
        and len(collection_id) == 37
        and all(character in "0123456789abcdef" for character in collection_id[-16:])
    ):
        raise ValueError("invalid knowledge TRAIN collection identifier")
    storage = knowledge_root(root)
    path = storage / "collections" / f"{collection_id}.json"
    try:
        payload = json.loads(_read_bounded(path, limit=MAX_COLLECTION_BYTES))
        manifest = KnowledgeTrainCollectionManifest.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise KnowledgeStoreError("stored knowledge TRAIN collection is invalid") from exc
    digest = _digest(train_collection_identity_payload(manifest))
    if manifest.content_digest != digest:
        raise KnowledgeStoreError("knowledge TRAIN collection digest is invalid")
    if manifest.collection_id != f"aqse-knowledge-train-{digest[:16]}":
        raise KnowledgeStoreError("knowledge TRAIN collection identity is invalid")
    return path, manifest


def load_approved_train_partition(
    collection_id: str,
    *,
    root: Path | None = None,
) -> DemoTaskPartition:
    """Reload a reviewed collection as the existing bounded TRAIN DTO."""

    _, manifest = _load_manifest(collection_id, root=root)
    storage = knowledge_root(root)
    expected_profile = state8_profile(profile_id_for(manifest.task))
    if manifest.profile_fingerprint != state8_profile_fingerprint(expected_profile):
        raise KnowledgeStoreError("knowledge TRAIN profile is no longer compatible")
    examples: list[DemoTaskExample] = []
    for entry in manifest.entries:
        observation_path = storage / "observations" / f"{entry.observation_id}.json"
        label_path = storage / "labels" / f"{entry.observation_id}.json"
        observation = _load_observation(observation_path)
        label = _load_label(label_path)
        if (
            file_sha256(observation_path) != entry.observation_file_sha256
            or file_sha256(label_path) != entry.label_file_sha256
        ):
            raise KnowledgeStoreError("approved TRAIN source content has changed")
        if (
            observation.content_digest != entry.observation_content_digest
            or label.content_digest != entry.label_content_digest
            or label.label_id != entry.label_id
            or label.observation_id != observation.observation_id
            or observation.task is not manifest.task
            or label.task is not manifest.task
        ):
            raise KnowledgeStoreError("approved TRAIN source identity is inconsistent")
        examples.append(
            DemoTaskExample(
                sample_id=observation.observation_id,
                episode_id=observation.observation_id,
                # Existing fit code calls this field "generative"; here it is a
                # deterministic independent acquisition identity, never generator truth.
                generative_lineage_id=observation.acquisition_id,
                node_count=observation.node_count,
                label=label.label,
                feature_values=observation.feature.values,
                eligible=observation.feature.quality.valid_for_quantum,
                quality_flags=observation.feature.quality.flags,
            )
        )
    return DemoTaskPartition(
        study_artifact_id=manifest.collection_id,
        study_content_digest=manifest.content_digest,
        partition="train",
        task_id=manifest.task_id,
        profile_id=manifest.profile_id,
        profile_fingerprint=manifest.profile_fingerprint,
        class_order=manifest.class_order,
        examples=tuple(examples),
    )


def knowledge_registry_view(*, root: Path | None = None) -> KnowledgeRegistryView:
    """Reload the bounded registry without joining labels into observation records."""

    storage = knowledge_root(root)
    observation_paths = tuple(sorted((storage / "observations").glob("*.json")))
    label_paths = tuple(sorted((storage / "labels").glob("*.json")))
    collection_paths = tuple(sorted((storage / "collections").glob("*.json")))
    if max(len(observation_paths), len(label_paths), len(collection_paths)) > 4096:
        raise KnowledgeStoreError("knowledge registry exceeds its bounded index")
    observations = tuple(_load_observation(path) for path in observation_paths)
    labels = tuple(_load_label(path) for path in label_paths)
    collections = tuple(
        _load_manifest(path.stem, root=root)[1] for path in collection_paths
    )
    return KnowledgeRegistryView(
        observations=observations,
        reviewed_labels=labels,
        approved_train_collections=collections,
    )
