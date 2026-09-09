from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.training.canonical import canonical_json_bytes, file_sha256
from app.training.encoding import (
    CANONICAL_DEVELOPMENT_DATASET_ID,
    CANONICAL_DEVELOPMENT_DIGEST,
)
from app.training.encoding_models import BankRowReference, QuantumBankArtifact
from app.training.models import DatasetManifest, DatasetPartition, EpisodeLabel
from app.training.storage import ObservablePartition

TRAIN_BANK_SIZE = 32
TRAIN_PER_CLASS = 16
TRAIN_SELECTION_SEED = 1_001_004
CENTRAL_WINDOW_ORDINAL = 9
VALIDATION_BANK_SIZE = 24


def _selection_source_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _bank_digest_payload(bank: QuantumBankArtifact) -> dict[str, Any]:
    return bank.model_dump(mode="json", exclude={"artifact_id", "content_digest"})


def validate_bank_artifact(bank: QuantumBankArtifact) -> None:
    if bank.source_dataset_id != CANONICAL_DEVELOPMENT_DATASET_ID:
        raise ValueError("quantum bank is not bound to the canonical development dataset")
    if bank.source_dataset_digest != CANONICAL_DEVELOPMENT_DIGEST:
        raise ValueError("quantum bank canonical dataset digest is invalid")
    if bank.window_ordinal != CENTRAL_WINDOW_ORDINAL:
        raise ValueError("quantum bank must use the fixed central window ordinal")
    if bank.selection_source_sha256 != _selection_source_sha256():
        raise ValueError("quantum-bank selection source identity is incompatible")
    if bank.bank_kind == "train":
        if bank.source_partition is not DatasetPartition.TRAIN:
            raise ValueError("training quantum bank must use TRAIN only")
        if bank.row_count != TRAIN_BANK_SIZE or bank.lineage_count != TRAIN_BANK_SIZE:
            raise ValueError("training quantum bank must contain 32 distinct lineages")
        if bank.selection_seed != TRAIN_SELECTION_SEED:
            raise ValueError("training quantum bank selection seed is invalid")
        if bank.class_balance != {"-1": TRAIN_PER_CLASS, "1": TRAIN_PER_CLASS}:
            raise ValueError("training quantum bank must preserve 16/16 class balance")
    else:
        if bank.source_partition is not DatasetPartition.VALIDATION:
            raise ValueError("validation quantum bank must use VALIDATION only")
        if bank.row_count != VALIDATION_BANK_SIZE or bank.lineage_count != VALIDATION_BANK_SIZE:
            raise ValueError("validation quantum bank must contain all 24 lineages")
        if bank.selection_seed is not None or bank.class_balance is not None:
            raise ValueError("validation bank selection must not depend on validation labels")
    expected_digest = _digest(_bank_digest_payload(bank))
    if bank.content_digest != expected_digest:
        raise ValueError("quantum-bank scientific content digest is invalid")
    expected_id = f"aqse-{bank.bank_kind}-bank-{expected_digest[:16]}"
    if bank.artifact_id != expected_id:
        raise ValueError("quantum-bank artifact identity is invalid")


def _validate_source(
    observations: ObservablePartition,
    manifest: DatasetManifest,
    expected_partition: DatasetPartition,
) -> None:
    if observations.partition is not expected_partition:
        raise ValueError(f"bank selection requires {expected_partition.value} observations")
    if manifest.dataset_id != observations.dataset_id:
        raise ValueError("bank manifest and observation dataset identities differ")
    if manifest.dataset_id != CANONICAL_DEVELOPMENT_DATASET_ID:
        raise ValueError("bank selection requires the explicit canonical dataset")
    if manifest.scientific_digest != CANONICAL_DEVELOPMENT_DIGEST:
        raise ValueError("bank canonical dataset digest mismatch")
    if observations.valid_mask.shape[0] != len(observations.episode_ids):
        raise ValueError("bank valid-mask rows do not align with episodes")


def _row_reference(
    observations: ObservablePartition, episode_index: int
) -> BankRowReference:
    if observations.valid_mask.shape[1] <= CENTRAL_WINDOW_ORDINAL:
        raise ValueError("fixed central window ordinal is absent")
    if not bool(observations.valid_mask[episode_index, CENTRAL_WINDOW_ORDINAL]):
        raise ValueError("fixed central window is not quantum-eligible")
    record = observations.windows[episode_index][CENTRAL_WINDOW_ORDINAL]
    if record.start_index < 0 or record.end_index <= record.start_index:
        raise ValueError("fixed central window reference is invalid")
    return BankRowReference(
        episode_id=observations.episode_ids[episode_index],
        lineage_id=observations.lineage_ids[episode_index],
        partition=observations.partition,  # type: ignore[arg-type]
        window_id=record.window_id,
    )


def _create_artifact(
    *,
    kind: Literal["train", "validation"],
    partition: DatasetPartition,
    manifest: DatasetManifest,
    encoder_artifact_id: str,
    rows: tuple[BankRowReference, ...],
    selection_seed: int | None,
    class_balance: dict[str, int] | None,
) -> QuantumBankArtifact:
    scientific = {
        "schema_version": "aqse.quantum-bank.v1",
        "bank_kind": kind,
        "source_dataset_id": manifest.dataset_id,
        "source_dataset_digest": manifest.scientific_digest,
        "source_partition": partition.value,
        "encoder_artifact_id": encoder_artifact_id,
        "selection_source_sha256": _selection_source_sha256(),
        "selection_policy": "aqse.fixed-central-window.v1",
        "selection_seed": selection_seed,
        "window_ordinal": CENTRAL_WINDOW_ORDINAL,
        "row_count": len(rows),
        "lineage_count": len({item.lineage_id for item in rows}),
        "class_balance": class_balance,
        "rows": [item.model_dump(mode="json") for item in rows],
    }
    content_digest = _digest(scientific)
    artifact = QuantumBankArtifact.model_validate(
        {
            **scientific,
            "artifact_id": f"aqse-{kind}-bank-{content_digest[:16]}",
            "content_digest": content_digest,
        }
    )
    validate_bank_artifact(artifact)
    return artifact


def select_training_quantum_bank(
    observations: ObservablePartition,
    labels: tuple[EpisodeLabel, ...],
    manifest: DatasetManifest,
    *,
    encoder_artifact_id: str,
    selection_seed: int = TRAIN_SELECTION_SEED,
) -> QuantumBankArtifact:
    _validate_source(observations, manifest, DatasetPartition.TRAIN)
    if selection_seed != TRAIN_SELECTION_SEED:
        raise ValueError("1D.2 training-bank seed is fixed at 1001004")
    labels_by_episode = {item.episode_id: item for item in labels}
    if len(labels_by_episode) != len(labels) or set(labels_by_episode) != set(
        observations.episode_ids
    ):
        raise ValueError("TRAIN labels and observations must have identical episode IDs")
    index_by_episode = {
        episode_id: index for index, episode_id in enumerate(observations.episode_ids)
    }
    rows: list[BankRowReference] = []
    for target in (-1, 1):
        candidates = sorted(
            episode_id
            for episode_id, label in labels_by_episode.items()
            if label.target == target
        )
        if len(candidates) < TRAIN_PER_CLASS:
            raise ValueError("TRAIN class does not contain 16 eligible episode candidates")
        rng = np.random.default_rng(
            np.random.SeedSequence([selection_seed, 0 if target == -1 else 1])
        )
        selected = [
            candidates[index]
            for index in rng.permutation(len(candidates))[:TRAIN_PER_CLASS]
        ]
        for episode_id in selected:
            label = labels_by_episode[episode_id]
            index = index_by_episode[episode_id]
            if label.lineage_id != observations.lineage_ids[index]:
                raise ValueError("TRAIN label lineage differs from observable lineage")
            rows.append(_row_reference(observations, index))
    return _create_artifact(
        kind="train",
        partition=DatasetPartition.TRAIN,
        manifest=manifest,
        encoder_artifact_id=encoder_artifact_id,
        rows=tuple(rows),
        selection_seed=selection_seed,
        class_balance={"-1": TRAIN_PER_CLASS, "1": TRAIN_PER_CLASS},
    )


def select_validation_quantum_bank(
    observations: ObservablePartition,
    manifest: DatasetManifest,
    *,
    encoder_artifact_id: str,
) -> QuantumBankArtifact:
    _validate_source(observations, manifest, DatasetPartition.VALIDATION)
    if len(observations.episode_ids) != VALIDATION_BANK_SIZE:
        raise ValueError("validation bank requires all 24 validation episodes")
    index_by_episode = {
        episode_id: index for index, episode_id in enumerate(observations.episode_ids)
    }
    rows = tuple(
        _row_reference(observations, index_by_episode[episode_id])
        for episode_id in sorted(observations.episode_ids)
    )
    return _create_artifact(
        kind="validation",
        partition=DatasetPartition.VALIDATION,
        manifest=manifest,
        encoder_artifact_id=encoder_artifact_id,
        rows=rows,
        selection_seed=None,
        class_balance=None,
    )
