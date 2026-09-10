from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.training.assembly import (
    CANONICAL_DATASET_ID,
    CANONICAL_ENCODER_DIGEST,
    CANONICAL_ENCODER_ID,
    CANONICAL_PROFILE_FINGERPRINT,
    LABEL_POLICY,
    assemble_training_input,
)
from app.training.canonical import array_identity, canonical_json_bytes
from app.training.encoding import input_contract_for
from app.training.encoding_models import QuantumBankArtifact
from app.training.encoding_storage import load_bank_artifact, load_encoder_artifact
from app.training.evaluation_models import ComparisonInputIdentity
from app.training.evaluation_protocol import (
    DATASET_DIGEST,
    TRAIN_BANK_DIGEST,
    VALIDATION_BANK_DIGEST,
)
from app.training.models import DatasetManifest, DatasetPartition, LegacyDatasetManifest
from app.training.storage import (
    ObservablePartition,
    load_labels,
    load_observations,
    verify_archive_opaque,
)

TRAIN_BANK_ID = "aqse-train-bank-32b5f93897676535"
VALIDATION_BANK_ID = "aqse-validation-bank-105467fa5a4a5cc7"


@dataclass(frozen=True)
class ComparisonInput:
    X_train_raw: NDArray[np.float64]
    X_train_encoded: NDArray[np.float64]
    y_train: NDArray[np.int8]
    X_validation_raw: NDArray[np.float64]
    X_validation_encoded: NDArray[np.float64]
    y_validation: NDArray[np.int8]
    identity: ComparisonInputIdentity


@dataclass(frozen=True)
class _BankRows:
    raw: NDArray[np.float64]
    labels: NDArray[np.int8]
    lineage_ids: tuple[str, ...]
    window_ids: tuple[str, ...]


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_manifest(dataset_path: Path) -> DatasetManifest:
    manifest = verify_archive_opaque(dataset_path)
    if isinstance(manifest, LegacyDatasetManifest):
        raise ValueError("Milestone 1D.4a requires the canonical archive-v2 dataset")
    if (
        manifest.dataset_id != CANONICAL_DATASET_ID
        or manifest.scientific_digest != DATASET_DIGEST
        or manifest.feature_profile_fingerprint != CANONICAL_PROFILE_FINGERPRINT
        or manifest.test_state != "sealed"
    ):
        raise ValueError("Milestone 1D.4a dataset or TEST-seal identity is incompatible")
    return manifest


def _ordered_bank_rows(
    observations: ObservablePartition,
    bank: QuantumBankArtifact,
    labels_by_episode: dict[str, object],
) -> _BankRows:
    observation_index = {
        episode_id: index for index, episode_id in enumerate(observations.episode_ids)
    }
    raw_rows: list[NDArray[np.float64]] = []
    targets: list[int] = []
    for reference in bank.rows:
        index = observation_index.get(reference.episode_id)
        if index is None:
            raise ValueError("comparison bank references an absent observation episode")
        if observations.lineage_ids[index] != reference.lineage_id:
            raise ValueError("comparison bank lineage differs from observation lineage")
        if not bool(observations.valid_mask[index, reference.window_ordinal]):
            raise ValueError("comparison bank references an ineligible window")
        window = observations.windows[index][reference.window_ordinal]
        if window.window_id != reference.window_id:
            raise ValueError("comparison bank window differs from observation ledger")
        label = labels_by_episode.get(reference.episode_id)
        if label is None:
            raise ValueError("comparison label is absent")
        if getattr(label, "lineage_id") != reference.lineage_id:
            raise ValueError("comparison label lineage differs from the bank")
        if getattr(label, "origin_policy") != LABEL_POLICY:
            raise ValueError("comparison label policy is incompatible")
        raw_rows.append(np.asarray(observations.features[index, reference.window_ordinal]))
        targets.append(int(getattr(label, "target")))
    raw = np.asarray(raw_rows, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int8)
    return _BankRows(
        raw=raw,
        labels=labels,
        lineage_ids=tuple(item.lineage_id for item in bank.rows),
        window_ids=tuple(item.window_id for item in bank.rows),
    )


def _load_rows(
    *,
    dataset_path: Path,
    bank_path: Path,
    bank_id: str,
    bank_digest: str,
    partition: DatasetPartition,
) -> _BankRows:
    observations = load_observations(dataset_path, partition)
    labels = load_labels(dataset_path, partition)
    labels_by_episode = {item.episode_id: item for item in labels}
    if len(labels_by_episode) != len(labels):
        raise ValueError("comparison labels contain duplicate episode identities")
    bank = load_bank_artifact(
        bank_path,
        expected_artifact_id=bank_id,
        expected_dataset_id=CANONICAL_DATASET_ID,
        expected_dataset_digest=DATASET_DIGEST,
        expected_encoder_artifact_id=CANONICAL_ENCODER_ID,
    )
    if bank.content_digest != bank_digest or bank.source_partition is not partition:
        raise ValueError("comparison bank identity or partition is incompatible")
    return _ordered_bank_rows(observations, bank, labels_by_episode)


def _freeze(array: NDArray[np.generic]) -> NDArray[np.generic]:
    value = np.asarray(array).copy()
    value.setflags(write=False)
    return value


def assemble_comparison_input(
    *,
    dataset_path: Path,
    encoder_path: Path,
    train_bank_path: Path,
    validation_bank_path: Path,
) -> ComparisonInput:
    manifest = _canonical_manifest(dataset_path.resolve())
    encoder = load_encoder_artifact(
        encoder_path,
        expected_artifact_id=CANONICAL_ENCODER_ID,
        expected_dataset_id=CANONICAL_DATASET_ID,
        expected_dataset_digest=DATASET_DIGEST,
        expected_profile_fingerprint=CANONICAL_PROFILE_FINGERPRINT,
    )
    if encoder.artifact.content_digest != CANONICAL_ENCODER_DIGEST:
        raise ValueError("comparison encoder digest is incompatible")
    train = _load_rows(
        dataset_path=dataset_path,
        bank_path=train_bank_path,
        bank_id=TRAIN_BANK_ID,
        bank_digest=TRAIN_BANK_DIGEST,
        partition=DatasetPartition.TRAIN,
    )
    validation = _load_rows(
        dataset_path=dataset_path,
        bank_path=validation_bank_path,
        bank_id=VALIDATION_BANK_ID,
        bank_digest=VALIDATION_BANK_DIGEST,
        partition=DatasetPartition.VALIDATION,
    )
    context = input_contract_for(encoder.artifact)
    encoded_train = np.asarray(encoder.transform(train.raw, context=context), dtype=np.float64)
    encoded_validation = np.asarray(
        encoder.transform(validation.raw, context=context),
        dtype=np.float64,
    )
    canonical_training = assemble_training_input(
        dataset_path=dataset_path,
        encoder_path=encoder_path,
        train_bank_path=train_bank_path,
    )
    if not np.array_equal(encoded_train, canonical_training.X_train):
        raise ValueError("comparison TRAIN encoding differs from the frozen 1D.3 input")
    if not np.array_equal(train.labels, canonical_training.y_train):
        raise ValueError("comparison TRAIN labels differ from the frozen 1D.3 input")
    if encoded_train.shape != (32, 8) or encoded_validation.shape != (24, 8):
        raise ValueError("comparison encoded matrix shapes must be 32x8 and 24x8")
    if train.raw.shape != (32, 8) or validation.raw.shape != (24, 8):
        raise ValueError("comparison raw matrix shapes must be 32x8 and 24x8")
    if set(train.labels.tolist()) != {-1, 1} or set(validation.labels.tolist()) != {-1, 1}:
        raise ValueError("comparison partitions must both contain the two approved classes")
    if int(np.count_nonzero(validation.labels == -1)) != 12:
        raise ValueError("VALIDATION must contain 12 nominal lineages")
    if int(np.count_nonzero(validation.labels == 1)) != 12:
        raise ValueError("VALIDATION must contain 12 elevated lineages")
    if set(train.lineage_ids) & set(validation.lineage_ids):
        raise ValueError("TRAIN and VALIDATION lineages overlap")

    identity_payload = {
        "schema_version": "aqse.comparison-input.v1",
        "dataset_id": manifest.dataset_id,
        "dataset_digest": manifest.scientific_digest,
        "encoder_id": encoder.artifact.artifact_id,
        "encoder_digest": encoder.artifact.content_digest,
        "train_bank_id": TRAIN_BANK_ID,
        "train_bank_digest": TRAIN_BANK_DIGEST,
        "validation_bank_id": VALIDATION_BANK_ID,
        "validation_bank_digest": VALIDATION_BANK_DIGEST,
        "train_raw_identity": array_identity(train.raw),
        "train_encoded_identity": array_identity(encoded_train),
        "train_label_identity": array_identity(train.labels),
        "validation_raw_identity": array_identity(validation.raw),
        "validation_encoded_identity": array_identity(encoded_validation),
        "validation_label_identity": array_identity(validation.labels),
        "train_lineage_ids": train.lineage_ids,
        "validation_lineage_ids": validation.lineage_ids,
        "train_window_ids": train.window_ids,
        "validation_window_ids": validation.window_ids,
    }
    digest = _digest(identity_payload)
    identity = ComparisonInputIdentity.model_validate(
        {
            **identity_payload,
            "fingerprint_id": f"aqse-comparison-input-{digest[:16]}",
            "content_digest": digest,
        }
    )
    return ComparisonInput(
        X_train_raw=_freeze(train.raw),
        X_train_encoded=_freeze(encoded_train),
        y_train=_freeze(train.labels),
        X_validation_raw=_freeze(validation.raw),
        X_validation_encoded=_freeze(encoded_validation),
        y_validation=_freeze(validation.labels),
        identity=identity,
    )
