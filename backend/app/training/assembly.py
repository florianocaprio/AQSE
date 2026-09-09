from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.training.canonical import array_identity, canonical_json_bytes
from app.training.encoding import input_contract_for
from app.training.encoding_storage import load_bank_artifact, load_encoder_artifact
from app.training.models import DatasetManifest, DatasetPartition, LegacyDatasetManifest
from app.training.run_models import TrainingInputIdentity
from app.training.storage import load_labels, load_observations, verify_archive_opaque

CANONICAL_DATASET_ID = "aqse-development-064acca20fc788c6"
CANONICAL_DATASET_DIGEST = (
    "064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7"
)
CANONICAL_PROFILE_FINGERPRINT = (
    "cc6f91470912cfc25025bb6190673f67cd3f56fc8265a7f4da311ad3c0eaeb17"
)
CANONICAL_ENCODER_ID = "aqse-encoder-f9cf4bc12a767410"
CANONICAL_ENCODER_DIGEST = (
    "f9cf4bc12a767410c8759dd01c5bb232ffc3877269af3ed219f3b6484db14e51"
)
CANONICAL_SCALER_ID = "aqse-angle-scaler-fc1d44f00f4cc1be"
CANONICAL_TRAIN_BANK_ID = "aqse-train-bank-32b5f93897676535"
CANONICAL_TRAIN_BANK_DIGEST = (
    "32b5f938976765355f0cd2886bee88082a693a06816f2bae25b9ec268973d8dd"
)
LABEL_POLICY = "aqse.white-noise-regime.v1"


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_training_input_identity(identity: TrainingInputIdentity) -> None:
    scientific = identity.model_dump(
        mode="json", exclude={"fingerprint_id", "content_digest"}
    )
    expected = _digest(scientific)
    if identity.content_digest != expected:
        raise ValueError("training-input content digest is invalid")
    if identity.fingerprint_id != f"aqse-training-input-{expected[:16]}":
        raise ValueError("training-input fingerprint identity is invalid")
    if (
        identity.source_dataset_id != CANONICAL_DATASET_ID
        or identity.source_dataset_digest != CANONICAL_DATASET_DIGEST
        or identity.feature_profile_fingerprint != CANONICAL_PROFILE_FINGERPRINT
        or identity.encoder_artifact_id != CANONICAL_ENCODER_ID
        or identity.encoder_content_digest != CANONICAL_ENCODER_DIGEST
        or identity.scaler_id != CANONICAL_SCALER_ID
        or identity.train_bank_id != CANONICAL_TRAIN_BANK_ID
        or identity.train_bank_digest != CANONICAL_TRAIN_BANK_DIGEST
        or identity.label_policy != LABEL_POLICY
    ):
        raise ValueError("training-input compatibility identity is invalid")


@dataclass(frozen=True)
class TrainingInput:
    X_train: NDArray[np.float64]
    y_train: NDArray[np.int8]
    identity: TrainingInputIdentity
    ordered_lineage_ids: tuple[str, ...]
    ordered_window_ids: tuple[str, ...]


def _canonical_manifest(dataset_path: Path) -> DatasetManifest:
    manifest = verify_archive_opaque(dataset_path)
    if isinstance(manifest, LegacyDatasetManifest):
        raise ValueError("AQSE 1D.3 requires the canonical archive-v2 dataset")
    if (
        manifest.dataset_id != CANONICAL_DATASET_ID
        or manifest.scientific_digest != CANONICAL_DATASET_DIGEST
        or manifest.feature_profile_fingerprint != CANONICAL_PROFILE_FINGERPRINT
    ):
        raise ValueError("AQSE 1D.3 dataset identity is incompatible")
    return manifest


def assemble_training_input(
    *,
    dataset_path: Path,
    encoder_path: Path,
    train_bank_path: Path,
) -> TrainingInput:
    manifest = _canonical_manifest(dataset_path.resolve())
    observations = load_observations(dataset_path, DatasetPartition.TRAIN)
    labels = load_labels(dataset_path, DatasetPartition.TRAIN)
    encoder = load_encoder_artifact(
        encoder_path,
        expected_artifact_id=CANONICAL_ENCODER_ID,
        expected_dataset_id=CANONICAL_DATASET_ID,
        expected_dataset_digest=CANONICAL_DATASET_DIGEST,
        expected_profile_fingerprint=CANONICAL_PROFILE_FINGERPRINT,
    )
    if (
        encoder.artifact.content_digest != CANONICAL_ENCODER_DIGEST
        or encoder.artifact.scaler_id != CANONICAL_SCALER_ID
    ):
        raise ValueError("AQSE 1D.3 encoder/scaler identity is incompatible")
    bank = load_bank_artifact(
        train_bank_path,
        expected_artifact_id=CANONICAL_TRAIN_BANK_ID,
        expected_dataset_id=CANONICAL_DATASET_ID,
        expected_dataset_digest=CANONICAL_DATASET_DIGEST,
        expected_encoder_artifact_id=CANONICAL_ENCODER_ID,
    )
    if bank.content_digest != CANONICAL_TRAIN_BANK_DIGEST:
        raise ValueError("AQSE 1D.3 TRAIN bank digest is incompatible")

    observation_index = {
        episode_id: index for index, episode_id in enumerate(observations.episode_ids)
    }
    labels_by_episode = {label.episode_id: label for label in labels}
    if len(labels_by_episode) != len(labels):
        raise ValueError("TRAIN labels contain duplicate episode identities")
    raw_rows: list[NDArray[np.float64]] = []
    ordered_labels: list[int] = []
    for reference in bank.rows:
        index = observation_index.get(reference.episode_id)
        if index is None:
            raise ValueError("TRAIN bank references an absent observation episode")
        if observations.lineage_ids[index] != reference.lineage_id:
            raise ValueError("TRAIN bank lineage differs from observation lineage")
        if not bool(observations.valid_mask[index, reference.window_ordinal]):
            raise ValueError("TRAIN bank references an ineligible window")
        window = observations.windows[index][reference.window_ordinal]
        if window.window_id != reference.window_id:
            raise ValueError("TRAIN bank window identity differs from observation ledger")
        label = labels_by_episode.get(reference.episode_id)
        if label is None or label.lineage_id != reference.lineage_id:
            raise ValueError("TRAIN label identity/lineage does not match bank reference")
        if label.origin_policy != LABEL_POLICY:
            raise ValueError("TRAIN label policy is incompatible")
        raw_rows.append(np.asarray(observations.features[index, reference.window_ordinal]))
        ordered_labels.append(label.target)

    raw = np.asarray(raw_rows, dtype=np.float64)
    encoded = encoder.transform(raw, context=input_contract_for(encoder.artifact))
    y_train = np.asarray(ordered_labels, dtype=np.int8)
    if encoded.shape != (32, 8) or y_train.shape != (32,):
        raise ValueError("canonical training input must have shapes (32,8) and (32,)")
    if not np.isfinite(encoded).all() or set(y_train.tolist()) != {-1, 1}:
        raise ValueError("canonical training input is non-finite or has invalid labels")
    if int(np.count_nonzero(y_train == -1)) != 16 or int(np.count_nonzero(y_train == 1)) != 16:
        raise ValueError("canonical training input must preserve 16/16 label balance")

    identity_payload = {
        "schema_version": "aqse.training-input.v1",
        "source_dataset_id": manifest.dataset_id,
        "source_dataset_digest": manifest.scientific_digest,
        "feature_profile_fingerprint": manifest.feature_profile_fingerprint,
        "encoding_policy_id": encoder.artifact.encoding_policy_id,
        "encoder_artifact_id": encoder.artifact.artifact_id,
        "encoder_content_digest": encoder.artifact.content_digest,
        "scaler_id": encoder.artifact.scaler_id,
        "train_bank_id": bank.artifact_id,
        "train_bank_digest": bank.content_digest,
        "label_policy": LABEL_POLICY,
        "ordered_rows": [row.model_dump(mode="json") for row in bank.rows],
        "encoded_matrix_identity": array_identity(encoded),
        "ordered_label_identity": array_identity(y_train),
    }
    content_digest = _digest(identity_payload)
    identity = TrainingInputIdentity.model_validate(
        {
            **identity_payload,
            "fingerprint_id": f"aqse-training-input-{content_digest[:16]}",
            "content_digest": content_digest,
        }
    )
    validate_training_input_identity(identity)
    frozen_x = np.asarray(encoded, dtype=np.float64).copy()
    frozen_y = y_train.copy()
    frozen_x.setflags(write=False)
    frozen_y.setflags(write=False)
    return TrainingInput(
        X_train=frozen_x,
        y_train=frozen_y,
        identity=identity,
        ordered_lineage_ids=tuple(row.lineage_id for row in bank.rows),
        ordered_window_ids=tuple(row.window_id for row in bank.rows),
    )
