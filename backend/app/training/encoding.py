from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.features.models import FeatureProfile
from app.quantum.user_pipeline.tqk8 import AngleScaler
from app.training.canonical import array_identity, canonical_json_bytes, file_sha256
from app.training.encoding_models import (
    ENCODING_POLICY_ID,
    PHASE_WRAP_CONVENTION,
    EncoderScientificArtifact,
    EncodingInputContract,
    FittingPopulation,
    TQKCompatibility,
)
from app.training.models import DatasetManifest, DatasetPartition
from app.training.storage import ObservablePartition

CANONICAL_DEVELOPMENT_DATASET_ID = "aqse-development-064acca20fc788c6"
CANONICAL_DEVELOPMENT_DIGEST = (
    "064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7"
)
CANONICAL_FEATURE_PROFILE_FINGERPRINT = (
    "cc6f91470912cfc25025bb6190673f67cd3f56fc8265a7f4da311ad3c0eaeb17"
)
TQK8_SOURCE_SHA256 = "cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689"
PHASE_INDEX = 1
FEATURE_COUNT = 8


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _effective_source_hashes() -> dict[str, str]:
    backend_root = _backend_root()
    relative_paths = (
        "app/quantum/user_pipeline/tqk8.py",
        "app/training/encoding.py",
        "app/training/encoding_models.py",
    )
    return {relative: file_sha256(backend_root / relative) for relative in relative_paths}


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def wrap_phase_direct(values: NDArray[Any] | list[float]) -> NDArray[np.float64]:
    phase = np.asarray(values, dtype=np.float64)
    if not np.isfinite(phase).all():
        raise ValueError("phase-direct input contains NaN or infinity")
    wrapped = np.remainder(phase + pi, 2.0 * pi) - pi
    return np.asarray(wrapped, dtype=np.float64)


def _artifact_digest_payload(artifact: EncoderScientificArtifact) -> dict[str, Any]:
    return artifact.model_dump(mode="json", exclude={"artifact_id", "content_digest"})


def validate_encoder_artifact(artifact: EncoderScientificArtifact) -> None:
    profile = FeatureProfile.model_validate(artifact.feature_profile)
    profile_payload = profile.model_dump(mode="json")
    profile_fingerprint = _digest(profile_payload)
    if profile_fingerprint != artifact.feature_profile_fingerprint:
        raise ValueError("encoder feature-profile fingerprint is invalid")
    if tuple(profile.feature_names) != artifact.feature_order:
        raise ValueError("encoder feature order differs from the full profile")
    if tuple(profile.feature_units) != artifact.feature_units:
        raise ValueError("encoder feature units differ from the full profile")
    if profile.phase_reference != artifact.phase_reference:
        raise ValueError("encoder phase reference differs from the full profile")
    if artifact.phase_wrap_convention != PHASE_WRAP_CONVENTION:
        raise ValueError("encoder phase-wrap convention is incompatible")
    expected_sources = _effective_source_hashes()
    if artifact.effective_source_hashes != expected_sources:
        raise ValueError("encoder effective source identity is incompatible")
    if expected_sources["app/quantum/user_pipeline/tqk8.py"] != TQK8_SOURCE_SHA256:
        raise ValueError("protected TQK8 source hash differs from the approved identity")
    if artifact.tqk_compatibility.tqk8_source_sha256 != TQK8_SOURCE_SHA256:
        raise ValueError("encoder TQK compatibility identity is invalid")
    scaler_payload = {
        "algorithm": artifact.scaler_algorithm,
        "source_sha256": artifact.scaler_source_sha256,
        "fitting_population_id": artifact.fitting_population.population_id,
        "mean": list(artifact.fitted_mean),
        "scale": list(artifact.fitted_scale),
    }
    expected_scaler_id = f"aqse-angle-scaler-{_digest(scaler_payload)[:16]}"
    if artifact.scaler_id != expected_scaler_id:
        raise ValueError("encoder scaler identity is invalid")
    expected_digest = _digest(_artifact_digest_payload(artifact))
    if artifact.content_digest != expected_digest:
        raise ValueError("encoder scientific content digest is invalid")
    if artifact.artifact_id != f"aqse-encoder-{expected_digest[:16]}":
        raise ValueError("encoder artifact identity is invalid")


def validate_input_contract(
    artifact: EncoderScientificArtifact, context: EncodingInputContract
) -> None:
    expected = {
        "source_dataset_id": artifact.source_dataset_id,
        "source_dataset_digest": artifact.source_dataset_digest,
        "feature_profile_fingerprint": artifact.feature_profile_fingerprint,
        "feature_profile": artifact.feature_profile,
        "phase_index": artifact.phase_index,
        "phase_reference": artifact.phase_reference,
        "encoding_policy_id": artifact.encoding_policy_id,
        "scaler_id": artifact.scaler_id,
        "tqk_compatibility": artifact.tqk_compatibility.model_dump(mode="json"),
    }
    if context.model_dump(mode="json") != expected:
        raise ValueError("input is incompatible with the frozen phase-direct encoder space")
    profile = FeatureProfile.model_validate(context.feature_profile)
    if _digest(profile.model_dump(mode="json")) != context.feature_profile_fingerprint:
        raise ValueError("input feature-profile fingerprint is internally inconsistent")


@dataclass(frozen=True)
class PhaseDirectEncoder:
    artifact: EncoderScientificArtifact
    scaler: AngleScaler

    def transform(
        self,
        raw_features: NDArray[Any],
        *,
        context: EncodingInputContract,
    ) -> NDArray[np.float64]:
        validate_encoder_artifact(self.artifact)
        validate_input_contract(self.artifact, context)
        values = np.asarray(raw_features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != FEATURE_COUNT or len(values) == 0:
            raise ValueError("phase-direct features must have shape (N, 8)")
        if not np.isfinite(values).all():
            raise ValueError("phase-direct features contain NaN or infinity")
        mean_before = self.scaler.mean.copy()
        scale_before = self.scaler.scale.copy()
        encoded = np.asarray(self.scaler.transform(values), dtype=np.float64)
        encoded[:, PHASE_INDEX] = wrap_phase_direct(values[:, PHASE_INDEX])
        if not np.array_equal(self.scaler.mean, mean_before) or not np.array_equal(
            self.scaler.scale, scale_before
        ):
            raise RuntimeError("protected AngleScaler.transform mutated fitted statistics")
        if encoded.shape != values.shape or not np.isfinite(encoded).all():
            raise ValueError("phase-direct encoding produced an invalid output")
        return encoded


def input_contract_for(artifact: EncoderScientificArtifact) -> EncodingInputContract:
    return EncodingInputContract(
        source_dataset_id=artifact.source_dataset_id,
        source_dataset_digest=artifact.source_dataset_digest,
        feature_profile_fingerprint=artifact.feature_profile_fingerprint,
        feature_profile=artifact.feature_profile,
        phase_index=artifact.phase_index,
        phase_reference=artifact.phase_reference,
        encoding_policy_id=artifact.encoding_policy_id,
        scaler_id=artifact.scaler_id,
        tqk_compatibility=artifact.tqk_compatibility,
    )


def fit_phase_direct_encoder(
    observations: ObservablePartition,
    manifest: DatasetManifest,
) -> PhaseDirectEncoder:
    if observations.partition is not DatasetPartition.TRAIN:
        raise ValueError("AngleScaler fitting is permitted on TRAIN only")
    if manifest.dataset_id != observations.dataset_id:
        raise ValueError("manifest and observable dataset identity differ")
    if manifest.dataset_id != CANONICAL_DEVELOPMENT_DATASET_ID:
        raise ValueError("1D.2 requires the explicit canonical development dataset")
    if manifest.scientific_digest != CANONICAL_DEVELOPMENT_DIGEST:
        raise ValueError("canonical development scientific digest mismatch")
    if manifest.feature_profile_fingerprint != CANONICAL_FEATURE_PROFILE_FINGERPRINT:
        raise ValueError("canonical feature-profile fingerprint mismatch")
    profile_payload = observations.profile.model_dump(mode="json")
    if _digest(profile_payload) != manifest.feature_profile_fingerprint:
        raise ValueError("TRAIN feature profile is incompatible with the manifest")
    accepted = observations.features[observations.valid_mask]
    accepted = np.asarray(accepted, dtype=np.float64)
    if accepted.ndim != 2 or accepted.shape[1] != FEATURE_COUNT or not len(accepted):
        raise ValueError("TRAIN fitting population must contain eligible 8D windows")
    episode_count = sum(bool(np.any(mask)) for mask in observations.valid_mask)
    row_references = [
        observations.windows[episode_index][window_index].window_id
        for episode_index in range(len(observations.episode_ids))
        for window_index, valid in enumerate(observations.valid_mask[episode_index])
        if bool(valid)
    ]
    population_payload = {
        "schema_version": "aqse.fitting-population.v1",
        "dataset_id": manifest.dataset_id,
        "dataset_digest": manifest.scientific_digest,
        "partition": DatasetPartition.TRAIN.value,
        "episode_ids": list(observations.episode_ids),
        "eligible_window_ids": row_references,
        "feature_content_identity": array_identity(accepted),
    }
    population_digest = _digest(population_payload)
    population = FittingPopulation(
        population_id=f"aqse-fitting-population-{population_digest[:16]}",
        content_digest=population_digest,
        episode_count=episode_count,
        accepted_window_count=len(accepted),
    )
    protected_scaler = AngleScaler.fit(accepted)
    mean = np.asarray(protected_scaler.mean, dtype=np.float64)
    scale = np.asarray(protected_scaler.scale, dtype=np.float64)
    scaler_payload = {
        "algorithm": "protected.AngleScaler.fit.v1",
        "source_sha256": TQK8_SOURCE_SHA256,
        "fitting_population_id": population.population_id,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    scaler_id = f"aqse-angle-scaler-{_digest(scaler_payload)[:16]}"
    scientific = {
        "schema_version": "aqse.training-encoder.v1",
        "encoding_policy_id": ENCODING_POLICY_ID,
        "source_dataset_id": manifest.dataset_id,
        "source_dataset_digest": manifest.scientific_digest,
        "source_partition": DatasetPartition.TRAIN.value,
        "feature_profile_fingerprint": manifest.feature_profile_fingerprint,
        "feature_profile": profile_payload,
        "feature_order": list(observations.profile.feature_names),
        "feature_units": list(observations.profile.feature_units),
        "phase_index": PHASE_INDEX,
        "phase_reference": observations.profile.phase_reference,
        "phase_wrap_convention": PHASE_WRAP_CONVENTION,
        "scaler_algorithm": "protected.AngleScaler.fit.v1",
        "scaler_source_sha256": TQK8_SOURCE_SHA256,
        "scaler_id": scaler_id,
        "fitting_population": population.model_dump(mode="json"),
        "fitted_mean": mean.tolist(),
        "fitted_scale": scale.tolist(),
        "phase_scaler_statistics_used": False,
        "tqk_compatibility": TQKCompatibility(
            tqk8_source_sha256=TQK8_SOURCE_SHA256
        ).model_dump(mode="json"),
        "effective_source_hashes": _effective_source_hashes(),
    }
    content_digest = _digest(scientific)
    artifact = EncoderScientificArtifact.model_validate(
        {
            **scientific,
            "artifact_id": f"aqse-encoder-{content_digest[:16]}",
            "content_digest": content_digest,
        }
    )
    validate_encoder_artifact(artifact)
    frozen_mean = mean.copy()
    frozen_scale = scale.copy()
    frozen_mean.setflags(write=False)
    frozen_scale.setflags(write=False)
    return PhaseDirectEncoder(
        artifact=artifact,
        scaler=AngleScaler(mean=frozen_mean, scale=frozen_scale),
    )
