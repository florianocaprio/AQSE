from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from app.features.state8 import require_state8_profile, state8_profile_fingerprint
from app.features.state8_models import (
    FrozenState8Model,
    State8FeatureProfile,
    State8FeatureRecord,
    State8ProfileId,
)
from app.quantum.user_pipeline.tqk8 import AngleScaler

STATE8_ENCODING_POLICY_ID = "aqse.state8.angle-scaler-all8.v1"
PROTECTED_TQK8_SOURCE_SHA256 = (
    "cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689"
)
STATE8_FEATURE_COUNT = 8


def _canonical_digest(value: Any) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _current_tqk8_source_hash() -> str:
    path = Path(__file__).resolve().parents[1] / "quantum/user_pipeline/tqk8.py"
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class State8EncodingArtifact(FrozenState8Model):
    schema_version: Literal["aqse.state8-encoding-artifact.v1"] = (
        "aqse.state8-encoding-artifact.v1"
    )
    encoder_id: str = Field(pattern=r"^aqse-state8-encoder-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    encoding_policy_id: Literal["aqse.state8.angle-scaler-all8.v1"] = (
        STATE8_ENCODING_POLICY_ID
    )
    source_partition: Literal["train"] = "train"
    training_identity: str = Field(min_length=1, max_length=256)
    profile_id: State8ProfileId
    profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    profile: dict[str, Any]
    feature_order: tuple[str, ...]
    feature_units: tuple[str, ...]
    fit_row_count: int = Field(gt=0)
    scaler_algorithm: Literal["protected.AngleScaler.fit.v1"] = (
        "protected.AngleScaler.fit.v1"
    )
    scaler_source_sha256: Literal[
        "cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689"
    ] = PROTECTED_TQK8_SOURCE_SHA256
    scaler_id: str = Field(pattern=r"^aqse-state8-angle-scaler-[a-f0-9]{16}$")
    fitted_mean: tuple[float, float, float, float, float, float, float, float]
    fitted_scale: tuple[float, float, float, float, float, float, float, float]
    all_coordinates_scaled: Literal[True] = True

    @model_validator(mode="after")
    def validate_scaler_contract(self) -> State8EncodingArtifact:
        if len(self.feature_order) != STATE8_FEATURE_COUNT:
            raise ValueError("state8 encoder requires exactly eight ordered features")
        if len(self.feature_units) != STATE8_FEATURE_COUNT:
            raise ValueError("state8 encoder requires exactly eight feature units")
        if any(value <= 0.0 for value in self.fitted_scale):
            raise ValueError("state8 fitted scales must be positive")
        return self


def _artifact_payload(artifact: State8EncodingArtifact) -> dict[str, Any]:
    return artifact.model_dump(mode="json", exclude={"encoder_id", "content_digest"})


def validate_state8_encoder_artifact(artifact: State8EncodingArtifact) -> None:
    if _current_tqk8_source_hash() != PROTECTED_TQK8_SOURCE_SHA256:
        raise ValueError("protected TQK8/AngleScaler source identity has changed")
    profile = State8FeatureProfile.model_validate(artifact.profile)
    require_state8_profile(profile, artifact.profile_id)
    if state8_profile_fingerprint(profile) != artifact.profile_fingerprint:
        raise ValueError("state8 encoder profile fingerprint is invalid")
    if profile.feature_names != artifact.feature_order:
        raise ValueError("state8 encoder feature order differs from its profile")
    if profile.feature_units != artifact.feature_units:
        raise ValueError("state8 encoder feature units differ from its profile")
    scaler_payload = {
        "algorithm": artifact.scaler_algorithm,
        "source_sha256": artifact.scaler_source_sha256,
        "training_identity": artifact.training_identity,
        "profile_fingerprint": artifact.profile_fingerprint,
        "fit_row_count": artifact.fit_row_count,
        "mean": list(artifact.fitted_mean),
        "scale": list(artifact.fitted_scale),
        "all_coordinates_scaled": True,
    }
    expected_scaler_id = (
        f"aqse-state8-angle-scaler-{_canonical_digest(scaler_payload)[:16]}"
    )
    if artifact.scaler_id != expected_scaler_id:
        raise ValueError("state8 scaler identity is invalid")
    expected_digest = _canonical_digest(_artifact_payload(artifact))
    if artifact.content_digest != expected_digest:
        raise ValueError("state8 encoder content digest is invalid")
    if artifact.encoder_id != f"aqse-state8-encoder-{expected_digest[:16]}":
        raise ValueError("state8 encoder identity is invalid")


def _matrix(
    features: NDArray[Any] | Sequence[State8FeatureRecord],
    *,
    profile: State8FeatureProfile,
) -> NDArray[np.float64]:
    if isinstance(features, np.ndarray):
        matrix = np.asarray(features, dtype=np.float64)
    else:
        rows = tuple(features)
        expected_fingerprint = state8_profile_fingerprint(profile)
        for row in rows:
            if row.profile_id != profile.profile_id:
                raise ValueError("state8 record belongs to an incompatible profile")
            if row.profile_fingerprint != expected_fingerprint:
                raise ValueError("state8 record profile fingerprint is incompatible")
            if not row.quality.valid_for_quantum or any(
                value is None for value in row.values
            ):
                raise ValueError("only eligible complete state8 records can be encoded")
        matrix = np.asarray([row.values for row in rows], dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != STATE8_FEATURE_COUNT
        or len(matrix) == 0
    ):
        raise ValueError("state8 features must have shape (N, 8)")
    if not np.isfinite(matrix).all():
        raise ValueError("state8 features contain NaN or infinity")
    return matrix


@dataclass(frozen=True)
class State8AngleEncoder:
    artifact: State8EncodingArtifact
    scaler: AngleScaler

    def transform(
        self,
        features: NDArray[Any] | Sequence[State8FeatureRecord],
        *,
        profile: State8FeatureProfile,
    ) -> NDArray[np.float64]:
        validate_state8_encoder_artifact(self.artifact)
        require_state8_profile(profile, self.artifact.profile_id)
        if state8_profile_fingerprint(profile) != self.artifact.profile_fingerprint:
            raise ValueError("query state8 profile differs from the fitted encoder")
        matrix = _matrix(features, profile=profile)
        mean_before = np.asarray(self.scaler.mean, dtype=np.float64).copy()
        scale_before = np.asarray(self.scaler.scale, dtype=np.float64).copy()
        encoded = np.asarray(self.scaler.transform(matrix), dtype=np.float64)
        if not np.array_equal(self.scaler.mean, mean_before) or not np.array_equal(
            self.scaler.scale, scale_before
        ):
            raise RuntimeError("protected AngleScaler.transform mutated fitted statistics")
        if encoded.shape != matrix.shape or not np.isfinite(encoded).all():
            raise ValueError("state8 angle encoding produced an invalid output")
        encoded.setflags(write=False)
        return encoded


def load_state8_encoder(artifact: State8EncodingArtifact) -> State8AngleEncoder:
    """Rehydrate a validated non-executable scaler snapshot from numeric values."""

    validate_state8_encoder_artifact(artifact)
    mean = np.asarray(artifact.fitted_mean, dtype=np.float64)
    scale = np.asarray(artifact.fitted_scale, dtype=np.float64)
    mean.setflags(write=False)
    scale.setflags(write=False)
    return State8AngleEncoder(
        artifact=artifact,
        scaler=AngleScaler(mean=mean, scale=scale),
    )


def fit_state8_encoder(
    training_features: NDArray[Any] | Sequence[State8FeatureRecord],
    *,
    profile: State8FeatureProfile,
    partition: str,
    training_identity: str,
) -> State8AngleEncoder:
    """Fit the protected scaler once on TRAIN and apply it to all eight values."""

    if partition != "train":
        raise ValueError("state8 AngleScaler fitting is permitted on TRAIN only")
    if not training_identity or len(training_identity) > 256:
        raise ValueError("a bounded non-empty TRAIN identity is required")
    require_state8_profile(profile, profile.profile_id)
    if _current_tqk8_source_hash() != PROTECTED_TQK8_SOURCE_SHA256:
        raise ValueError("protected TQK8/AngleScaler source identity has changed")
    matrix = _matrix(training_features, profile=profile)
    protected_scaler = AngleScaler.fit(matrix)
    mean = np.asarray(protected_scaler.mean, dtype=np.float64).copy()
    scale = np.asarray(protected_scaler.scale, dtype=np.float64).copy()
    profile_fingerprint = state8_profile_fingerprint(profile)
    scaler_payload = {
        "algorithm": "protected.AngleScaler.fit.v1",
        "source_sha256": PROTECTED_TQK8_SOURCE_SHA256,
        "training_identity": training_identity,
        "profile_fingerprint": profile_fingerprint,
        "fit_row_count": len(matrix),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "all_coordinates_scaled": True,
    }
    scaler_id = f"aqse-state8-angle-scaler-{_canonical_digest(scaler_payload)[:16]}"
    scientific = {
        "schema_version": "aqse.state8-encoding-artifact.v1",
        "encoding_policy_id": STATE8_ENCODING_POLICY_ID,
        "source_partition": "train",
        "training_identity": training_identity,
        "profile_id": profile.profile_id,
        "profile_fingerprint": profile_fingerprint,
        "profile": profile.model_dump(mode="json"),
        "feature_order": list(profile.feature_names),
        "feature_units": list(profile.feature_units),
        "fit_row_count": len(matrix),
        "scaler_algorithm": "protected.AngleScaler.fit.v1",
        "scaler_source_sha256": PROTECTED_TQK8_SOURCE_SHA256,
        "scaler_id": scaler_id,
        "fitted_mean": mean.tolist(),
        "fitted_scale": scale.tolist(),
        "all_coordinates_scaled": True,
    }
    content_digest = _canonical_digest(scientific)
    artifact = State8EncodingArtifact.model_validate(
        {
            **scientific,
            "encoder_id": f"aqse-state8-encoder-{content_digest[:16]}",
            "content_digest": content_digest,
        }
    )
    validate_state8_encoder_artifact(artifact)
    mean.setflags(write=False)
    scale.setflags(write=False)
    return State8AngleEncoder(
        artifact=artifact,
        scaler=AngleScaler(
            mean=cast(NDArray[np.float64], mean),
            scale=cast(NDArray[np.float64], scale),
        ),
    )
