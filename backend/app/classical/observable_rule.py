from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.features.state8 import (
    require_state8_profile,
    state8_profile,
    state8_profile_fingerprint,
)
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
    State8FeatureProfile,
    State8ProfileId,
)
from app.training.canonical import canonical_json_bytes, file_sha256

OBSERVABLE_RULE_ID = "aqse.classical.observable-dispersion-rule.v1"
OBSERVABLE_RULE_PERCENTILE = 0.99

ObservableRuleStatus = Literal[
    "NO_OBSERVED_CHANGE",
    "COMMON_CHANGE_AMBIGUOUS",
    "SPATIAL_DISAGREEMENT_AMBIGUOUS",
    "MIXED_OBSERVABLE_CHANGE",
]
TriggeredObservable = Literal[
    "absolute_mean_north_anomaly",
    "north_anomaly_mad",
    "absolute_north_anomaly_slope",
    "peer_median_residual_rms",
]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
CompleteState8Values = tuple[
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
]


class FrozenObservableRuleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ObservableRuleThresholds(FrozenObservableRuleModel):
    absolute_mean_north_anomaly: float = Field(ge=0.0)
    north_anomaly_mad: float = Field(ge=0.0)
    absolute_north_anomaly_slope: float = Field(ge=0.0)
    peer_median_residual_rms: float | None = Field(default=None, ge=0.0)


class ObservableRuleArtifact(FrozenObservableRuleModel):
    """TRAIN-only thresholds for a deliberately non-causal observable rule."""

    schema_version: Literal["aqse.observable-rule-artifact.v1"] = "aqse.observable-rule-artifact.v1"
    artifact_id: str = Field(pattern=r"^aqse-observable-rule-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    rule_id: Literal["aqse.classical.observable-dispersion-rule.v1"] = OBSERVABLE_RULE_ID
    profile_id: State8ProfileId
    profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_order: tuple[str, ...]
    fitted_on_partition: Literal["TRAIN"] = "TRAIN"
    fitted_on_dataset_id: str = Field(min_length=1)
    fitted_on_dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    normal_class: Literal["NORMAL"] = "NORMAL"
    normal_sample_ids: tuple[str, ...] = Field(min_length=1)
    train_sample_count: int = Field(gt=0)
    percentile: Literal[0.99] = OBSERVABLE_RULE_PERCENTILE
    quantile_method: Literal["linear"] = "linear"
    thresholds: ObservableRuleThresholds
    decision_semantics: Literal["observable-threshold-heuristic;non-causal;not-a-model-score"] = (
        "observable-threshold-heuristic;non-causal;not-a-model-score"
    )
    persistence_format: Literal["bounded-json-numeric-values;no-pickle"] = (
        "bounded-json-numeric-values;no-pickle"
    )
    effective_source_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_profile_and_thresholds(self) -> ObservableRuleArtifact:
        profile = state8_profile(self.profile_id)
        if self.profile_fingerprint != state8_profile_fingerprint(profile):
            raise ValueError("observable rule profile fingerprint is incompatible")
        if self.feature_order != profile.feature_names:
            raise ValueError("observable rule feature order is incompatible")
        if tuple(sorted(self.normal_sample_ids)) != self.normal_sample_ids:
            raise ValueError("NORMAL sample identifiers must use canonical sorted order")
        if len(set(self.normal_sample_ids)) != len(self.normal_sample_ids):
            raise ValueError("NORMAL sample identifiers must be distinct")
        if len(self.normal_sample_ids) > self.train_sample_count:
            raise ValueError("NORMAL sample count cannot exceed TRAIN sample count")
        spatial_threshold = self.thresholds.peer_median_residual_rms
        if self.profile_id == LOCAL_STATE8_PROFILE_ID and spatial_threshold is not None:
            raise ValueError("local observable rules cannot fit a spatial residual")
        if self.profile_id == NETWORK_STATE8_PROFILE_ID and spatial_threshold is None:
            raise ValueError("network observable rules require a spatial residual threshold")
        if not self.effective_source_hashes:
            raise ValueError("observable rule source provenance is required")
        return self


class ObservableRuleQuery(FrozenObservableRuleModel):
    """Observable-only query. Supervisory labels and scenarios are forbidden extras."""

    schema_version: Literal["aqse.observable-rule-query.v1"] = "aqse.observable-rule-query.v1"
    sample_id: str = Field(min_length=1)
    profile_id: State8ProfileId
    profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_order: tuple[str, ...]
    feature_values: CompleteState8Values

    @model_validator(mode="after")
    def validate_observable_contract(self) -> ObservableRuleQuery:
        profile = state8_profile(self.profile_id)
        if self.profile_fingerprint != state8_profile_fingerprint(profile):
            raise ValueError("observable query profile fingerprint is incompatible")
        if self.feature_order != profile.feature_names:
            raise ValueError("observable query feature order is incompatible")
        if self.feature_values[1] < 0.0:
            raise ValueError("observable dispersion cannot be negative")
        if self.profile_id == NETWORK_STATE8_PROFILE_ID and self.feature_values[5] < 0.0:
            raise ValueError("observable spatial residual cannot be negative")
        return self


class ObservableRuleResult(FrozenObservableRuleModel):
    schema_version: Literal["aqse.observable-rule-result.v1"] = "aqse.observable-rule-result.v1"
    sample_id: str
    artifact_id: str
    profile_id: State8ProfileId
    status: ObservableRuleStatus
    common_change_observed: bool
    dispersion_or_spatial_residual_observed: bool
    triggered_observables: tuple[TriggeredObservable, ...]
    interpretation: Literal["observable engineering heuristic;ambiguous and non-causal"] = (
        "observable engineering heuristic;ambiguous and non-causal"
    )

    @model_validator(mode="after")
    def validate_status_flags(self) -> ObservableRuleResult:
        expected: ObservableRuleStatus
        if self.common_change_observed and self.dispersion_or_spatial_residual_observed:
            expected = "MIXED_OBSERVABLE_CHANGE"
        elif self.common_change_observed:
            expected = "COMMON_CHANGE_AMBIGUOUS"
        elif self.dispersion_or_spatial_residual_observed:
            expected = "SPATIAL_DISAGREEMENT_AMBIGUOUS"
        else:
            expected = "NO_OBSERVED_CHANGE"
        if self.status != expected:
            raise ValueError("observable rule status is inconsistent with its flags")
        return self


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _effective_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "app/classical/observable_rule.py",
        "app/features/state8.py",
        "app/features/state8_models.py",
    )
    return {relative: file_sha256(root / relative) for relative in relative_paths}


def _as_feature_matrix(
    features: NDArray[Any] | Sequence[Sequence[float]],
) -> NDArray[np.float64]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != 8:
        raise ValueError("observable rule requires a non-empty State8 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(
            "observable rule features contain NaN or infinity; no imputation is allowed"
        )
    if np.any(matrix[:, 1] < 0.0):
        raise ValueError("observable dispersion cannot be negative")
    return np.asarray(matrix, dtype=np.float64)


def _artifact_digest_payload(artifact: ObservableRuleArtifact) -> dict[str, Any]:
    return artifact.model_dump(mode="json", exclude={"artifact_id", "content_digest"})


def validate_observable_rule_artifact(artifact: ObservableRuleArtifact) -> None:
    expected_digest = _digest(_artifact_digest_payload(artifact))
    if artifact.content_digest != expected_digest:
        raise ValueError("observable rule scientific content digest is invalid")
    if artifact.artifact_id != f"aqse-observable-rule-{expected_digest[:16]}":
        raise ValueError("observable rule artifact identity is invalid")
    if any(not _is_sha256(value) for value in artifact.effective_source_hashes.values()):
        raise ValueError("observable rule source provenance contains an invalid SHA-256")


def observable_rule_query(
    artifact: ObservableRuleArtifact,
    *,
    sample_id: str,
    feature_values: Sequence[float],
) -> ObservableRuleQuery:
    """Construct an observable-only query compatible with a frozen rule."""

    return ObservableRuleQuery(
        sample_id=sample_id,
        profile_id=artifact.profile_id,
        profile_fingerprint=artifact.profile_fingerprint,
        feature_order=artifact.feature_order,
        feature_values=tuple(feature_values),
    )


@dataclass(frozen=True)
class ObservableRule:
    artifact: ObservableRuleArtifact

    @classmethod
    def from_artifact(cls, artifact: ObservableRuleArtifact) -> ObservableRule:
        validate_observable_rule_artifact(artifact)
        return cls(artifact=artifact)

    def evaluate(self, query: ObservableRuleQuery) -> ObservableRuleResult:
        validate_observable_rule_artifact(self.artifact)
        if (
            query.profile_id != self.artifact.profile_id
            or query.profile_fingerprint != self.artifact.profile_fingerprint
            or query.feature_order != self.artifact.feature_order
        ):
            raise ValueError("observable query is incompatible with the frozen rule")

        values = query.feature_values
        thresholds = self.artifact.thresholds
        mean_triggered = abs(values[0]) > thresholds.absolute_mean_north_anomaly
        dispersion_triggered = values[1] > thresholds.north_anomaly_mad
        slope_triggered = abs(values[2]) > thresholds.absolute_north_anomaly_slope
        residual_triggered = (
            thresholds.peer_median_residual_rms is not None
            and values[5] > thresholds.peer_median_residual_rms
        )
        common_change = mean_triggered or slope_triggered
        disagreement = dispersion_triggered or residual_triggered

        triggered: list[TriggeredObservable] = []
        if mean_triggered:
            triggered.append("absolute_mean_north_anomaly")
        if dispersion_triggered:
            triggered.append("north_anomaly_mad")
        if slope_triggered:
            triggered.append("absolute_north_anomaly_slope")
        if residual_triggered:
            triggered.append("peer_median_residual_rms")

        status: ObservableRuleStatus
        if common_change and disagreement:
            status = "MIXED_OBSERVABLE_CHANGE"
        elif common_change:
            status = "COMMON_CHANGE_AMBIGUOUS"
        elif disagreement:
            status = "SPATIAL_DISAGREEMENT_AMBIGUOUS"
        else:
            status = "NO_OBSERVED_CHANGE"
        return ObservableRuleResult(
            sample_id=query.sample_id,
            artifact_id=self.artifact.artifact_id,
            profile_id=self.artifact.profile_id,
            status=status,
            common_change_observed=common_change,
            dispersion_or_spatial_residual_observed=disagreement,
            triggered_observables=tuple(triggered),
        )


def fit_observable_rule(
    train_features: NDArray[Any] | Sequence[Sequence[float]],
    train_classes: Sequence[str],
    *,
    sample_ids: Sequence[str],
    profile: State8FeatureProfile,
    fitted_on_dataset_id: str,
    fitted_on_dataset_digest: str,
    effective_source_hashes: Mapping[str, str] | None = None,
) -> ObservableRule:
    """Fit fixed empirical p99 thresholds using only NORMAL rows from TRAIN."""

    require_state8_profile(profile, profile.profile_id)
    features = _as_feature_matrix(train_features)
    labels = tuple(train_classes)
    identifiers = tuple(sample_ids)
    if len(features) != len(labels) or len(features) != len(identifiers):
        raise ValueError("TRAIN features, classes and sample IDs must have equal lengths")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("TRAIN sample identifiers must be distinct")
    if not _is_sha256(fitted_on_dataset_digest):
        raise ValueError("observable rule fitted dataset digest must be a SHA-256")
    normal_indices = np.asarray(
        [index for index, label in enumerate(labels) if label == "NORMAL"],
        dtype=np.int64,
    )
    if len(normal_indices) == 0:
        raise ValueError("observable rule fitting requires at least one NORMAL TRAIN row")
    normal = features[normal_indices]
    if profile.profile_id == NETWORK_STATE8_PROFILE_ID and np.any(normal[:, 5] < 0.0):
        raise ValueError("observable spatial residual cannot be negative")

    percentile = OBSERVABLE_RULE_PERCENTILE
    thresholds = ObservableRuleThresholds(
        absolute_mean_north_anomaly=float(
            np.quantile(np.abs(normal[:, 0]), percentile, method="linear")
        ),
        north_anomaly_mad=float(np.quantile(normal[:, 1], percentile, method="linear")),
        absolute_north_anomaly_slope=float(
            np.quantile(np.abs(normal[:, 2]), percentile, method="linear")
        ),
        peer_median_residual_rms=(
            float(np.quantile(normal[:, 5], percentile, method="linear"))
            if profile.profile_id == NETWORK_STATE8_PROFILE_ID
            else None
        ),
    )
    normal_sample_ids = tuple(sorted(identifiers[index] for index in normal_indices))
    source_hashes = dict(effective_source_hashes or _effective_source_hashes())
    scientific: dict[str, Any] = {
        "schema_version": "aqse.observable-rule-artifact.v1",
        "rule_id": OBSERVABLE_RULE_ID,
        "profile_id": profile.profile_id,
        "profile_fingerprint": state8_profile_fingerprint(profile),
        "feature_order": list(profile.feature_names),
        "fitted_on_partition": "TRAIN",
        "fitted_on_dataset_id": fitted_on_dataset_id,
        "fitted_on_dataset_digest": fitted_on_dataset_digest,
        "normal_class": "NORMAL",
        "normal_sample_ids": list(normal_sample_ids),
        "train_sample_count": len(features),
        "percentile": percentile,
        "quantile_method": "linear",
        "thresholds": thresholds.model_dump(mode="json"),
        "decision_semantics": ("observable-threshold-heuristic;non-causal;not-a-model-score"),
        "persistence_format": "bounded-json-numeric-values;no-pickle",
        "effective_source_hashes": source_hashes,
    }
    content_digest = _digest(scientific)
    artifact = ObservableRuleArtifact.model_validate(
        {
            **scientific,
            "artifact_id": f"aqse-observable-rule-{content_digest[:16]}",
            "content_digest": content_digest,
        }
    )
    return ObservableRule.from_artifact(artifact)
