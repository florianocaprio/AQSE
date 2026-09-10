from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from app.classical.observable_rule import (
    ObservableRule,
    ObservableRuleArtifact,
    ObservableRuleQuery,
    fit_observable_rule,
    observable_rule_query,
    validate_observable_rule_artifact,
)
from app.features.state8 import state8_profile, state8_profile_fingerprint
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
)

DATASET_DIGEST = "a" * 64
SOURCE_HASHES = {"fixture.py": "b" * 64}


def _training_rows() -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    features = np.asarray(
        [
            [-1.0, 0.10, -0.20, 0.0, 0.0, 0.30, 0.0, 1.0],
            [2.0, 0.20, 0.10, 0.0, 0.0, 0.40, 0.0, 1.0],
            [-3.0, 0.30, 0.30, 0.0, 0.0, 0.50, 0.0, 1.0],
            [500.0, 50.0, 50.0, 0.0, 0.0, 50.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = ("NORMAL", "NORMAL", "NORMAL", "DEVICE_COMPATIBLE")
    sample_ids = ("normal-c", "normal-a", "normal-b", "changed")
    return features, labels, sample_ids


def _fit(profile_id: str = NETWORK_STATE8_PROFILE_ID) -> ObservableRule:
    features, labels, sample_ids = _training_rows()
    return fit_observable_rule(
        features,
        labels,
        sample_ids=sample_ids,
        profile=state8_profile(profile_id),
        fitted_on_dataset_id="network-study-fixture",
        fitted_on_dataset_digest=DATASET_DIGEST,
        effective_source_hashes=SOURCE_HASHES,
    )


def _query(rule: ObservableRule, values: tuple[float, ...]) -> ObservableRuleQuery:
    return observable_rule_query(
        rule.artifact,
        sample_id="query-1",
        feature_values=values,
    )


def test_fit_uses_only_normal_train_rows_and_exact_linear_p99() -> None:
    rule = _fit()
    artifact = rule.artifact

    assert artifact.fitted_on_partition == "TRAIN"
    assert artifact.normal_sample_ids == ("normal-a", "normal-b", "normal-c")
    assert artifact.train_sample_count == 4
    assert artifact.percentile == 0.99
    assert artifact.quantile_method == "linear"
    assert artifact.thresholds.absolute_mean_north_anomaly == pytest.approx(2.98)
    assert artifact.thresholds.north_anomaly_mad == pytest.approx(0.298)
    assert artifact.thresholds.absolute_north_anomaly_slope == pytest.approx(0.298)
    assert artifact.thresholds.peer_median_residual_rms == pytest.approx(0.498)
    assert (
        max(
            artifact.thresholds.absolute_mean_north_anomaly,
            artifact.thresholds.north_anomaly_mad,
            artifact.thresholds.absolute_north_anomaly_slope,
            artifact.thresholds.peer_median_residual_rms or 0.0,
        )
        < 4.0
    )


def test_fit_is_deterministic_under_train_row_reordering() -> None:
    features, labels, sample_ids = _training_rows()
    order = (3, 1, 0, 2)
    first = _fit()
    second = fit_observable_rule(
        features[list(order)],
        tuple(labels[index] for index in order),
        sample_ids=tuple(sample_ids[index] for index in order),
        profile=state8_profile(NETWORK_STATE8_PROFILE_ID),
        fitted_on_dataset_id="network-study-fixture",
        fitted_on_dataset_digest=DATASET_DIGEST,
        effective_source_hashes=SOURCE_HASHES,
    )

    assert first.artifact == second.artifact


@pytest.mark.parametrize(
    ("values", "expected_status", "expected_triggers"),
    [
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "NO_OBSERVED_CHANGE",
            (),
        ),
        (
            (4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "COMMON_CHANGE_AMBIGUOUS",
            ("absolute_mean_north_anomaly",),
        ),
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 1.0),
            "SPATIAL_DISAGREEMENT_AMBIGUOUS",
            ("peer_median_residual_rms",),
        ),
        (
            (0.0, 0.4, -0.4, 0.0, 0.0, 0.0, 0.0, 1.0),
            "MIXED_OBSERVABLE_CHANGE",
            ("north_anomaly_mad", "absolute_north_anomaly_slope"),
        ),
    ],
)
def test_rule_reports_only_non_causal_observable_statuses(
    values: tuple[float, ...],
    expected_status: str,
    expected_triggers: tuple[str, ...],
) -> None:
    rule = _fit()
    result = rule.evaluate(_query(rule, values))

    assert result.status == expected_status
    assert result.triggered_observables == expected_triggers
    assert result.interpretation.endswith("ambiguous and non-causal")


def test_local_rule_does_not_fit_or_use_feature_five_as_spatial_residual() -> None:
    rule = _fit(LOCAL_STATE8_PROFILE_ID)
    assert rule.artifact.thresholds.peer_median_residual_rms is None

    result = rule.evaluate(_query(rule, (0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 1.0)))
    assert result.status == "NO_OBSERVED_CHANGE"
    assert "peer_median_residual_rms" not in result.triggered_observables


def test_json_round_trip_is_non_executable_and_evaluation_is_stable() -> None:
    fitted = _fit()
    payload = fitted.artifact.model_dump_json()
    reloaded_artifact = ObservableRuleArtifact.model_validate_json(payload)
    reloaded = ObservableRule.from_artifact(reloaded_artifact)
    query = _query(reloaded, (4.0, 0.4, 0.4, 0.0, 0.0, 0.7, 0.0, 1.0))

    assert reloaded.evaluate(query) == fitted.evaluate(query)
    assert reloaded_artifact.persistence_format.endswith("no-pickle")


def test_query_schema_forbids_supervisory_fields_and_rejects_profile_mismatch() -> None:
    rule = _fit()
    valid = _query(rule, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    payload = valid.model_dump(mode="json")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ObservableRuleQuery.model_validate({**payload, "label": "NORMAL"})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ObservableRuleQuery.model_validate({**payload, "scenario": "NORMAL"})

    local_profile = state8_profile(LOCAL_STATE8_PROFILE_ID)
    incompatible = ObservableRuleQuery(
        sample_id="query-local",
        profile_id=local_profile.profile_id,
        profile_fingerprint=state8_profile_fingerprint(local_profile),
        feature_order=local_profile.feature_names,
        feature_values=valid.feature_values,
    )
    with pytest.raises(ValueError, match="incompatible"):
        rule.evaluate(incompatible)


def test_invalid_inputs_corrupted_identity_and_noncanonical_profile_are_rejected() -> None:
    features, labels, sample_ids = _training_rows()
    features[0, 1] = np.nan
    with pytest.raises(ValueError, match="no imputation"):
        fit_observable_rule(
            features,
            labels,
            sample_ids=sample_ids,
            profile=state8_profile(NETWORK_STATE8_PROFILE_ID),
            fitted_on_dataset_id="network-study-fixture",
            fitted_on_dataset_digest=DATASET_DIGEST,
            effective_source_hashes=SOURCE_HASHES,
        )

    rule = _fit()
    with pytest.raises(ValueError, match="content digest"):
        validate_observable_rule_artifact(
            rule.artifact.model_copy(update={"content_digest": "0" * 64})
        )
    with pytest.raises(ValidationError, match="fingerprint"):
        ObservableRuleArtifact.model_validate(
            {
                **rule.artifact.model_dump(mode="json"),
                "profile_fingerprint": "0" * 64,
            }
        )

    no_normal = tuple("DEVICE_COMPATIBLE" for _ in labels)
    with pytest.raises(ValueError, match="at least one NORMAL"):
        fit_observable_rule(
            _training_rows()[0],
            no_normal,
            sample_ids=sample_ids,
            profile=state8_profile(NETWORK_STATE8_PROFILE_ID),
            fitted_on_dataset_id="network-study-fixture",
            fitted_on_dataset_digest=DATASET_DIGEST,
            effective_source_hashes=SOURCE_HASHES,
        )
