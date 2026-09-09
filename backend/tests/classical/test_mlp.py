from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from app.classical.mlp import (
    NumpyMLPClassifier,
    fit_mlp_classifier,
    fit_raw_feature_baseline,
    query_context_for,
    uncertainty_gate,
    validate_mlp_artifact,
)
from app.classical.models import MLPClassifierArtifact

DATASET_DIGEST = "c" * 64
SOURCE_HASHES = {"fixture.py": "d" * 64}


def _training_rows(
    *,
    class_count: int = 4,
    rows_per_class: int = 14,
    feature_count: int = 8,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    generator = np.random.default_rng(7)
    classes = tuple(f"CLASS_{index}" for index in range(class_count))
    labels = tuple(label for label in classes for _ in range(rows_per_class))
    centers = np.linspace(-1.5, 1.5, class_count)
    features = np.vstack(
        [
            generator.normal(center, 0.35, size=(rows_per_class, feature_count))
            for center in centers
        ]
    )
    sample_ids = tuple(f"train-{index:03d}" for index in range(len(features)))
    return features, labels, sample_ids


def _fit(*, class_count: int = 4, feature_count: int = 8):
    features, labels, sample_ids = _training_rows(
        class_count=class_count,
        feature_count=feature_count,
    )
    fitted = fit_mlp_classifier(
        features,
        labels,
        sample_ids=sample_ids,
        task_id="aqse.network-pattern.v1",
        input_space_id="aqse-afse-fixture",
        fitted_on_dataset_id="aqse-network-demo-fixture",
        fitted_on_dataset_digest=DATASET_DIGEST,
        feature_order=tuple(f"z{index}" for index in range(feature_count)),
        effective_source_hashes=SOURCE_HASHES,
    )
    return fitted, features


def test_portable_numpy_inference_matches_fitted_sklearn_multiclass() -> None:
    fitted, features = _fit()
    expected = fitted.sklearn_predict_proba(features)
    actual = fitted.runtime.predict_proba(
        features,
        context=query_context_for(fitted.artifact),
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-9)
    assert fitted.artifact.numpy_reference_max_abs_error <= 1.0e-9
    assert fitted.artifact.classes == tuple(sorted(fitted.artifact.classes))
    assert fitted.artifact.output_activation == "softmax"


def test_portable_numpy_inference_matches_fitted_sklearn_binary() -> None:
    fitted, features = _fit(class_count=2)
    actual = fitted.runtime.predict_proba(
        features[:7],
        context=query_context_for(fitted.artifact),
    )

    np.testing.assert_allclose(
        actual,
        fitted.sklearn_predict_proba(features[:7]),
        rtol=0.0,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-14)
    assert fitted.artifact.output_activation == "logistic"


def test_json_reload_is_non_executable_and_prediction_stable() -> None:
    fitted, features = _fit()
    payload = fitted.artifact.model_dump_json()
    reloaded_artifact = MLPClassifierArtifact.model_validate_json(payload)
    reloaded = NumpyMLPClassifier.from_artifact(reloaded_artifact)
    context = query_context_for(reloaded_artifact)

    expected = fitted.runtime.predict_proba(features[:5], context=context)
    actual = reloaded.predict_proba(features[:5], context=context)
    np.testing.assert_array_equal(actual, expected)
    assert reloaded_artifact.persistence_format.endswith("no-pickle")


def test_fit_is_deterministic_and_standardizer_is_train_only() -> None:
    first, features = _fit()
    second, _ = _fit()

    assert first.artifact == second.artifact
    np.testing.assert_allclose(first.artifact.standardizer.mean, features.mean(axis=0))
    mean_before = first.artifact.standardizer.mean
    first.runtime.predict_proba(
        features[:2] + 1000.0,
        context=query_context_for(first.artifact),
    )
    assert first.artifact.standardizer.mean == mean_before
    assert first.artifact.fitted_on_partition == "TRAIN"
    assert first.artifact.random_state == 2_001_005
    assert first.artifact.hidden_layer_sizes == (32, 16)


def test_query_order_and_batch_invariance() -> None:
    fitted, features = _fit()
    context = query_context_for(fitted.artifact)
    expected = fitted.runtime.predict_proba(features[:6], context=context)
    order = np.asarray([4, 1, 5, 0, 3, 2])
    reordered = fitted.runtime.predict_proba(features[:6][order], context=context)
    single = fitted.runtime.predict_proba(features[3:4], context=context)

    np.testing.assert_allclose(reordered, expected[order], rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(single[0], expected[3], rtol=0.0, atol=1.0e-15)


def test_fixed_uncertainty_gate_uses_top_score_or_margin() -> None:
    uncertain, top, margins = uncertainty_gate(
        np.asarray(
            [
                [0.80, 0.10, 0.05, 0.05],
                [0.69, 0.20, 0.06, 0.05],
                [0.72, 0.18, 0.05, 0.05],
                [0.56, 0.44, 0.00, 0.00],
            ]
        )
    )

    assert uncertain.tolist() == [False, True, False, True]
    np.testing.assert_allclose(top, [0.80, 0.69, 0.72, 0.56])
    np.testing.assert_allclose(margins, [0.70, 0.49, 0.54, 0.12])


def test_score_reports_raw_prediction_separately_from_uncertain_display() -> None:
    fitted, features = _fit()
    result = fitted.runtime.score(
        features[:4],
        sample_ids=("q0", "q1", "q2", "q3"),
        context=query_context_for(fitted.artifact),
    )

    assert len(result.scores) == 4
    assert len(result.predicted_classes) == 4
    assert all(
        displayed == "UNCERTAIN" if uncertain else displayed == predicted
        for displayed, predicted, uncertain in zip(
            result.displayed_classes,
            result.predicted_classes,
            result.uncertain,
            strict=True,
        )
    )
    assert result.score_semantics == "model-score;not-probability-calibrated"


def test_input_space_mismatch_is_rejected() -> None:
    fitted, features = _fit()
    wrong = query_context_for(fitted.artifact).model_copy(
        update={"input_space_id": "aqse-afse-from-another-theta"}
    )

    with pytest.raises(ValueError, match="incompatible"):
        fitted.runtime.predict_proba(features[:1], context=wrong)


def test_raw_feature_baseline_uses_same_architecture_with_explicit_role() -> None:
    features, labels, sample_ids = _training_rows(feature_count=6)
    fitted = fit_raw_feature_baseline(
        features,
        labels,
        sample_ids=sample_ids,
        task_id="aqse.network-pattern.v1",
        input_space_id="aqse.network-state8.v1.raw",
        fitted_on_dataset_id="aqse-network-demo-fixture",
        fitted_on_dataset_digest=DATASET_DIGEST,
        feature_order=tuple(f"f{index}" for index in range(6)),
        effective_source_hashes=SOURCE_HASHES,
    )

    assert fitted.artifact.model_role == "raw_feature_baseline"
    assert fitted.artifact.hidden_layer_sizes == (32, 16)


def test_convergence_warnings_are_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = MLPClassifier.fit

    def warning_fit(self, features, labels):
        warnings.warn("fixture convergence warning", ConvergenceWarning, stacklevel=2)
        return original_fit(self, features, labels)

    monkeypatch.setattr(MLPClassifier, "fit", warning_fit)
    fitted, _ = _fit(class_count=2)

    assert any("fixture convergence warning" in value for value in fitted.artifact.convergence_warnings)


def test_invalid_inputs_and_corrupted_artifact_are_rejected() -> None:
    fitted, features = _fit()
    context = query_context_for(fitted.artifact)

    with pytest.raises(ValueError, match="dimension"):
        fitted.runtime.predict_proba(features[:1, :7], context=context)
    with pytest.raises(ValueError, match="NaN"):
        fitted.runtime.predict_proba(
            np.full((1, features.shape[1]), np.nan),
            context=context,
        )
    with pytest.raises(ValueError, match="content digest"):
        validate_mlp_artifact(
            fitted.artifact.model_copy(update={"content_digest": "0" * 64})
        )
