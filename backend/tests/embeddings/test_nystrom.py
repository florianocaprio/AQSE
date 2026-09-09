from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from app.embeddings.nystrom import (
    AFSE_METHOD_ID,
    NystromAFSE,
    NystromAFSEArtifact,
    fit_nystrom_afse,
    query_context_for,
    validate_afse_artifact,
)
from app.quantum.adapter import TQK8Adapter

DATASET_DIGEST = "a" * 64
SOURCE_HASHES = {"fixture.py": "b" * 64}


def _training_rows(
    *,
    classes: tuple[str, ...] = ("DEVICE_COMPATIBLE", "ENVIRONMENT_COMPATIBLE"),
    rows_per_class: int = 4,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    generator = np.random.default_rng(42)
    row_count = len(classes) * rows_per_class
    features = generator.uniform(-1.0, 1.0, size=(row_count, 8))
    labels = tuple(label for label in classes for _ in range(rows_per_class))
    sample_ids = tuple(f"sample-{index:03d}" for index in range(row_count))
    lineage_ids = tuple(f"lineage-{index:03d}" for index in range(row_count))
    return features, sample_ids, lineage_ids, labels


def _fit(
    *,
    classes: tuple[str, ...] = ("DEVICE_COMPATIBLE", "ENVIRONMENT_COMPATIBLE"),
    rows_per_class: int = 4,
):
    features, sample_ids, lineage_ids, labels = _training_rows(
        classes=classes,
        rows_per_class=rows_per_class,
    )
    theta = np.linspace(-0.6, 0.6, 16, dtype=np.float64)
    fitted = fit_nystrom_afse(
        features,
        sample_ids=sample_ids,
        lineage_ids=lineage_ids,
        classes=labels,
        theta=theta,
        feature_profile_id="aqse.network-state8.v1",
        scaler_id="aqse-angle-scaler-test",
        encoding_policy_id="aqse.tqk8.encoding.all-scaled.v1",
        fitted_on_dataset_id="aqse-network-demo-fixture",
        fitted_on_dataset_digest=DATASET_DIGEST,
        effective_source_hashes=SOURCE_HASHES,
    )
    return fitted, features, sample_ids


def test_fit_is_deterministic_balanced_and_uses_distinct_train_lineages() -> None:
    first, _, _ = _fit(rows_per_class=5)
    second, _, _ = _fit(rows_per_class=5)

    assert first.artifact == second.artifact
    assert first.artifact.method_id == AFSE_METHOD_ID
    assert first.artifact.fitted_on_partition == "TRAIN"
    assert first.artifact.reference_size == 10
    assert first.artifact.output_dimension == 10
    assert len(set(first.artifact.landmark_sample_ids)) == 10
    assert len(set(first.artifact.landmark_lineage_ids)) == 10
    assert Counter(first.artifact.landmark_classes) == {
        "DEVICE_COMPATIBLE": 5,
        "ENVIRONMENT_COMPATIBLE": 5,
    }


def test_four_class_landmarks_are_balanced_and_capped_at_32() -> None:
    classes = ("DEVICE", "ENVIRONMENT", "MIXED", "NORMAL")
    fitted, _, _ = _fit(classes=classes, rows_per_class=10)

    assert fitted.artifact.reference_size == 32
    assert Counter(fitted.artifact.landmark_classes) == {label: 8 for label in classes}


def test_psd_basis_dimension_and_residual_policy_are_finite() -> None:
    fitted, features, _ = _fit()
    artifact = fitted.artifact
    eigenvalues = np.asarray(artifact.gram_eigenvalues)
    basis = np.asarray(artifact.b_matrix)
    context = query_context_for(artifact)
    result = fitted.transform(features[:3], sample_ids=("q0", "q1", "q2"), context=context)

    assert float(eigenvalues.min()) >= -1.0e-10 * max(1.0, float(eigenvalues.max()))
    assert basis.shape == (artifact.reference_size, artifact.reference_size)
    np.testing.assert_allclose(basis, basis.T, rtol=0.0, atol=1.0e-12)
    assert np.asarray(result.vectors).shape == (3, artifact.reference_size)
    assert np.isfinite(result.vectors).all()
    assert np.isfinite(result.reconstruction_residuals).all()
    assert all(value >= 0.0 for value in result.reconstruction_residuals)
    assert artifact.train_residual_p99 >= 0.0
    assert artifact.score_semantics.endswith("not-calibrated")


def test_query_order_batch_and_single_query_invariance() -> None:
    fitted, features, _ = _fit()
    queries = features[:4] + 0.013
    context = query_context_for(fitted.artifact)
    expected = fitted.transform(
        queries,
        sample_ids=("q0", "q1", "q2", "q3"),
        context=context,
    )
    order = np.asarray([2, 0, 3, 1])
    reordered = fitted.transform(
        queries[order],
        sample_ids=tuple(f"r{index}" for index in range(4)),
        context=context,
    )
    single = fitted.transform(queries[1:2], sample_ids=("single",), context=context)

    np.testing.assert_allclose(
        np.asarray(reordered.vectors),
        np.asarray(expected.vectors)[order],
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(single.vectors[0]),
        np.asarray(expected.vectors[1]),
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        single.reconstruction_residuals[0],
        expected.reconstruction_residuals[1],
        rtol=0.0,
        atol=2.0e-13,
    )


def test_json_reload_is_deterministic_and_uses_frozen_reference() -> None:
    fitted, features, _ = _fit()
    reloaded_artifact = NystromAFSEArtifact.model_validate_json(
        fitted.artifact.model_dump_json()
    )
    validate_afse_artifact(reloaded_artifact)
    reloaded = NystromAFSE.from_artifact(reloaded_artifact)
    context = query_context_for(reloaded_artifact)

    first = fitted.transform(features[:2], sample_ids=("a", "b"), context=context)
    second = reloaded.transform(features[:2], sample_ids=("a", "b"), context=context)
    np.testing.assert_allclose(first.vectors, second.vectors, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        first.reconstruction_residuals,
        second.reconstruction_residuals,
        rtol=0.0,
        atol=1.0e-13,
    )


def test_runtime_caches_landmark_states_and_prepares_only_query_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, features, _ = _fit()
    calls: list[tuple[float, ...]] = []
    original = TQK8Adapter.state

    def counted_state(self, feature_row, theta):
        calls.append(tuple(float(value) for value in feature_row))
        return original(self, feature_row, theta)

    monkeypatch.setattr(TQK8Adapter, "state", counted_state)
    runtime = NystromAFSE.from_artifact(fitted.artifact)
    reference_calls = len(calls)
    runtime.transform(
        features[:3],
        sample_ids=("q0", "q1", "q2"),
        context=query_context_for(fitted.artifact),
    )

    assert reference_calls == fitted.artifact.reference_size
    assert len(calls) == reference_calls + 3


@pytest.mark.parametrize(
    ("field", "wrong"),
    (
        ("feature_profile_id", "aqse.local-state8.v1"),
        ("scaler_id", "old-scaler"),
        ("encoding_policy_id", "legacy-phase-direct"),
        ("theta_id", "aqse-theta-0000000000000000"),
    ),
)
def test_incompatible_old_or_different_bundle_is_rejected(field: str, wrong: str) -> None:
    fitted, features, _ = _fit()
    context = query_context_for(fitted.artifact).model_copy(update={field: wrong})

    with pytest.raises(ValueError, match="incompatible"):
        fitted.transform(features[:1], sample_ids=("query",), context=context)


def test_invalid_training_inputs_are_rejected() -> None:
    features, sample_ids, lineage_ids, labels = _training_rows(rows_per_class=2)
    common = dict(
        sample_ids=sample_ids,
        lineage_ids=lineage_ids,
        classes=labels,
        theta=np.zeros(16),
        feature_profile_id="profile",
        scaler_id="scaler",
        encoding_policy_id="encoding",
        fitted_on_dataset_id="dataset",
        fitted_on_dataset_digest=DATASET_DIGEST,
        effective_source_hashes=SOURCE_HASHES,
    )

    with pytest.raises(ValueError, match="shape"):
        fit_nystrom_afse(features[:, :7], **common)
    with pytest.raises(ValueError, match="distinct"):
        fit_nystrom_afse(
            features,
            **{**common, "sample_ids": ("same",) * len(features)},
        )
    with pytest.raises(ValueError, match="multiple downstream classes"):
        fit_nystrom_afse(
            features,
            **{**common, "lineage_ids": ("shared",) * len(features)},
        )


def test_small_numpy_and_qiskit_query_paths_agree() -> None:
    fitted, features, _ = _fit(rows_per_class=2)
    qiskit_runtime = NystromAFSE.from_artifact(fitted.artifact, backend_type="qiskit")
    context = query_context_for(fitted.artifact)
    numpy_result = fitted.transform(features[:2], sample_ids=("q0", "q1"), context=context)
    qiskit_result = qiskit_runtime.transform(
        features[:2],
        sample_ids=("q0", "q1"),
        context=context,
    )

    np.testing.assert_allclose(numpy_result.vectors, qiskit_result.vectors, atol=1.0e-10)
    np.testing.assert_allclose(
        numpy_result.reconstruction_residuals,
        qiskit_result.reconstruction_residuals,
        atol=1.0e-10,
    )
