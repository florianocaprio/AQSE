from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from app.classical.models import (
    MLP_MODEL_ID,
    MLPClassifierArtifact,
    MLPQueryContext,
    MLPScoreBatch,
    ModelRole,
    StandardizerArtifact,
)
from app.training.canonical import canonical_json_bytes, file_sha256


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _effective_source_hashes() -> dict[str, str]:
    root = _backend_root()
    relative_paths = (
        "app/classical/mlp.py",
        "app/classical/models.py",
    )
    return {relative: file_sha256(root / relative) for relative in relative_paths}


def _as_feature_matrix(
    features: NDArray[Any] | Sequence[Sequence[float]],
    *,
    expected_columns: int | None = None,
) -> NDArray[np.float64]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) == 0 or matrix.shape[1] == 0:
        raise ValueError("MLP features must be a non-empty two-dimensional matrix")
    if expected_columns is not None and matrix.shape[1] != expected_columns:
        raise ValueError("MLP feature dimension is incompatible with the fitted model")
    if not np.isfinite(matrix).all():
        raise ValueError("MLP features contain NaN or infinity")
    return np.asarray(matrix, dtype=np.float64)


def _stable_sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(values, -709.0, 709.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)


def _stable_softmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return np.asarray(exponential / exponential.sum(axis=1, keepdims=True), dtype=np.float64)


def uncertainty_gate(
    scores: NDArray[Any] | Sequence[Sequence[float]],
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Apply the fixed engineering score/margin gate; scores are not calibrated."""

    probabilities = np.asarray(scores, dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) == 0 or probabilities.shape[1] < 2:
        raise ValueError("uncertainty gate requires at least two scores per row")
    if not np.isfinite(probabilities).all():
        raise ValueError("uncertainty gate scores contain NaN or infinity")
    order = np.argsort(probabilities, axis=1)
    top_scores = probabilities[np.arange(len(probabilities)), order[:, -1]]
    second_scores = probabilities[np.arange(len(probabilities)), order[:, -2]]
    margins = top_scores - second_scores
    uncertain = np.logical_or(top_scores < 0.70, margins < 0.15)
    return (
        np.asarray(uncertain, dtype=np.bool_),
        np.asarray(top_scores, dtype=np.float64),
        np.asarray(margins, dtype=np.float64),
    )


def _artifact_digest_payload(artifact: MLPClassifierArtifact) -> dict[str, Any]:
    return artifact.model_dump(mode="json", exclude={"artifact_id", "content_digest"})


def validate_mlp_artifact(artifact: MLPClassifierArtifact) -> None:
    expected_digest = _digest(_artifact_digest_payload(artifact))
    if artifact.content_digest != expected_digest:
        raise ValueError("MLP scientific content digest is invalid")
    if artifact.artifact_id != f"aqse-mlp-{expected_digest[:16]}":
        raise ValueError("MLP artifact identity is invalid")
    if any(not _is_sha256(value) for value in artifact.effective_source_hashes.values()):
        raise ValueError("MLP source provenance contains an invalid SHA-256")


def query_context_for(artifact: MLPClassifierArtifact) -> MLPQueryContext:
    return MLPQueryContext(
        model_role=artifact.model_role,
        task_id=artifact.task_id,
        input_space_id=artifact.input_space_id,
        feature_count=artifact.feature_count,
        feature_order=artifact.feature_order,
    )


@dataclass(frozen=True)
class NumpyMLPClassifier:
    """Small non-executable evaluator reconstructed only from numeric parameters."""

    artifact: MLPClassifierArtifact

    @classmethod
    def from_artifact(cls, artifact: MLPClassifierArtifact) -> NumpyMLPClassifier:
        validate_mlp_artifact(artifact)
        return cls(artifact=artifact)

    def predict_proba(
        self,
        features: NDArray[Any] | Sequence[Sequence[float]],
        *,
        context: MLPQueryContext,
    ) -> NDArray[np.float64]:
        validate_mlp_artifact(self.artifact)
        if context != query_context_for(self.artifact):
            raise ValueError("query is incompatible with the frozen MLP input space")
        matrix = _as_feature_matrix(features, expected_columns=self.artifact.feature_count)
        mean = np.asarray(self.artifact.standardizer.mean, dtype=np.float64)
        scale = np.asarray(self.artifact.standardizer.scale, dtype=np.float64)
        hidden = (matrix - mean) / scale
        coefficients = [
            np.asarray(values, dtype=np.float64) for values in self.artifact.coefficients
        ]
        intercepts = [
            np.asarray(values, dtype=np.float64) for values in self.artifact.intercepts
        ]
        hidden = np.tanh(hidden @ coefficients[0] + intercepts[0])
        hidden = np.tanh(hidden @ coefficients[1] + intercepts[1])
        logits = np.asarray(hidden @ coefficients[2] + intercepts[2], dtype=np.float64)
        if len(self.artifact.classes) == 2:
            positive = _stable_sigmoid(logits[:, 0])
            probabilities = np.column_stack((1.0 - positive, positive))
        else:
            probabilities = _stable_softmax(logits)
        if not np.isfinite(probabilities).all():
            raise ValueError("MLP inference produced NaN or infinity")
        return np.asarray(probabilities, dtype=np.float64)

    def score(
        self,
        features: NDArray[Any] | Sequence[Sequence[float]],
        *,
        sample_ids: Sequence[str],
        context: MLPQueryContext,
    ) -> MLPScoreBatch:
        probabilities = self.predict_proba(features, context=context)
        identifiers = tuple(sample_ids)
        if len(identifiers) != len(probabilities) or len(set(identifiers)) != len(identifiers):
            raise ValueError("MLP query sample identifiers must be complete and distinct")
        order = np.argsort(probabilities, axis=1)
        top_indices = order[:, -1]
        uncertain, top_scores, margins = uncertainty_gate(probabilities)
        predicted = tuple(self.artifact.classes[int(index)] for index in top_indices)
        displayed = tuple(
            "UNCERTAIN" if is_uncertain else label
            for is_uncertain, label in zip(uncertain, predicted, strict=True)
        )
        return MLPScoreBatch(
            sample_ids=identifiers,
            class_order=self.artifact.classes,
            scores=tuple(tuple(float(value) for value in row) for row in probabilities),
            predicted_classes=predicted,
            displayed_classes=displayed,
            uncertain=tuple(bool(value) for value in uncertain),
            top_scores=tuple(float(value) for value in top_scores),
            top_two_margins=tuple(float(value) for value in margins),
        )


@dataclass(frozen=True)
class FittedMLP:
    artifact: MLPClassifierArtifact
    runtime: NumpyMLPClassifier
    _sklearn_model: MLPClassifier = field(repr=False, compare=False)

    def sklearn_predict_proba(
        self,
        features: NDArray[Any] | Sequence[Sequence[float]],
    ) -> NDArray[np.float64]:
        """Expose fitted-estimator scores only for fit-time validation and tests."""

        matrix = _as_feature_matrix(
            features,
            expected_columns=self.artifact.feature_count,
        )
        mean = np.asarray(self.artifact.standardizer.mean, dtype=np.float64)
        scale = np.asarray(self.artifact.standardizer.scale, dtype=np.float64)
        return np.asarray(
            self._sklearn_model.predict_proba((matrix - mean) / scale),
            dtype=np.float64,
        )


def fit_mlp_classifier(
    train_features: NDArray[Any] | Sequence[Sequence[float]],
    train_classes: Sequence[str],
    *,
    sample_ids: Sequence[str],
    task_id: str,
    input_space_id: str,
    fitted_on_dataset_id: str,
    fitted_on_dataset_digest: str,
    feature_order: Sequence[str],
    model_role: ModelRole = "afse_classifier",
    effective_source_hashes: Mapping[str, str] | None = None,
) -> FittedMLP:
    """Fit the fixed TRAIN-only compact MLP and verify portable NumPy inference."""

    features = _as_feature_matrix(train_features)
    labels = np.asarray(tuple(train_classes), dtype=str)
    identifiers = tuple(sample_ids)
    ordered_features = tuple(feature_order)
    if len(features) != len(labels) or len(features) != len(identifiers):
        raise ValueError("MLP TRAIN features, classes and sample IDs must have equal lengths")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("MLP TRAIN sample identifiers must be distinct")
    if len(ordered_features) != features.shape[1] or len(set(ordered_features)) != len(
        ordered_features
    ):
        raise ValueError("MLP feature order must be complete and distinct")
    if len(set(labels.tolist())) < 2:
        raise ValueError("MLP training requires at least two classes")
    if not _is_sha256(fitted_on_dataset_digest):
        raise ValueError("MLP fitted dataset digest must be a SHA-256")

    standardizer = StandardScaler()
    standardized = np.asarray(standardizer.fit_transform(features), dtype=np.float64)
    classifier = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="tanh",
        solver="lbfgs",
        alpha=1.0e-3,
        max_iter=500,
        max_fun=15_000,
        random_state=2_001_005,
        early_stopping=False,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(standardized, labels)
    convergence_warnings = tuple(
        str(item.message) for item in captured if issubclass(item.category, ConvergenceWarning)
    )
    classes = tuple(str(value) for value in classifier.classes_)
    output_activation = "logistic" if len(classes) == 2 else "softmax"
    source_hashes = dict(effective_source_hashes or _effective_source_hashes())
    scientific: dict[str, Any] = {
        "schema_version": "aqse.classical-mlp-artifact.v1",
        "model_id": MLP_MODEL_ID,
        "model_role": model_role,
        "task_id": task_id,
        "input_space_id": input_space_id,
        "fitted_on_partition": "TRAIN",
        "fitted_on_dataset_id": fitted_on_dataset_id,
        "fitted_on_dataset_digest": fitted_on_dataset_digest,
        "train_sample_ids": list(identifiers),
        "feature_count": features.shape[1],
        "feature_order": list(ordered_features),
        "classes": list(classes),
        "standardizer": StandardizerArtifact(
            mean=tuple(float(value) for value in standardizer.mean_),
            scale=tuple(float(value) for value in standardizer.scale_),
        ).model_dump(mode="json"),
        "hidden_layer_sizes": [32, 16],
        "activation": "tanh",
        "output_activation": output_activation,
        "solver": "lbfgs",
        "alpha": 1.0e-3,
        "max_iter": 500,
        "max_fun": 15_000,
        "random_state": 2_001_005,
        "early_stopping": False,
        "coefficients": [matrix.tolist() for matrix in classifier.coefs_],
        "intercepts": [values.tolist() for values in classifier.intercepts_],
        "n_iter": int(classifier.n_iter_),
        "terminal_loss": float(classifier.loss_),
        "convergence_warnings": list(convergence_warnings),
        "uncertain_top_score_threshold": 0.70,
        "uncertain_margin_threshold": 0.15,
        "score_semantics": "model-score;not-probability-calibrated",
        "persistence_format": "bounded-json-numeric-arrays;no-pickle",
        "sklearn_version": sklearn.__version__,
        "numpy_reference_max_abs_error": 0.0,
        "effective_source_hashes": source_hashes,
    }
    provisional_digest = _digest(scientific)
    provisional = MLPClassifierArtifact.model_validate(
        {
            **scientific,
            "artifact_id": f"aqse-mlp-{provisional_digest[:16]}",
            "content_digest": provisional_digest,
        }
    )
    portable = NumpyMLPClassifier(artifact=provisional)
    expected = np.asarray(classifier.predict_proba(standardized), dtype=np.float64)
    actual = portable.predict_proba(features, context=query_context_for(provisional))
    maximum_error = float(np.max(np.abs(actual - expected)))
    if maximum_error > 1.0e-9:
        raise RuntimeError("portable NumPy MLP inference differs from fitted sklearn scores")

    scientific["numpy_reference_max_abs_error"] = maximum_error
    content_digest = _digest(scientific)
    artifact = MLPClassifierArtifact.model_validate(
        {
            **scientific,
            "artifact_id": f"aqse-mlp-{content_digest[:16]}",
            "content_digest": content_digest,
        }
    )
    validate_mlp_artifact(artifact)
    return FittedMLP(
        artifact=artifact,
        runtime=NumpyMLPClassifier.from_artifact(artifact),
        _sklearn_model=classifier,
    )


def fit_raw_feature_baseline(
    train_features: NDArray[Any] | Sequence[Sequence[float]],
    train_classes: Sequence[str],
    **kwargs: Any,
) -> FittedMLP:
    """Fit the declared same-architecture reference directly on raw features."""

    return fit_mlp_classifier(
        train_features,
        train_classes,
        model_role="raw_feature_baseline",
        **kwargs,
    )
