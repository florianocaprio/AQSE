from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.quantum.user_pipeline.tqk8 import (
    N_PARAMS,
    StateEngine,
    alignment_loss,
    fit_qng,
    loss_grad_metric,
)
from app.training.canonical import array_identity, canonical_json_bytes
from app.training.evaluation_assembly import ComparisonInput
from app.training.evaluation_models import (
    CandidateEvaluation,
    ClassicalPreprocessingSnapshot,
    ClassificationMetrics,
    ComparativeEvaluationArtifact,
    EvaluationProtocol,
    FinalEvaluationProcedure,
    MethodName,
    MethodRuntime,
    MethodSelection,
    MetricInterval,
    ModelSelectionFreeze,
)
from app.training.evaluation_protocol import METHODS, validate_evaluation_protocol
from app.training.run_models import TrainingRunArtifact
from app.training.runner import TQK8_SOURCE_SHA256

CLASSICAL_FEATURE_ORDER = (
    "amplitude",
    "phase_sin",
    "phase_cos",
    "frequency",
    "variance",
    "drift",
    "snr",
    "spectral_peak",
    "temperature",
)


@dataclass(frozen=True)
class ThetaCheckpoint:
    seed: int
    checkpoint_index: int
    theta: NDArray[np.float64]
    train_alignment_loss: float


@dataclass(frozen=True)
class EvaluationOutcome:
    artifact: ComparativeEvaluationArtifact
    method_runtimes: tuple[MethodRuntime, ...]
    total_wall_time_ms: float


@dataclass(frozen=True)
class _ScoredCandidate:
    artifact: CandidateEvaluation
    validation_predictions: NDArray[np.int8]
    validation_scores: NDArray[np.float64]


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _classification_metrics(
    labels: NDArray[np.int8],
    predictions: NDArray[np.int8],
    scores: NDArray[np.float64],
) -> ClassificationMetrics:
    y = np.asarray(labels, dtype=np.int8)
    predicted = np.asarray(predictions, dtype=np.int8)
    decision = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or predicted.shape != y.shape or decision.shape != y.shape:
        raise ValueError("classification arrays must be aligned one-dimensional vectors")
    if set(y.tolist()) != {-1, 1} or set(predicted.tolist()) - {-1, 1}:
        raise ValueError("classification metrics require the approved -1/+1 labels")
    matrix = confusion_matrix(y, predicted, labels=[-1, 1])
    recalls = recall_score(y, predicted, labels=[-1, 1], average=None, zero_division=0.0)
    return ClassificationMetrics(
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
        macro_f1=float(f1_score(y, predicted, labels=[-1, 1], average="macro")),
        roc_auc=float(roc_auc_score(y, decision)),
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
        negative_support=int(np.count_nonzero(y == -1)),
        positive_support=int(np.count_nonzero(y == 1)),
        negative_recall=float(recalls[0]),
        positive_recall=float(recalls[1]),
    )


def _candidate(**scientific: Any) -> CandidateEvaluation:
    digest = _digest(scientific)
    return CandidateEvaluation.model_validate(
        {
            **scientific,
            "candidate_id": f"aqse-comparison-candidate-{digest[:16]}",
            "content_digest": digest,
        }
    )


def validate_candidate(candidate: CandidateEvaluation) -> None:
    scientific = candidate.model_dump(
        mode="json",
        exclude={"candidate_id", "content_digest"},
    )
    digest = _digest(scientific)
    if candidate.content_digest != digest:
        raise ValueError("comparison candidate digest is invalid")
    if candidate.candidate_id != f"aqse-comparison-candidate-{digest[:16]}":
        raise ValueError("comparison candidate identity is invalid")


def _circular_features(raw: NDArray[np.float64]) -> NDArray[np.float64]:
    value = np.asarray(raw, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 8 or not np.isfinite(value).all():
        raise ValueError("classical circular features require a finite Nx8 matrix")
    return np.column_stack(
        (
            value[:, 0],
            np.sin(value[:, 1]),
            np.cos(value[:, 1]),
            value[:, 2:],
        )
    )


def _snr_threshold(train_snr: NDArray[np.float64], labels: NDArray[np.int8]) -> float:
    ordered = np.unique(np.asarray(train_snr, dtype=np.float64))
    thresholds = [float(np.nextafter(ordered[0], -np.inf))]
    thresholds.extend(float((left + right) / 2.0) for left, right in zip(ordered[:-1], ordered[1:]))
    thresholds.append(float(np.nextafter(ordered[-1], np.inf)))
    ranked: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        predicted = np.where(train_snr <= threshold, 1, -1).astype(np.int8)
        metrics = _classification_metrics(labels, predicted, threshold - train_snr)
        ranked.append((-metrics.balanced_accuracy, -metrics.macro_f1, threshold))
    return min(ranked)[2]


def _scored_candidate(
    *,
    method: MethodName,
    y_train: NDArray[np.int8],
    train_predictions: NDArray[np.int8],
    train_scores: NDArray[np.float64],
    y_validation: NDArray[np.int8],
    validation_predictions: NDArray[np.int8],
    validation_scores: NDArray[np.float64],
    seed: int | None = None,
    checkpoint_index: int | None = None,
    svc_c: float | None = None,
    rbf_gamma: float | None = None,
    snr_threshold: float | None = None,
    theta: NDArray[np.float64] | None = None,
    train_alignment_loss: float | None = None,
    validation_alignment_loss: float | None = None,
) -> _ScoredCandidate:
    predicted = np.asarray(validation_predictions, dtype=np.int8)
    scores = np.asarray(validation_scores, dtype=np.float64)
    train_metric_value = _classification_metrics(
        y_train,
        train_predictions,
        train_scores,
    )
    validation_metric_value = _classification_metrics(
        y_validation,
        predicted,
        scores,
    )
    scientific = {
        "method": method,
        "seed": seed,
        "checkpoint_index": checkpoint_index,
        "svc_c": svc_c,
        "rbf_gamma": rbf_gamma,
        "snr_threshold": snr_threshold,
        "theta": None if theta is None else tuple(float(value) for value in theta),
        "train_alignment_loss": train_alignment_loss,
        "validation_alignment_loss": validation_alignment_loss,
        "train_metrics": train_metric_value.model_dump(mode="json"),
        "validation_metrics": validation_metric_value.model_dump(mode="json"),
        "validation_prediction_digest": array_identity(predicted)["content_sha256"],
        "validation_score_digest": array_identity(scores)["content_sha256"],
    }
    return _ScoredCandidate(
        artifact=_candidate(**scientific),
        validation_predictions=predicted.copy(),
        validation_scores=scores.copy(),
    )


def _evaluate_snr(comparison: ComparisonInput) -> list[_ScoredCandidate]:
    train_snr = np.asarray(comparison.X_train_raw[:, 5], dtype=np.float64)
    validation_snr = np.asarray(comparison.X_validation_raw[:, 5], dtype=np.float64)
    threshold = _snr_threshold(train_snr, comparison.y_train)
    return [
        _scored_candidate(
            method="snr_threshold",
            y_train=comparison.y_train,
            train_predictions=np.where(train_snr <= threshold, 1, -1).astype(np.int8),
            train_scores=threshold - train_snr,
            y_validation=comparison.y_validation,
            validation_predictions=np.where(validation_snr <= threshold, 1, -1).astype(
                np.int8
            ),
            validation_scores=threshold - validation_snr,
            snr_threshold=threshold,
        )
    ]


def _fit_classical_scaler(
    comparison: ComparisonInput,
) -> tuple[ClassicalPreprocessingSnapshot, NDArray[np.float64], NDArray[np.float64]]:
    train = _circular_features(comparison.X_train_raw)
    validation = _circular_features(comparison.X_validation_raw)
    scaler = StandardScaler().fit(train)
    transformed_train = np.asarray(scaler.transform(train), dtype=np.float64)
    transformed_validation = np.asarray(scaler.transform(validation), dtype=np.float64)
    snapshot = ClassicalPreprocessingSnapshot(
        feature_order=CLASSICAL_FEATURE_ORDER,
        fitted_mean=tuple(float(value) for value in scaler.mean_),
        fitted_scale=tuple(float(value) for value in scaler.scale_),
    )
    return snapshot, transformed_train, transformed_validation


def _evaluate_rbf(
    comparison: ComparisonInput,
    protocol: EvaluationProtocol,
    train: NDArray[np.float64],
    validation: NDArray[np.float64],
) -> list[_ScoredCandidate]:
    candidates: list[_ScoredCandidate] = []
    for c_value in protocol.rbf_c_grid:
        for gamma in protocol.rbf_gamma_grid:
            model = SVC(kernel="rbf", C=c_value, gamma=gamma).fit(
                train,
                comparison.y_train,
            )
            candidates.append(
                _scored_candidate(
                    method="rbf_svc_circular",
                    y_train=comparison.y_train,
                    train_predictions=np.asarray(model.predict(train), dtype=np.int8),
                    train_scores=np.asarray(model.decision_function(train), dtype=np.float64),
                    y_validation=comparison.y_validation,
                    validation_predictions=np.asarray(
                        model.predict(validation),
                        dtype=np.int8,
                    ),
                    validation_scores=np.asarray(
                        model.decision_function(validation),
                        dtype=np.float64,
                    ),
                    svc_c=c_value,
                    rbf_gamma=gamma,
                )
            )
    return candidates


def _theta0(seed: int) -> NDArray[np.float64]:
    return np.asarray(
        np.random.default_rng(seed).uniform(-0.8, 0.8, N_PARAMS),
        dtype=np.float64,
    )


def _alignment(engine: StateEngine, X: NDArray[np.float64], y: NDArray[np.int8], theta: NDArray[np.float64]) -> float:
    return float(alignment_loss(engine.gram(X, theta), y)[0])


def _qng_checkpoints_from_primary_run(
    run: TrainingRunArtifact,
    comparison: ComparisonInput,
) -> tuple[ThetaCheckpoint, ...]:
    if (
        run.run_id != "aqse-qng-run-f00c702ad790df2b"
        or run.theta_seed != 1_001_005
        or run.tqk8_source_sha256 != TQK8_SOURCE_SHA256
        or run.training_input.encoded_matrix_identity
        != comparison.identity.train_encoded_identity
        or run.training_input.ordered_label_identity
        != comparison.identity.train_label_identity
    ):
        raise ValueError("primary QNG run is incompatible with the 1D.4a comparison")
    checkpoints = [
        ThetaCheckpoint(
            seed=run.theta_seed,
            checkpoint_index=0,
            theta=np.asarray(run.theta0, dtype=np.float64),
            train_alignment_loss=run.initial_train_alignment_loss,
        )
    ]
    checkpoints.extend(
        ThetaCheckpoint(
            seed=run.theta_seed,
            checkpoint_index=step.step_index + 1,
            theta=np.asarray(step.theta_after, dtype=np.float64),
            train_alignment_loss=step.loss_after,
        )
        for step in run.steps
    )
    return tuple(checkpoints)


def _train_qng_checkpoints(
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    seed: int,
    protocol: EvaluationProtocol,
) -> tuple[ThetaCheckpoint, ...]:
    engine = StateEngine("numpy")
    theta = _theta0(seed)
    checkpoints = [
        ThetaCheckpoint(
            seed=seed,
            checkpoint_index=0,
            theta=theta.copy(),
            train_alignment_loss=_alignment(engine, X, y, theta),
        )
    ]
    for checkpoint_index in range(1, protocol.maximum_updates + 1):
        theta_after, history = fit_qng(
            engine,
            X,
            y,
            theta,
            steps=1,
            learning_rate=protocol.qng_learning_rate,
            damping=protocol.qng_damping,
            max_step=protocol.qng_maximum_step_norm,
            verbose=False,
        )
        if not history:
            break
        if len(history) != 1:
            raise RuntimeError("protected QNG returned an invalid one-step history")
        theta = np.asarray(theta_after, dtype=np.float64)
        checkpoints.append(
            ThetaCheckpoint(
                seed=seed,
                checkpoint_index=checkpoint_index,
                theta=theta.copy(),
                train_alignment_loss=float(history[0]["loss_after"]),
            )
        )
    return tuple(checkpoints)


def _train_gradient_checkpoints(
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    seed: int,
    protocol: EvaluationProtocol,
) -> tuple[ThetaCheckpoint, ...]:
    engine = StateEngine("numpy")
    theta = _theta0(seed)
    checkpoints = [
        ThetaCheckpoint(
            seed=seed,
            checkpoint_index=0,
            theta=theta.copy(),
            train_alignment_loss=_alignment(engine, X, y, theta),
        )
    ]
    for checkpoint_index in range(1, protocol.maximum_updates + 1):
        _, gradient, _, _ = loss_grad_metric(engine, X, y, theta)
        delta = protocol.gradient_learning_rate * np.asarray(gradient, dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        if norm > protocol.gradient_maximum_step_norm:
            delta *= protocol.gradient_maximum_step_norm / norm
        theta = theta - delta
        checkpoints.append(
            ThetaCheckpoint(
                seed=seed,
                checkpoint_index=checkpoint_index,
                theta=theta.copy(),
                train_alignment_loss=_alignment(engine, X, y, theta),
            )
        )
    return tuple(checkpoints)


def _validate_kernel(
    train_kernel: NDArray[np.float64],
    validation_kernel: NDArray[np.float64],
    train_count: int,
    validation_count: int,
) -> None:
    if train_kernel.shape != (train_count, train_count):
        raise ValueError("quantum TRAIN kernel has an invalid shape")
    if validation_kernel.shape != (validation_count, train_count):
        raise ValueError("quantum VALIDATION/TRAIN kernel has an invalid orientation")
    if not np.isfinite(train_kernel).all() or not np.isfinite(validation_kernel).all():
        raise ValueError("quantum evaluation kernel contains non-finite values")
    if float(np.max(np.abs(train_kernel - train_kernel.T))) > 1.0e-12:
        raise ValueError("quantum TRAIN kernel is not symmetric")
    if float(np.max(np.abs(np.diag(train_kernel) - 1.0))) > 1.0e-12:
        raise ValueError("quantum TRAIN kernel diagonal differs from one")


def _evaluate_quantum_checkpoints(
    method: MethodName,
    checkpoints: tuple[ThetaCheckpoint, ...],
    comparison: ComparisonInput,
    protocol: EvaluationProtocol,
) -> list[_ScoredCandidate]:
    engine = StateEngine("numpy")
    candidates: list[_ScoredCandidate] = []
    for checkpoint in checkpoints:
        train_kernel = np.asarray(
            engine.gram(comparison.X_train_encoded, checkpoint.theta),
            dtype=np.float64,
        )
        validation_kernel = np.asarray(
            engine.gram(
                comparison.X_validation_encoded,
                checkpoint.theta,
                comparison.X_train_encoded,
            ),
            dtype=np.float64,
        )
        _validate_kernel(train_kernel, validation_kernel, 32, 24)
        validation_self_kernel = np.asarray(
            engine.gram(comparison.X_validation_encoded, checkpoint.theta),
            dtype=np.float64,
        )
        validation_alignment = float(
            alignment_loss(validation_self_kernel, comparison.y_validation)[0]
        )
        for c_value in protocol.quantum_svc_c_grid:
            model = SVC(kernel="precomputed", C=c_value).fit(
                train_kernel,
                comparison.y_train,
            )
            candidates.append(
                _scored_candidate(
                    method=method,
                    y_train=comparison.y_train,
                    train_predictions=np.asarray(
                        model.predict(train_kernel),
                        dtype=np.int8,
                    ),
                    train_scores=np.asarray(
                        model.decision_function(train_kernel),
                        dtype=np.float64,
                    ),
                    y_validation=comparison.y_validation,
                    validation_predictions=np.asarray(
                        model.predict(validation_kernel),
                        dtype=np.int8,
                    ),
                    validation_scores=np.asarray(
                        model.decision_function(validation_kernel),
                        dtype=np.float64,
                    ),
                    seed=checkpoint.seed,
                    checkpoint_index=checkpoint.checkpoint_index,
                    svc_c=c_value,
                    theta=checkpoint.theta,
                    train_alignment_loss=checkpoint.train_alignment_loss,
                    validation_alignment_loss=validation_alignment,
                )
            )
    return candidates


def _selection_key(item: _ScoredCandidate) -> tuple[float, float, int, float, float, int]:
    candidate = item.artifact
    seed = candidate.seed if candidate.seed is not None else 0
    return (
        -candidate.validation_metrics.balanced_accuracy,
        -candidate.validation_metrics.macro_f1,
        candidate.checkpoint_index if candidate.checkpoint_index is not None else 0,
        candidate.svc_c if candidate.svc_c is not None else 0.0,
        candidate.rbf_gamma if candidate.rbf_gamma is not None else 0.0,
        seed,
    )


def _bootstrap_intervals(
    selected: _ScoredCandidate,
    labels: NDArray[np.int8],
    protocol: EvaluationProtocol,
) -> tuple[MetricInterval, MetricInterval]:
    negative = np.flatnonzero(labels == -1)
    positive = np.flatnonzero(labels == 1)
    if len(negative) < 2 or len(positive) < 2:
        raise ValueError("VALIDATION has too few independent lineages for intervals")
    rng = np.random.default_rng(protocol.bootstrap_seed)
    balanced: list[float] = []
    macro_f1: list[float] = []
    for _ in range(protocol.bootstrap_resamples):
        indices = np.concatenate(
            (
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            )
        )
        observed = labels[indices]
        predicted = selected.validation_predictions[indices]
        balanced.append(float(balanced_accuracy_score(observed, predicted)))
        macro_f1.append(
            float(f1_score(observed, predicted, labels=[-1, 1], average="macro"))
        )

    def interval(metric: str, values: list[float], point: float) -> MetricInterval:
        lower, upper = np.quantile(np.asarray(values), (0.025, 0.975))
        return MetricInterval(
            method=selected.artifact.method,
            metric=metric,
            lower=float(lower),
            point=point,
            upper=float(upper),
        )

    metrics = selected.artifact.validation_metrics
    return (
        interval("balanced_accuracy", balanced, metrics.balanced_accuracy),
        interval("macro_f1", macro_f1, metrics.macro_f1),
    )


def _select(
    candidates: list[_ScoredCandidate],
    labels: NDArray[np.int8],
    protocol: EvaluationProtocol,
) -> MethodSelection:
    if not candidates:
        raise ValueError("method selection requires at least one candidate")
    selected = min(candidates, key=_selection_key)
    value = selected.artifact
    return MethodSelection(
        method=value.method,
        selected_candidate_id=value.candidate_id,
        selected_seed=value.seed,
        selected_checkpoint_index=value.checkpoint_index,
        selected_svc_c=value.svc_c,
        selected_rbf_gamma=value.rbf_gamma,
        selected_snr_threshold=value.snr_threshold,
        selected_theta=value.theta,
        validation_metrics=value.validation_metrics,
        confidence_intervals=_bootstrap_intervals(selected, labels, protocol),
    )


def evaluate_comparison(
    comparison: ComparisonInput,
    protocol: EvaluationProtocol,
    primary_qng_run: TrainingRunArtifact,
    *,
    test_ledger_sha256: str,
) -> EvaluationOutcome:
    validate_evaluation_protocol(protocol)
    started = perf_counter()
    groups: dict[MethodName, list[_ScoredCandidate]] = {}
    runtimes: list[MethodRuntime] = []

    method_started = perf_counter()
    groups["snr_threshold"] = _evaluate_snr(comparison)
    runtimes.append(
        MethodRuntime(
            method="snr_threshold",
            wall_time_ms=(perf_counter() - method_started) * 1_000.0,
            evaluated_kernel_coordinates=0,
        )
    )

    scaler, classical_train, classical_validation = _fit_classical_scaler(comparison)
    method_started = perf_counter()
    groups["rbf_svc_circular"] = _evaluate_rbf(
        comparison,
        protocol,
        classical_train,
        classical_validation,
    )
    runtimes.append(
        MethodRuntime(
            method="rbf_svc_circular",
            wall_time_ms=(perf_counter() - method_started) * 1_000.0,
            evaluated_kernel_coordinates=0,
        )
    )

    method_started = perf_counter()
    fixed_checkpoints = tuple(
        ThetaCheckpoint(
            seed=seed,
            checkpoint_index=0,
            theta=_theta0(seed),
            train_alignment_loss=_alignment(
                StateEngine("numpy"),
                comparison.X_train_encoded,
                comparison.y_train,
                _theta0(seed),
            ),
        )
        for seed in protocol.seed_schedule
    )
    groups["fixed_theta_tqk"] = _evaluate_quantum_checkpoints(
        "fixed_theta_tqk",
        fixed_checkpoints,
        comparison,
        protocol,
    )
    runtimes.append(
        MethodRuntime(
            method="fixed_theta_tqk",
            wall_time_ms=(perf_counter() - method_started) * 1_000.0,
            evaluated_kernel_coordinates=len(fixed_checkpoints),
        )
    )

    method_started = perf_counter()
    gradient_checkpoints = tuple(
        checkpoint
        for seed in protocol.seed_schedule
        for checkpoint in _train_gradient_checkpoints(
            comparison.X_train_encoded,
            comparison.y_train,
            seed,
            protocol,
        )
    )
    groups["gradient_tqk"] = _evaluate_quantum_checkpoints(
        "gradient_tqk",
        gradient_checkpoints,
        comparison,
        protocol,
    )
    runtimes.append(
        MethodRuntime(
            method="gradient_tqk",
            wall_time_ms=(perf_counter() - method_started) * 1_000.0,
            evaluated_kernel_coordinates=len(gradient_checkpoints),
        )
    )

    method_started = perf_counter()
    qng_checkpoints = list(_qng_checkpoints_from_primary_run(primary_qng_run, comparison))
    for seed in protocol.seed_schedule[1:]:
        qng_checkpoints.extend(
            _train_qng_checkpoints(
                comparison.X_train_encoded,
                comparison.y_train,
                seed,
                protocol,
            )
        )
    groups["qng_tqk"] = _evaluate_quantum_checkpoints(
        "qng_tqk",
        tuple(qng_checkpoints),
        comparison,
        protocol,
    )
    runtimes.append(
        MethodRuntime(
            method="qng_tqk",
            wall_time_ms=(perf_counter() - method_started) * 1_000.0,
            evaluated_kernel_coordinates=len(qng_checkpoints),
        )
    )

    selections = tuple(
        _select(groups[method], comparison.y_validation, protocol) for method in METHODS
    )
    candidates = tuple(
        item.artifact for method in METHODS for item in groups[method]
    )
    scientific = {
        "schema_version": "aqse.comparative-evaluation.v1",
        "scientific_scope": "TRAIN/VALIDATION selection only; TEST sealed",
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol.content_digest,
        "comparison_input": comparison.identity.model_dump(mode="json"),
        "classical_preprocessing": scaler.model_dump(mode="json"),
        "candidate_count": len(candidates),
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "selections": [item.model_dump(mode="json") for item in selections],
        "selected_method_order": METHODS,
        "qng_primary_seed_source_run_id": "aqse-qng-run-f00c702ad790df2b",
        "test_state_after": "sealed",
        "test_ledger_sha256_after": test_ledger_sha256,
    }
    digest = _digest(scientific)
    artifact = ComparativeEvaluationArtifact.model_validate(
        {
            **scientific,
            "evaluation_id": f"aqse-comparative-evaluation-{digest[:16]}",
            "content_digest": digest,
        }
    )
    validate_comparative_evaluation(artifact)
    return EvaluationOutcome(
        artifact=artifact,
        method_runtimes=tuple(runtimes),
        total_wall_time_ms=(perf_counter() - started) * 1_000.0,
    )


def validate_comparative_evaluation(artifact: ComparativeEvaluationArtifact) -> None:
    for candidate in artifact.candidates:
        validate_candidate(candidate)
    scientific = artifact.model_dump(
        mode="json",
        exclude={"evaluation_id", "content_digest"},
    )
    digest = _digest(scientific)
    if artifact.content_digest != digest:
        raise ValueError("comparative evaluation digest is invalid")
    if artifact.evaluation_id != f"aqse-comparative-evaluation-{digest[:16]}":
        raise ValueError("comparative evaluation identity is invalid")


def build_model_selection_freeze(
    protocol: EvaluationProtocol,
    evaluation: ComparativeEvaluationArtifact,
) -> ModelSelectionFreeze:
    validate_evaluation_protocol(protocol)
    validate_comparative_evaluation(evaluation)
    if (
        evaluation.protocol_id != protocol.protocol_id
        or evaluation.protocol_digest != protocol.content_digest
    ):
        raise ValueError("selection freeze protocol/evaluation mismatch")
    procedure = FinalEvaluationProcedure(
        metrics=(
            "balanced_accuracy",
            "macro_f1",
            "confusion_matrix",
            "per_class_recall",
            "roc_auc_when_both_classes_present",
            "eligibility_and_abstention",
            "actual_compute_cost",
        )
    )
    scientific = {
        "schema_version": "aqse.model-selection-freeze.v1",
        "protocol_id": protocol.protocol_id,
        "protocol_digest": protocol.content_digest,
        "evaluation_id": evaluation.evaluation_id,
        "evaluation_digest": evaluation.content_digest,
        "selections": [item.model_dump(mode="json") for item in evaluation.selections],
        "final_evaluation_procedure": procedure.model_dump(mode="json"),
        "test_state": "sealed",
        "test_ledger_sha256": evaluation.test_ledger_sha256_after,
    }
    digest = _digest(scientific)
    freeze = ModelSelectionFreeze.model_validate(
        {
            **scientific,
            "freeze_id": f"aqse-model-selection-freeze-{digest[:16]}",
            "content_digest": digest,
        }
    )
    validate_model_selection_freeze(freeze)
    return freeze


def validate_model_selection_freeze(freeze: ModelSelectionFreeze) -> None:
    scientific = freeze.model_dump(
        mode="json",
        exclude={"freeze_id", "content_digest"},
    )
    digest = _digest(scientific)
    if freeze.content_digest != digest:
        raise ValueError("model-selection freeze digest is invalid")
    if freeze.freeze_id != f"aqse-model-selection-freeze-{digest[:16]}":
        raise ValueError("model-selection freeze identity is invalid")
