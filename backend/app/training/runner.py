from __future__ import annotations

import hashlib
import resource
import sys
from collections.abc import Callable
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.quantum.user_pipeline.tqk8 import (
    N_PARAMS,
    StateEngine,
    alignment_loss,
    fit_qng,
    loss_grad_metric,
)
from app.training.assembly import TrainingInput, validate_training_input_identity
from app.training.canonical import canonical_json_bytes, file_sha256
from app.training.run_models import (
    EvaluationCounters,
    OptimizerContract,
    TrainingRunArtifact,
    TrainingStepRecord,
)

TQK8_SOURCE_SHA256 = "cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689"
THETA_INIT_POLICY_ID = "aqse.theta-init.uniform-v1"
THETA_SEED = 1_001_005
MAXIMUM_UPDATES = 10
LEARNING_RATE = 0.2
DAMPING = 1.0e-3
MAXIMUM_STEP_NORM = 0.4
KERNEL_TOLERANCE = 1.0e-12
STATE_TOLERANCE = 1.0e-12
CROSS_ENGINE_TOLERANCE = 1.0e-10
ProtectedOptimizer = Callable[..., tuple[NDArray[np.float64], list[dict[str, Any]]]]


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _effective_source_hashes() -> dict[str, str]:
    root = _backend_root()
    paths = (
        "app/quantum/user_pipeline/tqk8.py",
        "app/training/assembly.py",
        "app/training/run_models.py",
        "app/training/runner.py",
    )
    return {path: file_sha256(root / path) for path in paths}


def canonical_theta0() -> NDArray[np.float64]:
    theta = np.random.default_rng(THETA_SEED).uniform(-0.8, 0.8, N_PARAMS)
    theta = np.asarray(theta, dtype=np.float64)
    theta.setflags(write=False)
    return theta


def _validate_kernel(kernel: NDArray[Any]) -> None:
    value = np.asarray(kernel, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.isfinite(value).all():
        raise ValueError("QNG produced a non-square or non-finite TRAIN kernel")
    if float(np.min(value)) < -KERNEL_TOLERANCE or float(np.max(value)) > 1.0 + KERNEL_TOLERANCE:
        raise ValueError("QNG TRAIN fidelity kernel is outside its numerical range")
    if float(np.max(np.abs(value - value.T))) > KERNEL_TOLERANCE:
        raise ValueError("QNG TRAIN fidelity kernel is not symmetric")
    if float(np.max(np.abs(np.diag(value) - 1.0))) > KERNEL_TOLERANCE:
        raise ValueError("QNG TRAIN fidelity kernel diagonal differs from one")
    if float(np.linalg.eigvalsh((value + value.T) / 2.0)[0]) < -KERNEL_TOLERANCE:
        raise ValueError("QNG TRAIN fidelity kernel is not numerically PSD")


class CountingExactEngine:
    """Transparent counter/validator around the protected exact-state engine."""

    def __init__(self, backend: str = "numpy") -> None:
        self._engine = StateEngine(backend)
        self.backend = backend
        self.differential_calls = 0
        self.gram_calls = 0
        self.statevector_evaluations = 0

    def counters(self) -> EvaluationCounters:
        return EvaluationCounters(
            differential_calls=self.differential_calls,
            gram_calls=self.gram_calls,
            statevector_evaluations=self.statevector_evaluations,
        )

    def gram(
        self,
        X: NDArray[Any],
        theta: NDArray[Any],
        Z: NDArray[Any] | None = None,
    ) -> NDArray[np.float64]:
        result = np.asarray(self._engine.gram(X, theta, Z), dtype=np.float64)
        self.gram_calls += 1
        self.statevector_evaluations += len(X) + (0 if Z is None else len(Z))
        if Z is None:
            _validate_kernel(result)
        elif not np.isfinite(result).all():
            raise ValueError("QNG cross-kernel contains non-finite values")
        return result

    def differential(
        self,
        X: NDArray[Any],
        theta: NDArray[Any],
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        states, derivatives = self._engine.differential(X, theta)
        self.differential_calls += 1
        self.statevector_evaluations += len(X) * (1 + 2 * N_PARAMS)
        _validate_kernel(np.abs(states.conj() @ states.T) ** 2)
        return states, derivatives


def _counter_delta(
    after: EvaluationCounters,
    before: EvaluationCounters,
) -> EvaluationCounters:
    return EvaluationCounters(
        differential_calls=after.differential_calls - before.differential_calls,
        gram_calls=after.gram_calls - before.gram_calls,
        statevector_evaluations=(
            after.statevector_evaluations - before.statevector_evaluations
        ),
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1_024


def _artifact_payload(artifact: TrainingRunArtifact) -> dict[str, Any]:
    return artifact.model_dump(mode="json", exclude={"run_id", "content_digest"})


def validate_training_run_artifact(artifact: TrainingRunArtifact) -> None:
    validate_training_input_identity(artifact.training_input)
    sources = _effective_source_hashes()
    if artifact.effective_source_hashes != sources:
        raise ValueError("training-run effective source identity is incompatible")
    if sources["app/quantum/user_pipeline/tqk8.py"] != TQK8_SOURCE_SHA256:
        raise ValueError("protected TQK8 source identity differs from 1D.3")
    expected = _digest(_artifact_payload(artifact))
    if artifact.content_digest != expected:
        raise ValueError("training-run content digest is invalid")
    if artifact.run_id != f"aqse-qng-run-{expected[:16]}":
        raise ValueError("training-run identity is invalid")


def run_bounded_qng(
    training_input: TrainingInput,
    *,
    cancellation: Event | None = None,
    optimizer: ProtectedOptimizer = fit_qng,
) -> TrainingRunArtifact:
    if training_input.X_train.shape != (32, 8) or training_input.y_train.shape != (32,):
        raise ValueError("AQSE 1D.3 training is hard-limited to 32 rows")
    cancellation = cancellation or Event()
    engine = CountingExactEngine("numpy")
    theta0 = canonical_theta0()
    theta = theta0.copy()
    started = perf_counter()
    initial_kernel = engine.gram(training_input.X_train, theta)
    initial_loss = float(alignment_loss(initial_kernel, training_input.y_train)[0])
    steps: list[TrainingStepRecord] = []
    stop_reason = "MAX_UPDATES_REACHED"
    for step_index in range(MAXIMUM_UPDATES):
        if cancellation.is_set():
            stop_reason = "CANCELLED"
            break
        theta_before = theta.copy()
        counters_before = engine.counters()
        step_started = perf_counter()
        theta_after, history = optimizer(
            engine,
            training_input.X_train,
            training_input.y_train,
            theta_before,
            steps=1,
            learning_rate=LEARNING_RATE,
            damping=DAMPING,
            max_step=MAXIMUM_STEP_NORM,
            verbose=False,
        )
        step_wall_time_ms = (perf_counter() - step_started) * 1_000.0
        if not history:
            stop_reason = "NO_ACCEPTED_UPDATE"
            break
        if len(history) != 1:
            raise RuntimeError("protected one-step QNG returned multiple history rows")
        row = history[0]
        theta = np.asarray(theta_after, dtype=np.float64)
        steps.append(
            TrainingStepRecord(
                step_index=step_index,
                theta_before=tuple(float(value) for value in theta_before),
                theta_after=tuple(float(value) for value in theta),
                loss_before=float(row["loss_before"]),
                loss_after=float(row["loss_after"]),
                gradient_norm=float(row["gradient_norm"]),
                step_norm=float(row["step_norm"]),
                metric_min_eigenvalue=float(row["metric_min_eigenvalue"]),
                step_wall_time_ms=step_wall_time_ms,
                counters_delta=_counter_delta(engine.counters(), counters_before),
            )
        )
    total_wall_time_ms = (perf_counter() - started) * 1_000.0
    final_loss = steps[-1].loss_after if steps else initial_loss
    scientific = {
        "schema_version": "aqse.qng-training-run.v1",
        "scientific_scope": "frozen TRAIN candidate-theta trajectory only",
        "training_input": training_input.identity.model_dump(mode="json"),
        "backend_semantics": "numpy_statevector",
        "exact_simulator": True,
        "tqk8_source_sha256": TQK8_SOURCE_SHA256,
        "theta_init_policy_id": THETA_INIT_POLICY_ID,
        "theta_seed": THETA_SEED,
        "theta0": theta0.tolist(),
        "optimizer": OptimizerContract().model_dump(mode="json"),
        "requested_maximum_updates": MAXIMUM_UPDATES,
        "accepted_update_count": len(steps),
        "stop_reason": stop_reason,
        "initial_train_alignment_loss": initial_loss,
        "final_train_alignment_loss": final_loss,
        "steps": [step.model_dump(mode="json") for step in steps],
        "final_theta": theta.tolist(),
        "total_wall_time_ms": total_wall_time_ms,
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "counters": engine.counters().model_dump(mode="json"),
        "effective_source_hashes": _effective_source_hashes(),
    }
    digest = _digest(scientific)
    artifact = TrainingRunArtifact.model_validate(
        {
            **scientific,
            "run_id": f"aqse-qng-run-{digest[:16]}",
            "content_digest": digest,
        }
    )
    validate_training_run_artifact(artifact)
    return artifact


def wrapper_equivalence(
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    theta0: NDArray[np.float64],
    *,
    steps: int = 3,
) -> dict[str, float | int]:
    direct_theta, direct_history = fit_qng(
        StateEngine("numpy"),
        X,
        y,
        theta0,
        steps=steps,
        learning_rate=LEARNING_RATE,
        damping=DAMPING,
        max_step=MAXIMUM_STEP_NORM,
        verbose=False,
    )
    repeated_theta = np.asarray(theta0, dtype=np.float64).copy()
    repeated_history: list[dict[str, Any]] = []
    for _ in range(steps):
        repeated_theta, rows = fit_qng(
            StateEngine("numpy"),
            X,
            y,
            repeated_theta,
            steps=1,
            learning_rate=LEARNING_RATE,
            damping=DAMPING,
            max_step=MAXIMUM_STEP_NORM,
            verbose=False,
        )
        if not rows:
            break
        repeated_history.append(rows[0])
    if len(direct_history) != len(repeated_history):
        raise ValueError("repeated one-step QNG accepted-step count differs from N-step QNG")
    fields = (
        "loss_before",
        "loss_after",
        "gradient_norm",
        "step_norm",
        "metric_min_eigenvalue",
    )
    field_delta = max(
        (
            abs(float(left[field]) - float(right[field]))
            for left, right in zip(direct_history, repeated_history, strict=True)
            for field in fields
        ),
        default=0.0,
    )
    theta_delta = float(np.max(np.abs(direct_theta - repeated_theta)))
    if theta_delta > STATE_TOLERANCE or field_delta > STATE_TOLERANCE:
        raise ValueError("repeated one-step QNG is not equivalent to protected N-step QNG")
    return {
        "requested_steps": steps,
        "accepted_steps": len(direct_history),
        "final_theta_max_abs_delta": theta_delta,
        "history_max_abs_delta": field_delta,
        "absolute_tolerance": STATE_TOLERANCE,
    }


def cross_engine_one_step_gate(
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    theta0: NDArray[np.float64],
) -> dict[str, float | int]:
    numpy_engine = StateEngine("numpy")
    qiskit_engine = StateEngine("qiskit")
    numpy_loss, numpy_gradient, numpy_metric, numpy_kernel = loss_grad_metric(
        numpy_engine, X, y, theta0
    )
    qiskit_loss, qiskit_gradient, qiskit_metric, qiskit_kernel = loss_grad_metric(
        qiskit_engine, X, y, theta0
    )
    numpy_theta, numpy_history = fit_qng(
        numpy_engine,
        X,
        y,
        theta0,
        steps=1,
        learning_rate=LEARNING_RATE,
        damping=DAMPING,
        max_step=MAXIMUM_STEP_NORM,
        verbose=False,
    )
    qiskit_theta, qiskit_history = fit_qng(
        qiskit_engine,
        X,
        y,
        theta0,
        steps=1,
        learning_rate=LEARNING_RATE,
        damping=DAMPING,
        max_step=MAXIMUM_STEP_NORM,
        verbose=False,
    )
    if len(numpy_history) != 1 or len(qiskit_history) != 1:
        raise ValueError("cross-engine gate requires one accepted update from both engines")
    result = {
        "row_count": len(X),
        "kernel_max_abs_delta": float(np.max(np.abs(numpy_kernel - qiskit_kernel))),
        "initial_loss_abs_delta": abs(float(numpy_loss) - float(qiskit_loss)),
        "gradient_max_abs_delta": float(
            np.max(np.abs(numpy_gradient - qiskit_gradient))
        ),
        "metric_max_abs_delta": float(np.max(np.abs(numpy_metric - qiskit_metric))),
        "candidate_theta_max_abs_delta": float(
            np.max(np.abs(numpy_theta - qiskit_theta))
        ),
        "state_kernel_tolerance": STATE_TOLERANCE,
        "gradient_metric_update_tolerance": CROSS_ENGINE_TOLERANCE,
    }
    if result["kernel_max_abs_delta"] > STATE_TOLERANCE:
        raise ValueError("Qiskit/NumPy initial kernel agreement failed")
    for key in (
        "initial_loss_abs_delta",
        "gradient_max_abs_delta",
        "metric_max_abs_delta",
        "candidate_theta_max_abs_delta",
    ):
        if result[key] > CROSS_ENGINE_TOLERANCE:
            raise ValueError(f"Qiskit/NumPy cross-engine gate failed: {key}")
    return result
