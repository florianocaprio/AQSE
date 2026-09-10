from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Event
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from app.demo.protocol import FROZEN_NETWORK_DEMO_PROTOCOL
from app.quantum.user_pipeline.tqk8 import (
    N_PARAMS,
    StateEngine,
    alignment_loss,
    fit_qng,
    loss_grad_metric,
)
from app.training.canonical import canonical_json_bytes


@dataclass(frozen=True)
class DemoQngStep:
    step_index: int
    loss_before: float
    loss_after: float
    gradient_norm: float
    step_norm: float
    metric_min_eigenvalue: float
    theta: tuple[float, ...]
    wall_time_ms: float


@dataclass(frozen=True)
class DemoQngCandidate:
    candidate_id: str
    eligible: bool
    profile_id: str
    bank_sample_ids: tuple[str, ...]
    bank_lineage_ids: tuple[str, ...]
    theta0: tuple[float, ...]
    final_theta: tuple[float, ...]
    accepted_updates: int
    stop_reason: str
    initial_loss: float | None
    final_loss: float | None
    steps: tuple[DemoQngStep, ...]
    total_wall_time_ms: float

    def scientific_payload(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "eligible": self.eligible,
            "bank_sample_ids": self.bank_sample_ids,
            "bank_lineage_ids": self.bank_lineage_ids,
            "theta0": self.theta0,
            "final_theta": self.final_theta,
            "accepted_updates": self.accepted_updates,
            "stop_reason": self.stop_reason,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "steps": [
                {
                    "step_index": item.step_index,
                    "loss_before": item.loss_before,
                    "loss_after": item.loss_after,
                    "gradient_norm": item.gradient_norm,
                    "step_norm": item.step_norm,
                    "metric_min_eigenvalue": item.metric_min_eigenvalue,
                    "theta": item.theta,
                }
                for item in self.steps
            ],
        }


def canonical_demo_theta0() -> NDArray[np.float64]:
    random = np.random.default_rng(FROZEN_NETWORK_DEMO_PROTOCOL.seeds.theta)
    return random.uniform(-0.8, 0.8, N_PARAMS).astype(np.float64)


def balanced_binary_bank_indices(
    labels: Sequence[int],
    lineage_ids: Sequence[str],
    *,
    maximum_rows: int = 32,
) -> NDArray[np.int64]:
    values = np.asarray(labels, dtype=np.int8)
    if values.ndim != 1 or len(values) != len(lineage_ids):
        raise ValueError("binary labels and lineages must be aligned")
    if set(np.unique(values).tolist()) != {-1, 1}:
        raise ValueError("QNG alignment bank requires labels -1 and +1")
    if len(set(lineage_ids)) != len(lineage_ids):
        raise ValueError("QNG bank lineages must be distinct")
    per_class = min(
        maximum_rows // 2,
        int(np.sum(values == -1)),
        int(np.sum(values == 1)),
    )
    if per_class < FROZEN_NETWORK_DEMO_PROTOCOL.quantum.minimum_examples_per_binary_class:
        return np.asarray([], dtype=np.int64)
    random = np.random.default_rng(
        FROZEN_NETWORK_DEMO_PROTOCOL.seeds.bank_and_landmarks
    )
    selected: list[int] = []
    for label in (-1, 1):
        candidates = np.flatnonzero(values == label)
        order = np.argsort(np.asarray([lineage_ids[index] for index in candidates]))
        candidates = candidates[order]
        selected.extend(int(item) for item in random.permutation(candidates)[:per_class])
    return np.asarray(selected, dtype=np.int64)


def _candidate_id(payload: dict[str, object]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"aqse-demo-qng-{digest[:16]}"


def train_demo_qng(
    encoded_features: NDArray[np.float64],
    binary_labels: Sequence[int],
    *,
    sample_ids: Sequence[str],
    lineage_ids: Sequence[str],
    profile_id: str,
    cancellation: Event | None = None,
    progress: Callable[[DemoQngStep], None] | None = None,
) -> DemoQngCandidate:
    matrix = np.asarray(encoded_features, dtype=np.float64)
    labels = np.asarray(binary_labels, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[1] != 8:
        raise ValueError("demo QNG inputs must have shape (N, 8)")
    if matrix.shape[0] != len(labels) or len(labels) != len(sample_ids):
        raise ValueError("demo QNG inputs, labels and identities must align")
    indices = balanced_binary_bank_indices(labels, lineage_ids)
    theta0 = canonical_demo_theta0()
    if indices.size == 0:
        payload = {
            "profile_id": profile_id,
            "eligible": False,
            "theta0": theta0.tolist(),
            "reason": "INSUFFICIENT_BINARY_LINEAGES",
        }
        return DemoQngCandidate(
            candidate_id=_candidate_id(payload),
            eligible=False,
            profile_id=profile_id,
            bank_sample_ids=(),
            bank_lineage_ids=(),
            theta0=tuple(float(item) for item in theta0),
            final_theta=tuple(float(item) for item in theta0),
            accepted_updates=0,
            stop_reason="INSUFFICIENT_BINARY_LINEAGES",
            initial_loss=None,
            final_loss=None,
            steps=(),
            total_wall_time_ms=0.0,
        )

    bank = matrix[indices]
    bank_labels = labels[indices]
    bank_sample_ids = tuple(sample_ids[int(index)] for index in indices)
    bank_lineages = tuple(lineage_ids[int(index)] for index in indices)
    engine = StateEngine("numpy")
    started = perf_counter()
    theta = theta0.copy()
    initial_loss, _ = alignment_loss(engine.gram(bank, theta), bank_labels)
    steps: list[DemoQngStep] = []
    stop_reason = "MAX_UPDATES_REACHED"
    cancellation = cancellation or Event()
    for step_index in range(
        FROZEN_NETWORK_DEMO_PROTOCOL.quantum.maximum_qng_updates
    ):
        if cancellation.is_set():
            stop_reason = "CANCELLED"
            break
        step_started = perf_counter()
        candidate, history = fit_qng(
            engine,
            bank,
            bank_labels,
            theta,
            steps=1,
            learning_rate=FROZEN_NETWORK_DEMO_PROTOCOL.quantum.learning_rate,
            damping=FROZEN_NETWORK_DEMO_PROTOCOL.quantum.damping,
            max_step=FROZEN_NETWORK_DEMO_PROTOCOL.quantum.maximum_step_norm,
            verbose=False,
        )
        if not history:
            stop_reason = "NO_ACCEPTED_UPDATE"
            break
        row = history[0]
        theta = np.asarray(candidate, dtype=np.float64)
        step = DemoQngStep(
            step_index=step_index,
            loss_before=float(row["loss_before"]),
            loss_after=float(row["loss_after"]),
            gradient_norm=float(row["gradient_norm"]),
            step_norm=float(row["step_norm"]),
            metric_min_eigenvalue=float(row["metric_min_eigenvalue"]),
            theta=tuple(float(item) for item in theta),
            wall_time_ms=(perf_counter() - step_started) * 1_000.0,
        )
        steps.append(step)
        if progress is not None:
            progress(step)
    final_loss = steps[-1].loss_after if steps else initial_loss
    draft = DemoQngCandidate(
        candidate_id="pending",
        eligible=True,
        profile_id=profile_id,
        bank_sample_ids=bank_sample_ids,
        bank_lineage_ids=bank_lineages,
        theta0=tuple(float(item) for item in theta0),
        final_theta=tuple(float(item) for item in theta),
        accepted_updates=len(steps),
        stop_reason=stop_reason,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        steps=tuple(steps),
        total_wall_time_ms=(perf_counter() - started) * 1_000.0,
    )
    return DemoQngCandidate(
        **{**draft.__dict__, "candidate_id": _candidate_id(draft.scientific_payload())}
    )


def qng_wrapper_equivalence(
    encoded_bank: NDArray[np.float64],
    labels: Sequence[int],
    *,
    steps: int = 2,
) -> dict[str, float | int]:
    matrix = np.asarray(encoded_bank, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    theta0 = canonical_demo_theta0()
    engine_batch = StateEngine("numpy")
    expected_theta, expected_history = fit_qng(
        engine_batch,
        matrix,
        y,
        theta0,
        steps=steps,
        learning_rate=0.2,
        damping=0.001,
        max_step=0.4,
        verbose=False,
    )
    engine_step = StateEngine("numpy")
    actual_theta = theta0.copy()
    actual_history: list[dict[str, float]] = []
    for _ in range(steps):
        actual_theta, rows = fit_qng(
            engine_step,
            matrix,
            y,
            actual_theta,
            steps=1,
            learning_rate=0.2,
            damping=0.001,
            max_step=0.4,
            verbose=False,
        )
        if not rows:
            break
        actual_history.extend(rows)
    theta_delta = float(np.max(np.abs(expected_theta - actual_theta)))
    if len(expected_history) != len(actual_history) or theta_delta > 1.0e-12:
        raise RuntimeError("protected fit_qng wrapper equivalence failed")
    return {
        "requested_steps": steps,
        "accepted_batch": len(expected_history),
        "accepted_stepwise": len(actual_history),
        "maximum_theta_delta": theta_delta,
    }


def qiskit_numpy_preflight(
    encoded_bank: NDArray[np.float64],
    labels: Sequence[int],
) -> dict[str, float]:
    matrix = np.asarray(encoded_bank, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    theta = canonical_demo_theta0()
    numpy_engine = StateEngine("numpy")
    qiskit_engine = StateEngine("qiskit")
    numpy_loss, numpy_grad, numpy_metric, numpy_kernel = loss_grad_metric(
        numpy_engine, matrix, y, theta
    )
    qiskit_loss, qiskit_grad, qiskit_metric, qiskit_kernel = loss_grad_metric(
        qiskit_engine, matrix, y, theta
    )
    result = {
        "kernel_maximum_delta": float(np.max(np.abs(numpy_kernel - qiskit_kernel))),
        "loss_delta": abs(float(numpy_loss - qiskit_loss)),
        "gradient_maximum_delta": float(np.max(np.abs(numpy_grad - qiskit_grad))),
        "metric_maximum_delta": float(
            np.max(np.abs(numpy_metric - qiskit_metric))
        ),
    }
    if (
        result["kernel_maximum_delta"] > 1.0e-12
        or result["loss_delta"] > 1.0e-10
        or result["gradient_maximum_delta"] > 1.0e-10
        or result["metric_maximum_delta"] > 1.0e-10
    ):
        raise RuntimeError("Qiskit/NumPy demo preflight failed")
    return result
