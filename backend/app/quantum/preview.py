from __future__ import annotations

import hashlib
import json
from time import perf_counter
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.features.models import FeatureProfile, WindowFeatureRecord
from app.features.provenance import verify_window_record
from app.quantum.adapter import BackendType, TQK8Adapter
from app.quantum.user_pipeline.tqk8 import (
    EDGES,
    N_PARAMS,
    N_QUBITS,
    AngleScaler,
    build_vqc,
)

MAX_PREVIEW_SAMPLES = 128
DEFAULT_PREVIEW_SAMPLES = 16
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PreviewMode = Literal["self_reference", "reference_query"]


class QuantumPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mode: PreviewMode = "self_reference"
    backend: BackendType = "qiskit"
    feature_profile: FeatureProfile
    reference_dataset_id: str = Field(min_length=1, max_length=200)
    reference_windows: list[WindowFeatureRecord] = Field(
        min_length=2,
        max_length=MAX_PREVIEW_SAMPLES,
    )
    query_windows: list[WindowFeatureRecord] = Field(
        default_factory=list,
        max_length=MAX_PREVIEW_SAMPLES,
    )
    theta: tuple[FiniteFloat, ...] = Field(min_length=N_PARAMS, max_length=N_PARAMS)

    @model_validator(mode="after")
    def validate_preview_selection(self) -> QuantumPreviewRequest:
        if self.mode == "self_reference" and self.query_windows:
            raise ValueError("self_reference mode does not accept query_windows")
        if self.mode == "reference_query" and not self.query_windows:
            raise ValueError("reference_query mode requires query_windows")
        if len(self.reference_windows) + len(self.query_windows) > MAX_PREVIEW_SAMPLES:
            raise ValueError(
                f"a quantum preview may contain at most {MAX_PREVIEW_SAMPLES} windows"
            )

        reference_ids = [window.window_id for window in self.reference_windows]
        query_ids = [window.window_id for window in self.query_windows]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("reference window IDs must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query window IDs must be unique")
        if set(reference_ids).intersection(query_ids):
            raise ValueError("reference and query window IDs must be disjoint")
        if self.mode == "reference_query":
            shared_acquisitions = {
                window.acquisition_id for window in self.reference_windows
            }.intersection(window.acquisition_id for window in self.query_windows)
            for acquisition_id in shared_acquisitions:
                latest_reference_time = max(
                    window.end_time_s
                    for window in self.reference_windows
                    if window.acquisition_id == acquisition_id
                )
                earliest_query_time = min(
                    window.start_time_s
                    for window in self.query_windows
                    if window.acquisition_id == acquisition_id
                )
                if latest_reference_time >= earliest_query_time:
                    raise ValueError(
                        "reference windows from a shared acquisition must be "
                        "strictly causal and precede all query windows"
                    )

        invalid = [
            window.window_id
            for window in [*self.reference_windows, *self.query_windows]
            if (
                not window.quality.valid_for_quantum
                or not all(window.quality.per_feature_valid)
                or window.quality.saturation_fraction != 0.0
            )
        ]
        if invalid:
            raise ValueError(
                "quantum preview accepts only valid windows; rejected IDs: "
                + ", ".join(invalid)
            )
        for window in [*self.reference_windows, *self.query_windows]:
            if window.features.names != self.feature_profile.feature_names:
                raise ValueError(
                    f"window {window.window_id} feature names do not match the profile"
                )
            if window.features.units != self.feature_profile.feature_units:
                raise ValueError(
                    f"window {window.window_id} feature units do not match the profile"
                )
        return self


class AngleScalerSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    algorithm: str = "AQSE TQK8 AngleScaler"
    version: str
    reference_dataset_id: str
    feature_profile_id: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]


class KernelDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    minimum: float
    maximum: float
    mean_off_diagonal: float
    maximum_diagonal_deviation: float
    maximum_symmetry_deviation: float
    minimum_eigenvalue: float


class QuantumPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    preview_id: str
    mode: PreviewMode
    backend: str
    feature_profile: FeatureProfile
    reference_window_ids: list[str]
    query_window_ids: list[str]
    raw_reference_features: list[list[float]]
    encoded_reference_angles: list[list[float]]
    raw_query_features: list[list[float]]
    encoded_query_angles: list[list[float]]
    executed_theta: tuple[float, ...]
    scaler: AngleScalerSnapshot
    reference_kernel: list[list[float]]
    query_reference_kernel: list[list[float]] | None
    diagnostics: KernelDiagnostics
    execution_duration_ms: float = Field(ge=0.0)
    scientific_scope: str = "Fixed-theta infrastructure preview; no QNG training"


class CircuitOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int
    stage: str
    gate: str
    qubits: tuple[int, ...]
    parameters: tuple[str, ...]


class CircuitDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    qubits: int
    input_features: int
    trainable_parameters: int
    feature_uploads_per_feature: int
    trainable_groups: dict[str, tuple[str, ...]]
    cz_edges: tuple[tuple[int, int], ...]
    operations: list[CircuitOperation]


def _matrix(rows: list[WindowFeatureRecord]) -> np.ndarray:
    matrix = np.asarray([row.features.values for row in rows], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != N_QUBITS:
        raise ValueError("feature matrix must have shape (number_of_windows, 8)")
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix must contain only finite values")
    return matrix


def _validate_reference_diversity(matrix: np.ndarray) -> None:
    """Reject reference banks with no physically meaningful feature variation."""

    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    feature_scale = np.maximum(np.max(np.abs(matrix), axis=0), 1.0)
    relative_centered = centered / feature_scale
    singular_values = np.linalg.svd(relative_centered, compute_uv=False)
    relative_tolerance = 1.0e-9
    if singular_values.size == 0 or float(singular_values[0]) <= relative_tolerance:
        raise ValueError(
            "reference dataset is degenerate; provide windows with meaningful "
            "feature variation"
        )


def _scaler_version(
    scaler: AngleScaler,
    dataset_id: str,
    profile: FeatureProfile,
) -> str:
    payload = json.dumps(
        {
            "algorithm": "AQSE TQK8 AngleScaler",
            "dataset_id": dataset_id,
            "feature_profile": profile.model_dump(mode="json"),
            "mean": np.asarray(scaler.mean, dtype=float).tolist(),
            "scale": np.asarray(scaler.scale, dtype=float).tolist(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "anglescaler-" + hashlib.sha256(payload).hexdigest()[:16]


def _diagnostics(kernel: np.ndarray) -> KernelDiagnostics:
    if kernel.shape[0] != kernel.shape[1] or not np.isfinite(kernel).all():
        raise RuntimeError("TQK8 returned an invalid reference kernel")
    symmetry = float(np.max(np.abs(kernel - kernel.T)))
    diagonal_deviation = float(np.max(np.abs(np.diag(kernel) - 1.0)))
    tolerance = 1.0e-10
    if float(np.min(kernel)) < -tolerance or float(np.max(kernel)) > 1.0 + tolerance:
        raise RuntimeError("TQK8 fidelity kernel is outside its numerical range")
    if symmetry > tolerance or diagonal_deviation > tolerance:
        raise RuntimeError("TQK8 fidelity kernel failed numerical invariants")

    off_diagonal = kernel[~np.eye(kernel.shape[0], dtype=bool)]
    eigenvalues = np.linalg.eigvalsh((kernel + kernel.T) / 2.0)
    if float(eigenvalues[0]) < -tolerance:
        raise RuntimeError("TQK8 reference kernel is not positive semidefinite")
    return KernelDiagnostics(
        minimum=float(np.min(kernel)),
        maximum=float(np.max(kernel)),
        mean_off_diagonal=float(np.mean(off_diagonal)),
        maximum_diagonal_deviation=diagonal_deviation,
        maximum_symmetry_deviation=symmetry,
        minimum_eigenvalue=float(eigenvalues[0]),
    )


def run_quantum_preview(request: QuantumPreviewRequest) -> QuantumPreviewResponse:
    """Run a bounded, fixed-theta TQK8 preview without invoking QNG."""

    started = perf_counter()
    forged = [
        row.window_id
        for row in [*request.reference_windows, *request.query_windows]
        if not verify_window_record(request.feature_profile, row)
    ]
    if forged:
        raise ValueError(
            "feature provenance validation failed; rerun feature extraction for: "
            + ", ".join(forged)
        )
    raw_reference = _matrix(request.reference_windows)
    raw_query = _matrix(request.query_windows) if request.query_windows else None
    _validate_reference_diversity(raw_reference)
    theta = np.asarray(request.theta, dtype=np.float64).copy()

    scaler = AngleScaler.fit(raw_reference)
    encoded_reference = np.asarray(scaler.transform(raw_reference), dtype=np.float64)
    encoded_query = (
        np.asarray(scaler.transform(raw_query), dtype=np.float64)
        if raw_query is not None
        else None
    )
    engine = TQK8Adapter(request.backend)
    reference_kernel = engine.gram(encoded_reference, theta)
    query_kernel = (
        engine.cross_gram(encoded_query, encoded_reference, theta)
        if encoded_query is not None
        else None
    )
    if query_kernel is not None:
        if not np.isfinite(query_kernel).all():
            raise RuntimeError("TQK8 returned a non-finite query kernel")
        tolerance = 1.0e-10
        if (
            float(np.min(query_kernel)) < -tolerance
            or float(np.max(query_kernel)) > 1.0 + tolerance
        ):
            raise RuntimeError("TQK8 query fidelities are outside their numerical range")
    diagnostics = _diagnostics(reference_kernel)
    duration_ms = (perf_counter() - started) * 1_000.0

    scaler_version = _scaler_version(
        scaler,
        request.reference_dataset_id,
        request.feature_profile,
    )
    identity_payload = json.dumps(
        {
            "reference": [row.window_id for row in request.reference_windows],
            "query": [row.window_id for row in request.query_windows],
            "theta": theta.tolist(),
            "backend": request.backend,
            "scaler": scaler_version,
            "raw_reference": raw_reference.tolist(),
            "raw_query": [] if raw_query is None else raw_query.tolist(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    preview_id = "preview-" + hashlib.sha256(identity_payload).hexdigest()[:16]

    return QuantumPreviewResponse(
        preview_id=preview_id,
        mode=request.mode,
        backend=engine.metadata.backend_type,
        feature_profile=request.feature_profile,
        reference_window_ids=[row.window_id for row in request.reference_windows],
        query_window_ids=[row.window_id for row in request.query_windows],
        raw_reference_features=raw_reference.tolist(),
        encoded_reference_angles=encoded_reference.tolist(),
        raw_query_features=[] if raw_query is None else raw_query.tolist(),
        encoded_query_angles=[] if encoded_query is None else encoded_query.tolist(),
        executed_theta=tuple(float(value) for value in theta),
        scaler=AngleScalerSnapshot(
            version=scaler_version,
            reference_dataset_id=request.reference_dataset_id,
            feature_profile_id=request.feature_profile.profile_id,
            mean=tuple(float(value) for value in scaler.mean),
            scale=tuple(float(value) for value in scaler.scale),
        ),
        reference_kernel=reference_kernel.tolist(),
        query_reference_kernel=None if query_kernel is None else query_kernel.tolist(),
        diagnostics=diagnostics,
        execution_duration_ms=duration_ms,
    )


def _stage(gate: str, parameters: tuple[str, ...]) -> str:
    parameter = parameters[0] if parameters else ""
    if gate == "cz":
        return "entanglement"
    if gate == "ry" and parameter.startswith("x["):
        return "feature_upload_1"
    if gate == "rz" and parameter.startswith("theta["):
        return "alpha"
    if gate == "ry" and parameter.startswith("theta["):
        return "beta"
    if gate == "rz" and parameter.startswith("x["):
        return "feature_upload_2"
    return "unclassified"


def describe_tqk8_circuit() -> CircuitDescription:
    """Describe the actual built circuit for a technically faithful UI."""

    circuit, _, _ = build_vqc()
    operations: list[CircuitOperation] = []
    for order, instruction in enumerate(circuit.data):
        gate = instruction.operation.name
        parameters = tuple(str(parameter) for parameter in instruction.operation.params)
        operations.append(
            CircuitOperation(
                order=order,
                stage=_stage(gate, parameters),
                gate=gate,
                qubits=tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
                parameters=parameters,
            )
        )

    actual_edges = tuple(
        operation.qubits for operation in operations if operation.gate == "cz"
    )
    expected_stage_counts = {
        "feature_upload_1": N_QUBITS,
        "alpha": N_QUBITS,
        "entanglement": len(EDGES),
        "beta": N_QUBITS,
        "feature_upload_2": N_QUBITS,
    }
    actual_stage_counts = {
        stage: sum(operation.stage == stage for operation in operations)
        for stage in expected_stage_counts
    }
    if actual_edges != EDGES or actual_stage_counts != expected_stage_counts:
        raise RuntimeError("TQK8 circuit description does not match the scientific source")

    return CircuitDescription(
        name=circuit.name,
        qubits=N_QUBITS,
        input_features=N_QUBITS,
        trainable_parameters=N_PARAMS,
        feature_uploads_per_feature=2,
        trainable_groups={
            "alpha": tuple(f"theta[{index}]" for index in range(N_QUBITS)),
            "beta": tuple(
                f"theta[{index}]" for index in range(N_QUBITS, N_PARAMS)
            ),
        },
        cz_edges=EDGES,
        operations=operations,
    )
