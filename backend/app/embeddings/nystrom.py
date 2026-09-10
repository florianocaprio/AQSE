from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.embeddings.contracts import LocalEmbeddingBatch
from app.quantum.adapter import BackendType, TQK8Adapter
from app.quantum.engine import QuantumEngine
from app.training.canonical import array_identity, canonical_json_bytes, file_sha256

AFSE_METHOD_ID = "aqse.afse.nystrom-ridge32.v1"
AFSE_SCHEMA_VERSION = "aqse.afse-artifact.v1"
LANDMARK_SELECTION_SEED = 2_001_003
MAX_LANDMARKS = 32
RIDGE_LAMBDA = 1.0e-6
EIGENVALUE_RELATIVE_TOLERANCE = 1.0e-10
RESIDUAL_NUMERICAL_TOLERANCE = 1.0e-10
FEATURE_COUNT = 8
PARAMETER_COUNT = 16


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class AFSEQueryContext(_FrozenModel):
    """Compatibility identity for encoded query rows, separate from acquisition IDs."""

    method_id: Literal["aqse.afse.nystrom-ridge32.v1"] = AFSE_METHOD_ID
    feature_profile_id: str
    scaler_id: str
    encoding_policy_id: str
    theta_id: str = Field(pattern=r"^aqse-theta-[a-f0-9]{16}$")
    feature_count: Literal[8] = FEATURE_COUNT


class NystromAFSEArtifact(_FrozenModel):
    """Non-executable, JSON-safe fitted Nyström representation artifact."""

    schema_version: Literal["aqse.afse-artifact.v1"] = AFSE_SCHEMA_VERSION
    artifact_id: str = Field(pattern=r"^aqse-afse-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    method_id: Literal["aqse.afse.nystrom-ridge32.v1"] = AFSE_METHOD_ID
    fitted_on_partition: Literal["TRAIN"] = "TRAIN"
    fitted_on_dataset_id: str
    fitted_on_dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_profile_id: str
    scaler_id: str
    encoding_policy_id: str
    feature_count: Literal[8] = FEATURE_COUNT
    theta_id: str = Field(pattern=r"^aqse-theta-[a-f0-9]{16}$")
    theta_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    theta: tuple[float, ...] = Field(min_length=PARAMETER_COUNT, max_length=PARAMETER_COUNT)
    fit_backend_type: Literal["numpy", "qiskit"]
    selection_seed: Literal[2001003] = LANDMARK_SELECTION_SEED
    selection_policy: Literal[
        "balanced-classes;one-distinct-TRAIN-lineage-per-landmark;seeded-v1"
    ] = "balanced-classes;one-distinct-TRAIN-lineage-per-landmark;seeded-v1"
    requested_reference_size: Literal[32] = MAX_LANDMARKS
    reference_size: int = Field(gt=0, le=MAX_LANDMARKS)
    output_dimension: int = Field(gt=0, le=MAX_LANDMARKS)
    landmark_sample_ids: tuple[str, ...]
    landmark_lineage_ids: tuple[str, ...]
    landmark_classes: tuple[str, ...]
    landmark_encoded: tuple[tuple[float, ...], ...]
    ridge_lambda: Literal[1e-06] = RIDGE_LAMBDA
    b_matrix: tuple[tuple[float, ...], ...]
    gram_eigenvalues: tuple[float, ...]
    eigenvalue_relative_tolerance: Literal[1e-10] = EIGENVALUE_RELATIVE_TOLERANCE
    clipped_negative_eigenvalue_count: int = Field(ge=0)
    residual_numerical_tolerance: Literal[1e-10] = RESIDUAL_NUMERICAL_TOLERANCE
    train_residual_p99: float = Field(ge=0.0)
    ood_policy: Literal["TRAIN-residual-empirical-p99-heuristic"] = (
        "TRAIN-residual-empirical-p99-heuristic"
    )
    score_semantics: Literal[
        "regularized-kernel-reference-reconstruction-residual;not-calibrated"
    ] = "regularized-kernel-reference-reconstruction-residual;not-calibrated"
    effective_source_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_fitted_shape(self) -> NystromAFSEArtifact:
        size = self.reference_size
        if self.output_dimension != size:
            raise ValueError("AFSE output dimension must equal the frozen reference size")
        sequences = (
            self.landmark_sample_ids,
            self.landmark_lineage_ids,
            self.landmark_classes,
            self.landmark_encoded,
            self.b_matrix,
            self.gram_eigenvalues,
        )
        if any(len(value) != size for value in sequences):
            raise ValueError("AFSE landmark payload does not match the reference size")
        if len(set(self.landmark_sample_ids)) != size:
            raise ValueError("AFSE landmark sample identifiers must be distinct")
        if len(set(self.landmark_lineage_ids)) != size:
            raise ValueError("AFSE landmarks must use distinct TRAIN lineages")
        if any(len(row) != FEATURE_COUNT for row in self.landmark_encoded):
            raise ValueError("AFSE encoded landmarks must each contain eight features")
        if any(len(row) != size for row in self.b_matrix):
            raise ValueError("AFSE B matrix must be square in the reference dimension")
        class_counts = tuple(Counter(self.landmark_classes).values())
        if not class_counts or max(class_counts) != min(class_counts):
            raise ValueError("AFSE landmarks must be exactly class-balanced")
        if self.clipped_negative_eigenvalue_count > size:
            raise ValueError("AFSE clipped-eigenvalue count exceeds the reference size")
        if not self.effective_source_hashes:
            raise ValueError("AFSE effective source provenance is required")
        if any(not _is_sha256(value) for value in self.effective_source_hashes.values()):
            raise ValueError("AFSE source provenance contains an invalid SHA-256")
        return self


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _effective_source_hashes() -> dict[str, str]:
    root = _backend_root()
    relative_paths = (
        "app/embeddings/nystrom.py",
        "app/quantum/adapter.py",
        "app/quantum/user_pipeline/tqk8.py",
    )
    return {relative: file_sha256(root / relative) for relative in relative_paths}


def _theta_identity(theta: NDArray[np.float64]) -> tuple[str, str]:
    digest = _digest({"schema_version": "aqse.theta.v1", "theta": array_identity(theta)})
    return f"aqse-theta-{digest[:16]}", digest


def _artifact_digest_payload(artifact: NystromAFSEArtifact) -> dict[str, Any]:
    return artifact.model_dump(mode="json", exclude={"artifact_id", "content_digest"})


def validate_afse_artifact(artifact: NystromAFSEArtifact) -> None:
    """Validate all deterministic identities before executing an artifact."""

    theta = np.asarray(artifact.theta, dtype=np.float64)
    theta_id, theta_digest = _theta_identity(theta)
    if artifact.theta_id != theta_id or artifact.theta_digest != theta_digest:
        raise ValueError("AFSE theta identity is invalid")
    expected_digest = _digest(_artifact_digest_payload(artifact))
    if artifact.content_digest != expected_digest:
        raise ValueError("AFSE scientific content digest is invalid")
    if artifact.artifact_id != f"aqse-afse-{expected_digest[:16]}":
        raise ValueError("AFSE artifact identity is invalid")
    matrix = np.asarray(artifact.b_matrix, dtype=np.float64)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("AFSE B matrix must be symmetric")


def query_context_for(artifact: NystromAFSEArtifact) -> AFSEQueryContext:
    return AFSEQueryContext(
        feature_profile_id=artifact.feature_profile_id,
        scaler_id=artifact.scaler_id,
        encoding_policy_id=artifact.encoding_policy_id,
        theta_id=artifact.theta_id,
    )


def _as_feature_matrix(values: NDArray[Any] | Sequence[Sequence[float]]) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT or len(matrix) == 0:
        raise ValueError("AFSE encoded features must have shape (N, 8)")
    if not np.isfinite(matrix).all():
        raise ValueError("AFSE encoded features contain NaN or infinity")
    return np.asarray(matrix, dtype=np.float64)


def _as_theta(values: NDArray[Any] | Sequence[float]) -> NDArray[np.float64]:
    theta = np.asarray(values, dtype=np.float64)
    if theta.shape != (PARAMETER_COUNT,):
        raise ValueError("AFSE theta must contain exactly 16 parameters")
    if not np.isfinite(theta).all():
        raise ValueError("AFSE theta contains NaN or infinity")
    return theta


def _state_matrix(
    engine: QuantumEngine,
    features: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.complex128]:
    states = np.stack([engine.state(row, theta) for row in features]).astype(
        np.complex128,
        copy=False,
    )
    if states.ndim != 2 or not np.isfinite(states).all():
        raise ValueError("quantum engine produced invalid landmark states")
    norms = np.sum(np.abs(states) ** 2, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("quantum engine states must be normalized")
    return states


def _fidelity_cross_gram(
    query_states: NDArray[np.complex128],
    landmark_states: NDArray[np.complex128],
) -> NDArray[np.float64]:
    gram = np.asarray(np.abs(query_states.conj() @ landmark_states.T) ** 2, dtype=np.float64)
    tolerance = 1.0e-10
    if not np.isfinite(gram).all():
        raise ValueError("quantum fidelity matrix contains NaN or infinity")
    if np.any(gram < -tolerance) or np.any(gram > 1.0 + tolerance):
        raise ValueError("quantum fidelity matrix is outside [0, 1]")
    return np.clip(gram, 0.0, 1.0)


def _regularized_basis(
    landmark_states: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    fidelity = _fidelity_cross_gram(landmark_states, landmark_states)
    symmetric = np.asarray((fidelity + fidelity.T) / 2.0, dtype=np.float64)
    if not np.allclose(np.diag(symmetric), 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("landmark fidelity Gram diagonal must equal one")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(np.max(eigenvalues))
    rejection_floor = -EIGENVALUE_RELATIVE_TOLERANCE * max(1.0, largest)
    if float(np.min(eigenvalues)) < rejection_floor:
        raise ValueError("landmark fidelity Gram has a materially negative eigenvalue")
    clipped_count = int(np.count_nonzero(eigenvalues < 0.0))
    stabilized = np.maximum(eigenvalues, 0.0)
    inverse_root = np.power(stabilized + RIDGE_LAMBDA, -0.5)
    basis = (eigenvectors * inverse_root) @ eigenvectors.T
    basis = np.asarray((basis + basis.T) / 2.0, dtype=np.float64)
    if not np.isfinite(basis).all():
        raise ValueError("regularized Nyström basis contains NaN or infinity")
    return basis, np.asarray(eigenvalues, dtype=np.float64), clipped_count


def _embedding_and_residuals(
    query_states: NDArray[np.complex128],
    landmark_states: NDArray[np.complex128],
    basis: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    kernel_rows = _fidelity_cross_gram(query_states, landmark_states)
    vectors = np.asarray(kernel_rows @ basis, dtype=np.float64)
    self_fidelity = np.asarray(
        np.abs(np.sum(query_states.conj() * query_states, axis=1)) ** 2,
        dtype=np.float64,
    )
    residuals = self_fidelity - np.sum(vectors * vectors, axis=1)
    if np.any(residuals < -RESIDUAL_NUMERICAL_TOLERANCE):
        raise ValueError("AFSE reconstruction residual is materially negative")
    residuals = np.where(residuals < 0.0, 0.0, residuals)
    if not np.isfinite(vectors).all() or not np.isfinite(residuals).all():
        raise ValueError("AFSE transform produced NaN or infinity")
    return vectors, np.asarray(residuals, dtype=np.float64)


def _select_landmark_indices(
    sample_ids: Sequence[str],
    lineage_ids: Sequence[str],
    classes: Sequence[str],
) -> NDArray[np.int64]:
    class_order = tuple(sorted(set(classes)))
    if len(class_order) < 2:
        raise ValueError("AFSE landmark selection requires at least two TRAIN classes")
    lineage_class: dict[str, str] = {}
    candidates: dict[str, list[int]] = {label: [] for label in class_order}
    for index, (sample_id, lineage_id, label) in enumerate(
        zip(sample_ids, lineage_ids, classes, strict=True)
    ):
        known = lineage_class.setdefault(lineage_id, label)
        if known != label:
            raise ValueError("one TRAIN lineage cannot carry multiple downstream classes")
        candidates[label].append(index)
        if not sample_id or not lineage_id or not label:
            raise ValueError("AFSE landmark identities and classes must be non-empty")

    unique_candidates: dict[str, list[int]] = {}
    for label in class_order:
        by_lineage: dict[str, list[int]] = {}
        for index in candidates[label]:
            by_lineage.setdefault(lineage_ids[index], []).append(index)
        representatives = [
            min(indices, key=lambda item: sample_ids[item])
            for _, indices in sorted(by_lineage.items())
        ]
        unique_candidates[label] = sorted(representatives, key=lambda item: sample_ids[item])

    per_class = min(
        MAX_LANDMARKS // len(class_order),
        *(len(unique_candidates[label]) for label in class_order),
    )
    if per_class == 0:
        raise ValueError("each downstream class needs at least one distinct TRAIN lineage")

    generator = np.random.default_rng(LANDMARK_SELECTION_SEED)
    selected_by_class: dict[str, list[int]] = {}
    for label in class_order:
        available = unique_candidates[label]
        order = generator.permutation(len(available))[:per_class]
        selected_by_class[label] = [available[int(position)] for position in order]

    selected = [
        selected_by_class[label][ordinal]
        for ordinal in range(per_class)
        for label in class_order
    ]
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True)
class NystromAFSE:
    """Immutable AFSE runtime with a fixed landmark-state cache."""

    artifact: NystromAFSEArtifact
    _engine: QuantumEngine = field(repr=False, compare=False)
    _landmark_states: NDArray[np.complex128] = field(repr=False, compare=False)
    _basis: NDArray[np.float64] = field(repr=False, compare=False)
    _state_lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    @classmethod
    def from_artifact(
        cls,
        artifact: NystromAFSEArtifact,
        *,
        backend_type: BackendType = "numpy",
    ) -> NystromAFSE:
        validate_afse_artifact(artifact)
        engine = TQK8Adapter(backend_type)
        theta = np.asarray(artifact.theta, dtype=np.float64)
        landmarks = np.asarray(artifact.landmark_encoded, dtype=np.float64)
        states = _state_matrix(engine, landmarks, theta)
        basis = np.asarray(artifact.b_matrix, dtype=np.float64)
        states.setflags(write=False)
        basis.setflags(write=False)
        return cls(
            artifact=artifact,
            _engine=engine,
            _landmark_states=states,
            _basis=basis,
        )

    def transform(
        self,
        encoded_features: NDArray[Any] | Sequence[Sequence[float]],
        *,
        sample_ids: Sequence[str],
        context: AFSEQueryContext,
    ) -> LocalEmbeddingBatch:
        validate_afse_artifact(self.artifact)
        if context != query_context_for(self.artifact):
            raise ValueError("query is incompatible with the frozen AFSE space")
        features = _as_feature_matrix(encoded_features)
        identifiers = tuple(sample_ids)
        if len(identifiers) != len(features) or len(set(identifiers)) != len(identifiers):
            raise ValueError("AFSE query sample identifiers must be complete and distinct")
        theta = np.asarray(self.artifact.theta, dtype=np.float64)
        with self._state_lock:
            query_states = _state_matrix(self._engine, features, theta)
        vectors, residuals = _embedding_and_residuals(
            query_states,
            self._landmark_states,
            self._basis,
        )
        return LocalEmbeddingBatch(
            method_version=self.artifact.method_id,
            sample_ids=identifiers,
            vectors=vectors.tolist(),
            reference_size=self.artifact.reference_size,
            reconstruction_residuals=tuple(float(value) for value in residuals),
            heuristic_ood=tuple(
                bool(value > self.artifact.train_residual_p99) for value in residuals
            ),
        )


def fit_nystrom_afse(
    encoded_train: NDArray[Any] | Sequence[Sequence[float]],
    *,
    sample_ids: Sequence[str],
    lineage_ids: Sequence[str],
    classes: Sequence[str],
    theta: NDArray[Any] | Sequence[float],
    feature_profile_id: str,
    scaler_id: str,
    encoding_policy_id: str,
    fitted_on_dataset_id: str,
    fitted_on_dataset_digest: str,
    backend_type: BackendType = "numpy",
    effective_source_hashes: Mapping[str, str] | None = None,
) -> NystromAFSE:
    """Fit the approved fixed-landmark AFSE on TRAIN observations only."""

    features = _as_feature_matrix(encoded_train)
    theta_array = _as_theta(theta)
    identifiers = tuple(sample_ids)
    lineages = tuple(lineage_ids)
    labels = tuple(classes)
    if not (len(features) == len(identifiers) == len(lineages) == len(labels)):
        raise ValueError("AFSE TRAIN features and row metadata must have equal lengths")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("AFSE TRAIN sample identifiers must be distinct")
    if not _is_sha256(fitted_on_dataset_digest):
        raise ValueError("AFSE fitted dataset digest must be a SHA-256")

    selected = _select_landmark_indices(identifiers, lineages, labels)
    landmark_features = np.asarray(features[selected], dtype=np.float64)
    engine = TQK8Adapter(backend_type)
    landmark_states = _state_matrix(engine, landmark_features, theta_array)
    basis, eigenvalues, clipped_count = _regularized_basis(landmark_states)

    state_by_sample = {
        identifiers[int(index)]: landmark_states[position]
        for position, index in enumerate(selected)
    }
    train_states = np.stack(
        [
            state_by_sample.get(identifier)
            if identifier in state_by_sample
            else engine.state(features[index], theta_array)
            for index, identifier in enumerate(identifiers)
        ]
    ).astype(np.complex128, copy=False)
    _, train_residuals = _embedding_and_residuals(train_states, landmark_states, basis)
    residual_p99 = float(np.quantile(train_residuals, 0.99, method="linear"))
    theta_id, theta_digest = _theta_identity(theta_array)
    source_hashes = dict(effective_source_hashes or _effective_source_hashes())

    scientific = {
        "schema_version": AFSE_SCHEMA_VERSION,
        "method_id": AFSE_METHOD_ID,
        "fitted_on_partition": "TRAIN",
        "fitted_on_dataset_id": fitted_on_dataset_id,
        "fitted_on_dataset_digest": fitted_on_dataset_digest,
        "feature_profile_id": feature_profile_id,
        "scaler_id": scaler_id,
        "encoding_policy_id": encoding_policy_id,
        "feature_count": FEATURE_COUNT,
        "theta_id": theta_id,
        "theta_digest": theta_digest,
        "theta": theta_array.tolist(),
        "fit_backend_type": backend_type,
        "selection_seed": LANDMARK_SELECTION_SEED,
        "selection_policy": (
            "balanced-classes;one-distinct-TRAIN-lineage-per-landmark;seeded-v1"
        ),
        "requested_reference_size": MAX_LANDMARKS,
        "reference_size": len(selected),
        "output_dimension": len(selected),
        "landmark_sample_ids": [identifiers[int(index)] for index in selected],
        "landmark_lineage_ids": [lineages[int(index)] for index in selected],
        "landmark_classes": [labels[int(index)] for index in selected],
        "landmark_encoded": landmark_features.tolist(),
        "ridge_lambda": RIDGE_LAMBDA,
        "b_matrix": basis.tolist(),
        "gram_eigenvalues": eigenvalues.tolist(),
        "eigenvalue_relative_tolerance": EIGENVALUE_RELATIVE_TOLERANCE,
        "clipped_negative_eigenvalue_count": clipped_count,
        "residual_numerical_tolerance": RESIDUAL_NUMERICAL_TOLERANCE,
        "train_residual_p99": residual_p99,
        "ood_policy": "TRAIN-residual-empirical-p99-heuristic",
        "score_semantics": (
            "regularized-kernel-reference-reconstruction-residual;not-calibrated"
        ),
        "effective_source_hashes": source_hashes,
    }
    content_digest = _digest(scientific)
    artifact = NystromAFSEArtifact.model_validate(
        {
            **scientific,
            "artifact_id": f"aqse-afse-{content_digest[:16]}",
            "content_digest": content_digest,
        }
    )
    validate_afse_artifact(artifact)
    landmark_states.setflags(write=False)
    basis.setflags(write=False)
    return NystromAFSE(
        artifact=artifact,
        _engine=engine,
        _landmark_states=landmark_states,
        _basis=basis,
    )
