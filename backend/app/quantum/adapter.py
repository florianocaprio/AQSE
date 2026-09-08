from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from app.quantum.engine import QuantumEngine
from app.quantum.user_pipeline.tqk8 import (
    EDGES,
    N_PARAMS,
    N_QUBITS,
    StateEngine,
    build_vqc,
)

BackendType = Literal["qiskit", "numpy"]


@dataclass(frozen=True)
class QuantumEngineMetadata:
    engine_name: str
    number_of_qubits: int
    number_of_features: int
    number_of_parameters: int
    backend_type: str


@dataclass(frozen=True)
class QuantumInfrastructureStatus:
    engine: str
    qiskit: str
    numpy_reference: str
    qubits: int
    features: int
    trainable_parameters: int
    simulation: str


class TQK8Adapter(QuantumEngine):
    """Thin application adapter around Floriano's unmodified TQK8 engine."""

    def __init__(self, backend_type: BackendType) -> None:
        self._backend_type = backend_type
        self._engine = StateEngine(backend_type)

    @property
    def metadata(self) -> QuantumEngineMetadata:
        return quantum_engine_metadata(self._backend_type)

    def state(
        self,
        features: Sequence[float],
        parameters: Sequence[float],
    ) -> NDArray[np.complex128]:
        return np.asarray(self._engine.state(features, parameters), dtype=np.complex128)

    def gram(
        self,
        features: NDArray[np.float64],
        parameters: Sequence[float],
    ) -> NDArray[np.float64]:
        return np.asarray(self._engine.gram(features, parameters), dtype=np.float64)

    def cross_gram(
        self,
        query_features: NDArray[np.float64],
        reference_features: NDArray[np.float64],
        parameters: Sequence[float],
    ) -> NDArray[np.float64]:
        """Return query-to-reference fidelities using the existing TQK8 engine."""

        return np.asarray(
            self._engine.gram(query_features, parameters, reference_features),
            dtype=np.float64,
        )


def quantum_engine_metadata(backend_type: BackendType) -> QuantumEngineMetadata:
    """Return adapter metadata without constructing an exact-state engine."""

    simulation = (
        "qiskit_statevector" if backend_type == "qiskit" else "numpy_statevector"
    )
    return QuantumEngineMetadata(
        engine_name="AQSE TQK8",
        number_of_qubits=N_QUBITS,
        number_of_features=N_QUBITS,
        number_of_parameters=N_PARAMS,
        backend_type=simulation,
    )


def run_quantum_infrastructure_smoke_test() -> QuantumInfrastructureStatus:
    """Validate local exact-state infrastructure; this is not AQSE training logic."""

    circuit, _, trainable_parameters = build_vqc()
    if circuit.num_qubits != N_QUBITS:
        raise RuntimeError("Unexpected TQK8 qubit count")
    if len(trainable_parameters) != N_PARAMS:
        raise RuntimeError("Unexpected TQK8 trainable parameter count")
    if circuit.count_ops().get("cz", 0) != len(EDGES):
        raise RuntimeError("Unexpected TQK8 CZ gate count")

    sample = np.linspace(-0.35, 0.35, N_QUBITS, dtype=float)
    parameters = np.linspace(-0.7, 0.7, N_PARAMS, dtype=float)
    numpy_engine = TQK8Adapter("numpy")
    qiskit_engine = TQK8Adapter("qiskit")
    numpy_state = numpy_engine.state(sample, parameters)
    qiskit_state = qiskit_engine.state(sample, parameters)

    np.testing.assert_allclose(qiskit_state, numpy_state, atol=1e-12, rtol=1e-12)

    metadata = qiskit_engine.metadata
    return QuantumInfrastructureStatus(
        engine=metadata.engine_name,
        qiskit="ready",
        numpy_reference="ready",
        qubits=metadata.number_of_qubits,
        features=metadata.number_of_features,
        trainable_parameters=metadata.number_of_parameters,
        simulation="statevector",
    )
