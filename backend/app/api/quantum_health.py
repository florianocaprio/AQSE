import logging
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.quantum.adapter import (
    quantum_engine_metadata,
    run_quantum_infrastructure_smoke_test,
)
from app.quantum.admission import heavy_quantum_slot
from app.quantum.user_pipeline.tqk8 import EDGES

logger = logging.getLogger(__name__)
router = APIRouter()
diagnostics_slot = heavy_quantum_slot


class QuantumHealthResponse(BaseModel):
    status: str
    engine: str
    adapter: str
    qiskit: str
    qiskit_version: str
    numpy_reference: str
    qubits: int
    features: int
    trainable_parameters: int
    simulation: str
    circuit_metadata: str
    entangling_edges: int


class QuantumDiagnosticsResponse(BaseModel):
    status: str
    engine: str
    qiskit: str
    numpy_reference: str
    qubits: int
    features: int
    trainable_parameters: int
    simulation: str
    state_comparison: str
    execution_duration_ms: float


@router.get("/quantum/health", response_model=QuantumHealthResponse)
def quantum_health() -> QuantumHealthResponse:
    """Return lightweight adapter readiness without executing a statevector."""

    try:
        qiskit_version = version("qiskit")
        metadata = quantum_engine_metadata("qiskit")
    except (PackageNotFoundError, ImportError) as exc:
        logger.exception("Quantum adapter readiness check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Quantum adapter is unavailable",
        ) from exc

    return QuantumHealthResponse(
        status="ok",
        engine=metadata.engine_name,
        adapter="ready",
        qiskit="installed",
        qiskit_version=qiskit_version,
        numpy_reference="available_not_executed",
        qubits=metadata.number_of_qubits,
        features=metadata.number_of_features,
        trainable_parameters=metadata.number_of_parameters,
        simulation=metadata.backend_type,
        circuit_metadata="available",
        entangling_edges=len(EDGES),
    )


@router.post("/quantum/diagnostics", response_model=QuantumDiagnosticsResponse)
def quantum_diagnostics() -> QuantumDiagnosticsResponse:
    """Run the explicit Qiskit/NumPy statevector infrastructure diagnostic."""

    if not diagnostics_slot.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Quantum diagnostics are already running; retry after completion",
        )
    started = perf_counter()
    try:
        result = run_quantum_infrastructure_smoke_test()
        return QuantumDiagnosticsResponse(
            status="ok",
            **result.__dict__,
            state_comparison="passed",
            execution_duration_ms=(perf_counter() - started) * 1_000.0,
        )
    except Exception as exc:
        logger.exception("Quantum infrastructure diagnostic failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Quantum infrastructure diagnostic failed",
        ) from exc
    finally:
        diagnostics_slot.release()
