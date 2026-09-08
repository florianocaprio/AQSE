import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.quantum.adapter import run_quantum_infrastructure_smoke_test

logger = logging.getLogger(__name__)
router = APIRouter()


class QuantumHealthResponse(BaseModel):
    status: str
    engine: str
    qiskit: str
    numpy_reference: str
    qubits: int
    features: int
    trainable_parameters: int
    simulation: str


@router.get("/quantum/health", response_model=QuantumHealthResponse)
def quantum_health() -> QuantumHealthResponse:
    """Run a deterministic local statevector infrastructure smoke test."""

    try:
        result = run_quantum_infrastructure_smoke_test()
    except Exception as exc:
        logger.exception("Quantum infrastructure smoke test failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Quantum infrastructure smoke test failed",
        ) from exc

    return QuantumHealthResponse(status="ok", **result.__dict__)
