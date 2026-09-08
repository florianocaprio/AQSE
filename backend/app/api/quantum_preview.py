from threading import BoundedSemaphore

from fastapi import APIRouter, HTTPException

from app.quantum.preview import (
    CircuitDescription,
    QuantumPreviewRequest,
    QuantumPreviewResponse,
    describe_tqk8_circuit,
    run_quantum_preview,
)

router = APIRouter()
preview_slot = BoundedSemaphore(value=1)


@router.get("/quantum/circuit", response_model=CircuitDescription)
def quantum_circuit_description() -> CircuitDescription:
    return describe_tqk8_circuit()


@router.post("/quantum/preview", response_model=QuantumPreviewResponse)
def quantum_preview(request: QuantumPreviewRequest) -> QuantumPreviewResponse:
    if not preview_slot.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="A quantum preview is already running; retry after it completes",
        )
    try:
        return run_quantum_preview(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        preview_slot.release()
