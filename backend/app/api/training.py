from fastapi import APIRouter, HTTPException, Response, status

from app.training.benchmark import (
    EngineeringBenchmarkBusyError,
    EngineeringBenchmarkView,
    qng_responsiveness_benchmarks,
)
from app.training.execution_intent import (
    EngineeringBenchmarkIntent,
    TrainingExecutionIntent,
)
from app.training.intent_storage import ExecutionIntentConflictError
from app.training.run_models import TrainingJobView
from app.training.service import (
    TrainingJobBusyError,
    TrainingJobCapacityError,
    training_jobs,
)

router = APIRouter()


@router.post(
    "/training/jobs",
    response_model=TrainingJobView,
    status_code=status.HTTP_201_CREATED,
)
def start_training_job(
    intent: TrainingExecutionIntent,
    response: Response,
) -> TrainingJobView:
    """Admit one frozen-contract 1D.3 QNG job; no caller hyperparameters."""

    try:
        result = training_jobs.start(intent)
        response.status_code = (
            status.HTTP_201_CREATED if result.created else status.HTTP_200_OK
        )
        return result.job
    except TrainingJobBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except TrainingJobCapacityError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ExecutionIntentConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/training/jobs/{job_id}", response_model=TrainingJobView)
def get_training_job(job_id: str) -> TrainingJobView:
    job = training_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="AQSE QNG training job not found")
    return job


@router.post("/training/jobs/{job_id}/cancel", response_model=TrainingJobView)
def cancel_training_job(job_id: str) -> TrainingJobView:
    job = training_jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="AQSE QNG training job not found")
    return job


@router.post(
    "/training/benchmarks/responsiveness",
    response_model=EngineeringBenchmarkView,
    status_code=status.HTTP_201_CREATED,
)
def start_responsiveness_benchmark(
    intent: EngineeringBenchmarkIntent,
    response: Response,
) -> EngineeringBenchmarkView:
    """Run the one predeclared engineering load probe without publishing a model."""

    try:
        result = qng_responsiveness_benchmarks.start(intent)
    except EngineeringBenchmarkBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    response.status_code = (
        status.HTTP_201_CREATED if result.created else status.HTTP_200_OK
    )
    return result.benchmark


@router.get(
    "/training/benchmarks/responsiveness",
    response_model=EngineeringBenchmarkView,
)
def get_responsiveness_benchmark() -> EngineeringBenchmarkView:
    benchmark = qng_responsiveness_benchmarks.get()
    if benchmark is None:
        raise HTTPException(status_code=404, detail="responsiveness benchmark not started")
    return benchmark
