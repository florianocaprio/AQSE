from fastapi import APIRouter, HTTPException, status

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
def start_training_job() -> TrainingJobView:
    """Admit one frozen-contract 1D.3 QNG job; no caller hyperparameters."""

    try:
        return training_jobs.start()
    except TrainingJobBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except TrainingJobCapacityError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
