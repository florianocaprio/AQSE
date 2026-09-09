from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from app.demo.bundle_models import ActiveBundlePointer
from app.demo.bundle_storage import (
    apply_selection_freeze,
    load_selection_freeze_by_id,
)
from app.demo.provisioning import demo_registry_view
from app.demo.runtime import AnalysisCapacityError, demo_analysis
from app.demo.runtime_models import (
    DemoAnalysisStart,
    DemoAnalysisView,
    DemoBundleApplicationRequest,
    DemoRegistryView,
    DemoTrainingIntent,
    DemoTrainingJobView,
)
from app.demo.training_service import (
    DemoTrainingBusyError,
    DemoTrainingStoreError,
    demo_training_jobs,
)

router = APIRouter()


@router.get("/demo/registry", response_model=DemoRegistryView)
def get_demo_registry() -> DemoRegistryView:
    try:
        return demo_registry_view()
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/demo/bundles/apply",
    response_model=ActiveBundlePointer,
)
def apply_demo_bundle_pair(
    request: DemoBundleApplicationRequest,
) -> ActiveBundlePointer:
    try:
        freeze = load_selection_freeze_by_id(request.selection_freeze_id)
        selected = {item.task_id: item.selected_bundle_id for item in freeze.selections}
        if (
            selected.get("aqse.local-change.v1") != request.local_bundle_id
            or selected.get("aqse.network-pattern.v1") != request.network_bundle_id
        ):
            raise ValueError(
                "requested bundle pair differs from its immutable selection freeze"
            )
        pointer = apply_selection_freeze(freeze)
        demo_analysis.invalidate_bundle_cache()
        return pointer
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="selection freeze not found") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/demo/analysis/{session_id}/start",
    response_model=DemoAnalysisView,
)
def start_demo_analysis(
    session_id: str,
    _request: DemoAnalysisStart,
) -> DemoAnalysisView:
    try:
        return demo_analysis.start(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AnalysisCapacityError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/demo/analysis/{session_id}",
    response_model=DemoAnalysisView,
)
def get_demo_analysis(session_id: str) -> DemoAnalysisView:
    view = demo_analysis.get(session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="analysis worker not found")
    return view


@router.post(
    "/demo/analysis/{session_id}/stop",
    response_model=DemoAnalysisView,
)
def stop_demo_analysis(session_id: str) -> DemoAnalysisView:
    view = demo_analysis.stop(session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="analysis worker not found")
    return view


@router.post(
    "/demo/training/jobs",
    response_model=DemoTrainingJobView,
    status_code=status.HTTP_201_CREATED,
)
def start_demo_training(
    intent: DemoTrainingIntent,
    response: Response,
) -> DemoTrainingJobView:
    try:
        job, created = demo_training_jobs.start(intent)
        response.status_code = (
            status.HTTP_201_CREATED if created else status.HTTP_200_OK
        )
        return job
    except DemoTrainingBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except DemoTrainingStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/demo/training/jobs/{job_id}",
    response_model=DemoTrainingJobView,
)
def get_demo_training(job_id: str) -> DemoTrainingJobView:
    try:
        job = demo_training_jobs.get(job_id)
    except (DemoTrainingStoreError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="demo training job not found")
    return job

@router.post(
    "/demo/training/jobs/{job_id}/cancel",
    response_model=DemoTrainingJobView,
)
def cancel_demo_training(job_id: str) -> DemoTrainingJobView:
    try:
        job = demo_training_jobs.cancel(job_id)
    except (DemoTrainingStoreError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="demo training job not found")
    return job
