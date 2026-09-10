from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from app.demo.bundle_models import ActiveBundlePointer
from app.demo.bundle_storage import (
    apply_selection_freeze,
    load_selection_freeze_by_id,
)
from app.demo.knowledge_models import (
    KnowledgeCaptureRequest,
    KnowledgeCollectionWriteResponse,
    KnowledgeLabelWriteResponse,
    KnowledgeObservationWriteResponse,
    KnowledgeRegistryView,
    KnowledgeReviewedLabelRequest,
    KnowledgeTask,
    KnowledgeTrainApprovalRequest,
)
from app.demo.knowledge_storage import (
    KnowledgeStoreError,
    approve_train_collection,
    capture_observation_episode,
    create_reviewed_label,
    knowledge_registry_view,
    load_observation_episode,
    write_observation_episode,
    write_reviewed_label,
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
from app.features.state8_models import (
    STATE8_WINDOW_SAMPLES,
    State8FeatureQuality,
    State8FeatureRecord,
)

router = APIRouter()


@router.get("/demo/registry", response_model=DemoRegistryView)
def get_demo_registry() -> DemoRegistryView:
    try:
        return demo_registry_view()
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/demo/knowledge", response_model=KnowledgeRegistryView)
def get_demo_knowledge() -> KnowledgeRegistryView:
    try:
        return knowledge_registry_view()
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/demo/knowledge/observations",
    response_model=KnowledgeObservationWriteResponse,
    status_code=status.HTTP_201_CREATED,
)
def capture_demo_observation(
    request: KnowledgeCaptureRequest,
    response: Response,
) -> KnowledgeObservationWriteResponse:
    view = demo_analysis.get(request.session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="analysis worker not found")
    result = next(
        (
            item
            for item in view.latest_results
            if item.sensor_id == request.sensor_id
        ),
        None,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="current sensor result not found")
    expected_task = (
        KnowledgeTask.LOCAL
        if result.task_id == "aqse.local-change.v1"
        else KnowledgeTask.NETWORK
    )
    if request.task is not expected_task:
        raise HTTPException(
            status_code=409,
            detail="requested knowledge task differs from the current feature profile",
        )
    if not result.feature_valid or len(result.source_frame_ids) != STATE8_WINDOW_SAMPLES:
        raise HTTPException(
            status_code=409,
            detail="only complete eligible State8 windows can be captured",
        )
    feature = State8FeatureRecord(
        window_id=result.window_id,
        session_id=result.session_id,
        sensor_id=result.sensor_id,
        profile_id=result.profile_id,
        profile_fingerprint=result.profile_fingerprint,
        reference_id=result.reference_id,
        start_time_s=result.window_start_s,
        end_exclusive_time_s=result.window_end_exclusive_s,
        source_frame_ids=result.source_frame_ids,
        peer_sensor_ids=result.peer_sensor_ids,
        values=result.feature_values,
        quality=State8FeatureQuality(
            valid_for_quantum=True,
            flags=(),
            per_feature_valid=(True, True, True, True, True, True, True, True),
            received_sample_count=STATE8_WINDOW_SAMPLES,
            usable_sample_count=STATE8_WINDOW_SAMPLES,
        ),
    )
    try:
        observation = capture_observation_episode(
            feature,
            task=request.task,
            node_count=result.node_count,
        )
        _, reused = write_observation_episode(observation)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if reused:
        response.status_code = status.HTTP_200_OK
    return KnowledgeObservationWriteResponse(
        observation=observation,
        reused=reused,
    )


@router.post(
    "/demo/knowledge/labels",
    response_model=KnowledgeLabelWriteResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_demo_reviewed_label(
    request: KnowledgeReviewedLabelRequest,
    response: Response,
) -> KnowledgeLabelWriteResponse:
    try:
        observation = load_observation_episode(request.observation_id)
        if observation.task is not request.task:
            raise ValueError("reviewed-label task differs from its observation")
        label = create_reviewed_label(
            observation,
            label=request.label,
            reviewer_id=request.reviewer_id,
            reviewed_at_utc=request.reviewed_at_utc,
        )
        _, reused = write_reviewed_label(label)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if reused:
        response.status_code = status.HTTP_200_OK
    return KnowledgeLabelWriteResponse(label=label, reused=reused)


@router.post(
    "/demo/knowledge/train-collections",
    response_model=KnowledgeCollectionWriteResponse,
    status_code=status.HTTP_201_CREATED,
)
def approve_demo_train_collection(
    request: KnowledgeTrainApprovalRequest,
    response: Response,
) -> KnowledgeCollectionWriteResponse:
    try:
        _, collection, reused = approve_train_collection(request)
    except (KnowledgeStoreError, OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if reused:
        response.status_code = status.HTTP_200_OK
    return KnowledgeCollectionWriteResponse(
        collection=collection,
        reused=reused,
    )


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
