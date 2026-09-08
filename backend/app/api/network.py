from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Annotated, Callable

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from app.network.defaults import (
    default_network_configuration,
    default_vector_configuration,
    network_preset_configuration,
    network_presets,
    vector_scenarios,
)
from app.network.models import (
    FieldProviderCatalog,
    FiniteGeneratorTruthSeries,
    FiniteMeasurementSeries,
    FiniteVectorSimulationConfiguration,
    FiniteVectorSimulationResponse,
    FrameBatch,
    NetworkEventConfiguration,
    NetworkHealth,
    NetworkPresetCatalog,
    NetworkSessionConfiguration,
    NodeCount,
    ObservabilityDiagnostics,
    ObservabilityRequest,
    ObservationSnapshot,
    ScheduledEventResponse,
    SessionView,
    StepRequest,
    TruthFrameBatch,
    TruthSnapshot,
    VectorScenario,
    VectorScenarioCatalog,
)
from app.network.observability import calculate_observability
from app.network.providers import field_provider_catalog
from app.network.session import (
    NetworkSession,
    SessionCapacityError,
    SessionConflictError,
    network_sessions,
)
from app.network.simulation import NetworkSimulator

router = APIRouter()


def _session(session_id: str) -> NetworkSession:
    session = network_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="sensor-network session not found")
    return session


def _control(operation: Callable[[], SessionView]) -> SessionView:
    try:
        return operation()
    except SessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/network/health", response_model=NetworkHealth)
def network_health() -> NetworkHealth:
    return NetworkHealth(
        active_sessions=network_sessions.count(),
        maximum_sessions=network_sessions.maximum_sessions,
    )


@router.get("/network/defaults", response_model=NetworkSessionConfiguration)
def network_defaults(
    node_count: Annotated[NodeCount, Query()] = 4,
) -> NetworkSessionConfiguration:
    return default_network_configuration(node_count)


@router.get("/network/presets", response_model=NetworkPresetCatalog)
def list_network_presets() -> NetworkPresetCatalog:
    return network_presets()


@router.get("/network/field-providers", response_model=FieldProviderCatalog)
def list_field_providers() -> FieldProviderCatalog:
    return field_provider_catalog()


@router.get(
    "/network/presets/{preset_id}",
    response_model=NetworkSessionConfiguration,
)
def network_preset(preset_id: str) -> NetworkSessionConfiguration:
    try:
        return network_preset_configuration(preset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/network/sessions",
    response_model=SessionView,
    status_code=status.HTTP_201_CREATED,
)
def create_network_session(
    configuration: NetworkSessionConfiguration,
) -> SessionView:
    try:
        return network_sessions.create(configuration).view()
    except SessionCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@router.get("/network/sessions/{session_id}", response_model=SessionView)
def get_network_session(session_id: str) -> SessionView:
    return _session(session_id).view()


@router.post("/network/sessions/{session_id}/start", response_model=SessionView)
def start_network_session(session_id: str) -> SessionView:
    return _control(_session(session_id).start)


@router.post("/network/sessions/{session_id}/pause", response_model=SessionView)
def pause_network_session(session_id: str) -> SessionView:
    return _control(_session(session_id).pause)


@router.post("/network/sessions/{session_id}/resume", response_model=SessionView)
def resume_network_session(session_id: str) -> SessionView:
    return _control(_session(session_id).resume)


@router.post("/network/sessions/{session_id}/stop", response_model=SessionView)
def stop_network_session(session_id: str) -> SessionView:
    return _control(_session(session_id).stop)


@router.post("/network/sessions/{session_id}/reset", response_model=SessionView)
def reset_network_session(session_id: str) -> SessionView:
    return _control(_session(session_id).reset)


@router.post("/network/sessions/{session_id}/replay", response_model=SessionView)
def replay_network_session(session_id: str) -> SessionView:
    return _control(_session(session_id).replay)


@router.post("/network/sessions/{session_id}/step", response_model=FrameBatch)
def step_network_session(session_id: str, request: StepRequest) -> FrameBatch:
    try:
        return _session(session_id).step(request.frames)
    except SessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/network/sessions/{session_id}/snapshot",
    response_model=ObservationSnapshot,
)
def network_snapshot(session_id: str) -> ObservationSnapshot:
    return _session(session_id).observation_snapshot()


@router.get(
    "/network/sessions/{session_id}/truth/snapshot",
    response_model=TruthSnapshot,
)
def network_truth_snapshot(session_id: str) -> TruthSnapshot:
    return _session(session_id).truth_snapshot()


@router.get("/network/sessions/{session_id}/frames", response_model=FrameBatch)
def network_frames(
    session_id: str,
    after_frame_id: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1_000)] = 250,
) -> FrameBatch:
    return _session(session_id).observation_batch(after_frame_id, limit)


@router.get(
    "/network/sessions/{session_id}/truth/frames",
    response_model=TruthFrameBatch,
)
def network_truth_frames(
    session_id: str,
    after_frame_id: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1_000)] = 250,
) -> TruthFrameBatch:
    return _session(session_id).truth_batch(after_frame_id, limit)


@router.post(
    "/network/sessions/{session_id}/events",
    response_model=ScheduledEventResponse,
    status_code=status.HTTP_201_CREATED,
)
def schedule_network_event(
    session_id: str,
    event: NetworkEventConfiguration,
) -> ScheduledEventResponse:
    try:
        return _session(session_id).schedule_event(event)
    except SessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@router.delete(
    "/network/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_network_session(session_id: str) -> Response:
    if not network_sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="sensor-network session not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/network/sessions/{session_id}/stream")
async def stream_network_observations(
    request: Request,
    session_id: str,
    after_frame_id: Annotated[int | None, Query(ge=0)] = None,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    session = _session(session_id)
    # Browsers retain the original EventSource URL when reconnecting and send
    # the most recently processed event in Last-Event-ID.  The header must win
    # over the bootstrap query cursor or an automatic reconnect would replay
    # the original backlog and could report a false bounded-buffer gap.
    cursor = after_frame_id if after_frame_id is not None else 0
    if last_event_id is not None:
        try:
            cursor = max(0, int(last_event_id))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Last-Event-ID must be an integer") from exc

    async def events():  # type: ignore[no-untyped-def]
        nonlocal cursor
        while not await request.is_disconnected():
            batch = await asyncio.to_thread(
                session.wait_for_observations,
                cursor,
                5.0,
                250,
            )
            if batch.frames:
                cursor = batch.to_frame_id or cursor
                payload = json.dumps(
                    batch.model_dump(mode="json"),
                    allow_nan=False,
                    separators=(",", ":"),
                )
                yield f"id: {cursor}\nevent: observations\ndata: {payload}\n\n"
            elif batch.status.state.value == "stopped":
                payload = json.dumps(batch.status.model_dump(mode="json"), allow_nan=False)
                yield f"event: session_status\ndata: {payload}\n\n"
                return
            else:
                yield ": keepalive\n\n"
                await asyncio.sleep(max(1.0, session.stream_refresh_interval_s))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/network/observability", response_model=ObservabilityDiagnostics)
def network_observability(request: ObservabilityRequest) -> ObservabilityDiagnostics:
    try:
        return calculate_observability(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/sensors/vector-magnetometer/defaults",
    response_model=FiniteVectorSimulationConfiguration,
)
def vector_magnetometer_defaults(
    scenario: VectorScenario = VectorScenario.STATIC_REFERENCE,
) -> FiniteVectorSimulationConfiguration:
    return default_vector_configuration(scenario)


@router.get(
    "/sensors/vector-magnetometer/scenarios",
    response_model=VectorScenarioCatalog,
)
def vector_magnetometer_scenarios() -> VectorScenarioCatalog:
    return vector_scenarios()


@router.post(
    "/sensors/vector-magnetometer/simulate",
    response_model=FiniteVectorSimulationResponse,
)
def simulate_vector_magnetometer(
    configuration: FiniteVectorSimulationConfiguration,
) -> FiniteVectorSimulationResponse:
    network_configuration = NetworkSessionConfiguration(
        session_name="finite-vector-magnetometer",
        random_seed=configuration.random_seed,
        sampling_rate_Hz=configuration.sampling_rate_Hz,
        ui_refresh_rate_Hz=min(5.0, configuration.sampling_rate_Hz),
        buffer_duration_s=configuration.duration_s,
        environment=configuration.environment,
        nodes=(configuration.node,),
        events=configuration.events,
    )
    simulator = NetworkSimulator(network_configuration)
    epoch = datetime.now(timezone.utc)
    generated = tuple(
        simulator.generate(
            session_id="finite-vector-simulation",
            frame_id=index + 1,
            sim_time_s=index / configuration.sampling_rate_Hz,
            epoch_utc=epoch,
            events=configuration.events,
        )
        for index in range(configuration.sample_count)
    )
    readings = tuple(frame.observation.readings[0] for frame in generated)
    fields = tuple(frame.truth.fields[0] for frame in generated)
    return FiniteVectorSimulationResponse(
        configuration=configuration,
        sample_count=configuration.sample_count,
        measurement=FiniteMeasurementSeries(
            sensor_id=configuration.node.sensor_id,
            time_s=tuple(frame.observation.sim_time_s for frame in generated),
            position_m=tuple(reading.position_m for reading in readings),
            orientation_world_to_sensor_wxyz=tuple(
                reading.orientation_world_to_sensor_wxyz for reading in readings
            ),
            temperature_K=tuple(reading.observed_temperature_K for reading in readings),
            measured_field_T=tuple(reading.components_T for reading in readings),
            saturation_mask=tuple(
                reading.saturation_mask or (False, False, False) for reading in readings
            ),
            valid=tuple(reading.valid for reading in readings),
            quality_flags=tuple(reading.quality_flags for reading in readings),
        ),
        generator_truth=FiniteGeneratorTruthSeries(
            time_s=tuple(frame.truth.sim_time_s for frame in generated),
            earth_field_world_T=tuple(field.uniform_field_world_T for field in fields),
            environment_field_world_T=tuple(field.field_true_world_T for field in fields),
            anomaly_field_world_T=tuple(field.anomaly_field_world_T for field in fields),
            periodic_field_world_T=tuple(field.periodic_field_world_T for field in fields),
            ideal_field_sensor_T=tuple(field.ideal_field_sensor_T for field in fields),
            active_causes=tuple(frame.truth.active_causes for frame in generated),
        ),
    )
