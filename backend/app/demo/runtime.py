from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, RLock, Thread
from time import perf_counter, sleep
from typing import Callable, Literal

import numpy as np

from app.demo.bundle_models import (
    BundlePredictionBatch,
    BundleQueryContext,
    BundleRuntime,
)
from app.demo.bundle_storage import AppliedBundleSet, load_active_bundle_set
from app.demo.runtime_models import (
    ContextMode,
    DemoAnalysisResult,
    DemoAnalysisView,
)
from app.features.state8 import (
    build_state8_reference,
    extract_latest_state8_window,
    state8_profile,
)
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
    STATE8_REFERENCE_SAMPLES,
    STATE8_WINDOW_SAMPLES,
    State8FeatureRecord,
    State8ObservedReference,
)
from app.network.models import ObservationFrame
from app.network.session import NetworkSession, network_sessions
from app.quantum.admission import heavy_quantum_slot
from app.training.canonical import canonical_json_bytes

MAX_ANALYSIS_WORKERS = 16
MAX_BUFFERED_OBSERVATION_FRAMES = 1_600
MAX_LATENCY_SAMPLES = 512
WORKER_POLL_INTERVAL_S = 0.05


class AnalysisCapacityError(RuntimeError):
    pass


@dataclass
class _RuntimeCache:
    loader: Callable[[], AppliedBundleSet | None] = load_active_bundle_set
    _loaded: AppliedBundleSet | None = None
    _lock: RLock = field(default_factory=RLock)

    def get(self) -> AppliedBundleSet | None:
        with self._lock:
            if self._loaded is None:
                self._loaded = self.loader()
            return self._loaded

    def invalidate(self) -> None:
        with self._lock:
            self._loaded = None


bundle_runtime_cache = _RuntimeCache()


@dataclass
class _WorkerRecord:
    session_id: str
    worker_epoch: int
    state: Literal[
        "awaiting_reference",
        "running",
        "paused_for_training",
        "stopped",
        "failed",
        "bundle_unavailable",
    ]
    state_detail: str
    cancellation: Event
    application_epoch: int = 1
    reference_origin_frame_id: int = 1
    lock: RLock = field(default_factory=RLock)
    thread: Thread | None = None
    frames: deque[ObservationFrame] = field(
        default_factory=lambda: deque(maxlen=MAX_BUFFERED_OBSERVATION_FRAMES)
    )
    reference: State8ObservedReference | None = None
    cursor_frame_id: int = 0
    latest_observation_time_s: float = 0.0
    latest_analyzed_window_start_s: float | None = None
    completed_window_count: int = 0
    skipped_window_count: int = 0
    quality_abstention_count: int = 0
    latest_results: tuple[DemoAnalysisResult, ...] = ()
    latency_samples_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=MAX_LATENCY_SAMPLES)
    )
    application_id: str | None = None
    active_local_bundle_id: str | None = None
    active_network_bundle_id: str | None = None
    error: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _result_id(
    *,
    session_id: str,
    application_id: str,
    window_id: str,
) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": "aqse.demo-analysis-result-identity.v1",
                "session_id": session_id,
                "application_id": application_id,
                "window_id": window_id,
            }
        )
    ).hexdigest()
    return f"aqse-demo-result-{digest[:16]}"


class DemoAnalysisService:
    """Bounded observation-only workers for the connected local demonstrator."""

    def __init__(
        self,
        *,
        runtime_cache: _RuntimeCache | None = None,
        maximum_workers: int = MAX_ANALYSIS_WORKERS,
    ) -> None:
        self._runtime_cache = runtime_cache or bundle_runtime_cache
        self.maximum_workers = maximum_workers
        self._records: dict[str, _WorkerRecord] = {}
        self._epochs: dict[str, int] = {}
        self._lock = RLock()

    def start(self, session_id: str) -> DemoAnalysisView:
        session = network_sessions.get(session_id)
        if session is None:
            raise KeyError("sensor-network session not found")
        with self._lock:
            existing = self._records.get(session_id)
            if existing is not None and existing.thread is not None:
                if existing.thread.is_alive():
                    return self._view(existing)
            if len(self._records) >= self.maximum_workers and session_id not in self._records:
                terminal = [
                    item
                    for item in self._records.values()
                    if item.thread is None or not item.thread.is_alive()
                ]
                if not terminal:
                    raise AnalysisCapacityError("bounded analysis-worker registry is full")
                oldest = min(terminal, key=lambda item: item.worker_epoch)
                del self._records[oldest.session_id]
            epoch = self._epochs.get(session_id, 0) + 1
            self._epochs[session_id] = epoch
            bundles = self._runtime_cache.get()
            initial_state = (
                "awaiting_reference" if bundles is not None else "bundle_unavailable"
            )
            detail = (
                "Collecting the first 8 seconds of observations for a frozen reference."
                if bundles is not None
                else "No compatible local/network bundle pair has been applied."
            )
            record = _WorkerRecord(
                session_id=session_id,
                worker_epoch=epoch,
                state=initial_state,
                state_detail=detail,
                cancellation=Event(),
            )
            self._records[session_id] = record
            if bundles is not None:
                self._bind_application(record, bundles)
                worker = Thread(
                    target=self._run,
                    args=(record, session),
                    name=f"aqse-analysis-{session_id[:8]}-{epoch}",
                    daemon=True,
                )
                record.thread = worker
                worker.start()
            return self._view(record)

    def get(self, session_id: str) -> DemoAnalysisView | None:
        with self._lock:
            record = self._records.get(session_id)
        return None if record is None else self._view(record)

    def stop(self, session_id: str) -> DemoAnalysisView | None:
        with self._lock:
            record = self._records.get(session_id)
        if record is None:
            return None
        record.cancellation.set()
        thread = record.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with record.lock:
            if thread is not None and thread.is_alive():
                record.state = "failed"
                record.state_detail = "Analysis worker did not stop within 2 seconds."
                record.error = record.state_detail
            elif record.state != "failed":
                record.state = "stopped"
                record.state_detail = "Continuous analysis stopped explicitly."
        return self._view(record)

    def invalidate_bundle_cache(self) -> None:
        self._runtime_cache.invalidate()
        with self._lock:
            records = tuple(self._records.values())
            for record in records:
                self._epochs[record.session_id] = record.worker_epoch + 1
        for record in records:
            with record.lock:
                worker_running = (
                    record.thread is not None
                    and record.thread.is_alive()
                    and not record.cancellation.is_set()
                )
                record.worker_epoch += 1
                record.application_epoch += 1
                record.application_id = None
                record.active_local_bundle_id = None
                record.active_network_bundle_id = None
                record.reference_origin_frame_id = record.cursor_frame_id + 1
                record.frames.clear()
                record.reference = None
                record.latest_analyzed_window_start_s = None
                record.completed_window_count = 0
                record.skipped_window_count = 0
                record.quality_abstention_count = 0
                record.latest_results = ()
                record.latency_samples_ms.clear()
                record.state = "awaiting_reference" if worker_running else "stopped"
                record.state_detail = (
                    "Bundle generation changed; acquiring a new observed reference."
                    if worker_running
                    else (
                        "Bundle generation changed while analysis was stopped; "
                        "start analysis to acquire a new observed reference."
                    )
                )
                record.error = None

    def clear(self) -> None:
        with self._lock:
            records = tuple(self._records.values())
            self._records.clear()
        for record in records:
            record.cancellation.set()
        for record in records:
            if record.thread is not None and record.thread.is_alive():
                record.thread.join(timeout=2.0)

    def _run(self, record: _WorkerRecord, session: NetworkSession) -> None:
        try:
            while not record.cancellation.is_set():
                status = session.view().status
                if status.state.value == "stopped":
                    with record.lock:
                        record.state = "stopped"
                        record.state_detail = (
                            "Continuous analysis stopped with its sensor session."
                        )
                    return
                with record.lock:
                    if status.latest_frame_id < record.cursor_frame_id:
                        self._reset_epoch_locked(record)
                batch = session.observation_batch(record.cursor_frame_id, 1_000)
                if batch.gap_detected:
                    self._fail(
                        record,
                        "Observation buffer gap prevents a causal reference; reset the session.",
                    )
                    return
                if batch.frames:
                    with record.lock:
                        record.frames.extend(batch.frames)
                        record.cursor_frame_id = batch.to_frame_id or record.cursor_frame_id
                        record.latest_observation_time_s = batch.frames[-1].sim_time_s
                    self._advance(record)
                elif record.state == "paused_for_training":
                    # Retry the one bounded newest-window slot after training releases
                    # the shared exact-state admission guard; no backlog is enqueued.
                    self._advance(record)
                sleep(WORKER_POLL_INTERVAL_S)
        except Exception as exc:
            self._fail(record, f"{type(exc).__name__}: {exc}"[:500])

    def _advance(self, record: _WorkerRecord) -> None:
        with record.lock:
            frames = tuple(record.frames)
            if record.reference is None:
                if not frames:
                    return
                origin_id = frames[0].frame_id
                if origin_id != record.reference_origin_frame_id:
                    self._fail(
                        record,
                        "Reference acquisition did not begin at its declared origin "
                        f"frame {record.reference_origin_frame_id}; reset the session.",
                    )
                    return
                if len(frames) < STATE8_REFERENCE_SAMPLES:
                    return
                reference_frames = frames[:STATE8_REFERENCE_SAMPLES]
                record.reference = build_state8_reference(reference_frames)
                record.state = "running"
                record.state_detail = "Observed reference frozen; waiting for a complete window."
            reference = record.reference
            assert reference is not None
            application_epoch = record.application_epoch
            latest_relative_index = int(
                round(
                    (
                        record.latest_observation_time_s
                        - reference.reference_start_time_s
                    )
                    * 100.0
                )
            )
            latest_possible_start = (
                latest_relative_index - STATE8_WINDOW_SAMPLES + 1
            )
            if latest_possible_start < STATE8_REFERENCE_SAMPLES:
                return
            start_index = STATE8_REFERENCE_SAMPLES + (
                (latest_possible_start - STATE8_REFERENCE_SAMPLES) // 100
            ) * 100
            start_time_s = reference.reference_start_time_s + start_index / 100.0
            previous_start = record.latest_analyzed_window_start_s
            if previous_start is not None and start_time_s <= previous_start:
                return
            window_frames = tuple(
                frame
                for frame in frames
                if start_time_s <= frame.sim_time_s < start_time_s + 4.0
            )
        if len(window_frames) < STATE8_WINDOW_SAMPLES:
            return
        if not heavy_quantum_slot.acquire(blocking=False):
            with record.lock:
                record.state = "paused_for_training"
                record.state_detail = (
                    "Measurements continue; exact-state inference waits for the shared "
                    "quantum slot."
                )
            return
        started = perf_counter()
        try:
            bundles = self._runtime_cache.get()
            if bundles is None:
                with record.lock:
                    record.state = "bundle_unavailable"
                    record.state_detail = "No compatible bundle pair is applied."
                return
            with record.lock:
                if record.application_id != bundles.pointer.application_id:
                    record.latest_results = ()
                    self._bind_application(record, bundles)
                application_id = record.application_id
            results = self._analyze_window(
                record,
                bundles,
                window_frames,
                reference=reference,
                start_time_s=start_time_s,
            )
            duration_ms = (perf_counter() - started) * 1_000.0
            results = tuple(
                result.model_copy(update={"processing_duration_ms": duration_ms})
                for result in results
            )
            with record.lock:
                if (
                    record.application_epoch != application_epoch
                    or record.application_id != application_id
                    or record.reference is None
                    or record.reference.reference_id != reference.reference_id
                ):
                    return
                if previous_start is not None:
                    skipped = max(
                        0,
                        int(round(start_time_s - previous_start)) - 1,
                    )
                    record.skipped_window_count += skipped
                record.latest_results = results
                record.latest_analyzed_window_start_s = start_time_s
                record.completed_window_count += 1
                record.quality_abstention_count += sum(
                    not result.feature_valid for result in results
                )
                record.latency_samples_ms.append(duration_ms)
                record.state = "running"
                record.state_detail = "Latest complete causal window analyzed."
        finally:
            heavy_quantum_slot.release()

    def _analyze_window(
        self,
        record: _WorkerRecord,
        bundles: AppliedBundleSet,
        frames: tuple[ObservationFrame, ...],
        *,
        reference: State8ObservedReference,
        start_time_s: float,
    ) -> tuple[DemoAnalysisResult, ...]:
        local = extract_latest_state8_window(
            frames,
            reference=reference,
            profile_id=LOCAL_STATE8_PROFILE_ID,
        )
        network = (
            extract_latest_state8_window(
                frames,
                reference=reference,
                profile_id=NETWORK_STATE8_PROFILE_ID,
            )
            if len(reference.node_order) >= 3
            else None
        )
        local_by_sensor = {item.sensor_id: item for item in local.records}
        network_by_sensor = (
            {} if network is None else {item.sensor_id: item for item in network.records}
        )
        selected: list[
            tuple[State8FeatureRecord, BundleRuntime, ContextMode, str]
        ] = []
        node_count = len(reference.node_order)
        for sensor_id in reference.node_order:
            local_record = local_by_sensor[sensor_id]
            network_record = network_by_sensor.get(sensor_id)
            if network_record is not None and network_record.quality.valid_for_quantum:
                selected.append(
                    (
                        network_record,
                        bundles.network,
                        "network",
                        "Peer context supports network-pattern classification.",
                    )
                )
            elif node_count == 2:
                selected.append(
                    (
                        local_record,
                        bundles.local,
                        "local_two_node_ambiguous",
                        "Two nodes detect change locally but cannot identify its cause.",
                    )
                )
            elif node_count >= 3:
                selected.append(
                    (
                        local_record,
                        bundles.local,
                        "degraded_local",
                        "Peer context is invalid; result is degraded to local change detection.",
                    )
                )
            else:
                selected.append(
                    (
                        local_record,
                        bundles.local,
                        "local",
                        "A single node supports change detection, not causal attribution.",
                    )
                )

        grouped: dict[str, list[tuple[State8FeatureRecord, BundleRuntime, ContextMode, str]]] = {}
        for item in selected:
            grouped.setdefault(item[1].artifact.bundle_id, []).append(item)
        output: list[DemoAnalysisResult] = []
        produced_at = _utc_now()
        for items in grouped.values():
            runtime = items[0][1]
            eligible = [item for item in items if item[0].quality.valid_for_quantum]
            prediction_by_window: dict[
                str,
                tuple[BundlePredictionBatch, tuple[float, ...]],
            ] = {}
            if eligible:
                raw = np.asarray([item[0].values for item in eligible], dtype=np.float64)
                identifiers = tuple(item[0].window_id for item in eligible)
                profile = state8_profile(runtime.artifact.profile_id)
                encoded = runtime.encoder.transform(raw, profile=profile)
                context = BundleQueryContext(
                    bundle_id=runtime.artifact.bundle_id,
                    task_id=runtime.artifact.task_id,
                    profile_id=runtime.artifact.profile_id,
                    profile_fingerprint=runtime.artifact.profile_fingerprint,
                    query_acquisition_id=f"{record.session_id}:{reference.reference_id}",
                )
                prediction = runtime.predict(
                    raw,
                    sample_ids=identifiers,
                    context=context,
                )
                for index, identifier in enumerate(identifiers):
                    prediction_by_window[identifier] = (
                        prediction,
                        tuple(float(value) for value in encoded[index]),
                    )
            for feature, bound_runtime, mode, note in items:
                prediction_entry = prediction_by_window.get(feature.window_id)
                data_age_ms = max(
                    0.0,
                    (record.latest_observation_time_s - feature.end_exclusive_time_s)
                    * 1_000.0,
                )
                if prediction_entry is None:
                    output.append(
                        self._abstention_result(
                            record,
                            bundles,
                            bound_runtime,
                            feature,
                            mode,
                            note,
                            produced_at,
                            data_age_ms,
                            node_count=len(reference.node_order),
                        )
                    )
                    continue
                prediction, encoded_angles = prediction_entry
                index = prediction.sample_ids.index(feature.window_id)
                output.append(
                    DemoAnalysisResult(
                        result_id=_result_id(
                            session_id=record.session_id,
                            application_id=bundles.pointer.application_id,
                            window_id=feature.window_id,
                        ),
                        produced_at_utc=produced_at,
                        session_id=record.session_id,
                        application_id=bundles.pointer.application_id,
                        bundle_id=bound_runtime.artifact.bundle_id,
                        task_id=bound_runtime.artifact.task_id,
                        profile_id=feature.profile_id,
                        profile_fingerprint=feature.profile_fingerprint,
                        theta_id=bound_runtime.artifact.afse.theta_id,
                        reference_id=feature.reference_id,
                        node_count=len(reference.node_order),
                        sensor_id=feature.sensor_id,
                        peer_sensor_ids=feature.peer_sensor_ids,
                        context_mode=mode,
                        attribution_note=note,
                        window_id=feature.window_id,
                        window_start_s=feature.start_time_s,
                        window_end_exclusive_s=feature.end_exclusive_time_s,
                        source_frame_ids=feature.source_frame_ids,
                        feature_names=state8_profile(feature.profile_id).feature_names,
                        feature_units=state8_profile(feature.profile_id).feature_units,
                        feature_values=feature.values,
                        feature_valid=True,
                        quality_flags=(),
                        encoded_angles=encoded_angles,
                        afse_vector=prediction.afse_vectors[index],
                        afse_reference_size=bound_runtime.artifact.afse.reference_size,
                        reconstruction_residual=(
                            prediction.reconstruction_residuals[index]
                        ),
                        heuristic_ood=prediction.heuristic_ood[index],
                        class_order=prediction.model_scores.class_order,
                        class_scores=prediction.model_scores.scores[index],
                        predicted_class=prediction.model_scores.predicted_classes[index],
                        displayed_class=prediction.displayed_classes[index],
                        uncertain=prediction.model_scores.uncertain[index],
                        top_score=prediction.model_scores.top_scores[index],
                        top_two_margin=prediction.model_scores.top_two_margins[index],
                        raw_baseline_scores=prediction.raw_baseline_scores.scores[index],
                        raw_baseline_class=(
                            prediction.raw_baseline_scores.predicted_classes[index]
                        ),
                        observable_rule_status=(
                            prediction.observable_rule_statuses[index]
                        ),
                        data_age_ms=data_age_ms,
                        processing_duration_ms=0.0,
                    )
                )
        return tuple(sorted(output, key=lambda item: item.sensor_id))

    @staticmethod
    def _abstention_result(
        record: _WorkerRecord,
        bundles: AppliedBundleSet,
        runtime: BundleRuntime,
        feature: State8FeatureRecord,
        mode: ContextMode,
        note: str,
        produced_at: str,
        data_age_ms: float,
        node_count: int,
    ) -> DemoAnalysisResult:
        profile = state8_profile(feature.profile_id)
        return DemoAnalysisResult(
            result_id=_result_id(
                session_id=record.session_id,
                application_id=bundles.pointer.application_id,
                window_id=feature.window_id,
            ),
            produced_at_utc=produced_at,
            session_id=record.session_id,
            application_id=bundles.pointer.application_id,
            bundle_id=runtime.artifact.bundle_id,
            task_id=runtime.artifact.task_id,
            profile_id=feature.profile_id,
            profile_fingerprint=feature.profile_fingerprint,
            theta_id=runtime.artifact.afse.theta_id,
            reference_id=feature.reference_id,
            node_count=node_count,
            sensor_id=feature.sensor_id,
            peer_sensor_ids=feature.peer_sensor_ids,
            context_mode=mode,
            attribution_note=note,
            window_id=feature.window_id,
            window_start_s=feature.start_time_s,
            window_end_exclusive_s=feature.end_exclusive_time_s,
            source_frame_ids=feature.source_frame_ids,
            feature_names=profile.feature_names,
            feature_units=profile.feature_units,
            feature_values=feature.values,
            feature_valid=False,
            quality_flags=feature.quality.flags,
            encoded_angles=None,
            afse_vector=None,
            afse_reference_size=None,
            reconstruction_residual=None,
            heuristic_ood=None,
            class_order=runtime.artifact.class_order,
            class_scores=None,
            predicted_class=None,
            displayed_class="ABSTAIN",
            uncertain=None,
            top_score=None,
            top_two_margin=None,
            raw_baseline_scores=None,
            raw_baseline_class=None,
            observable_rule_status=None,
            data_age_ms=data_age_ms,
            processing_duration_ms=0.0,
        )

    @staticmethod
    def _bind_application(
        record: _WorkerRecord,
        bundles: AppliedBundleSet,
    ) -> None:
        record.application_id = bundles.pointer.application_id
        record.active_local_bundle_id = bundles.pointer.local_bundle_id
        record.active_network_bundle_id = bundles.pointer.network_bundle_id

    @staticmethod
    def _reset_epoch_locked(record: _WorkerRecord) -> None:
        record.worker_epoch += 1
        record.state = "awaiting_reference"
        record.state_detail = "Session reset detected; acquiring a new observed reference."
        record.frames.clear()
        record.reference = None
        record.reference_origin_frame_id = 1
        record.application_epoch += 1
        record.cursor_frame_id = 0
        record.latest_observation_time_s = 0.0
        record.latest_analyzed_window_start_s = None
        record.completed_window_count = 0
        record.skipped_window_count = 0
        record.quality_abstention_count = 0
        record.latest_results = ()
        record.latency_samples_ms.clear()
        record.error = None

    @staticmethod
    def _fail(record: _WorkerRecord, message: str) -> None:
        with record.lock:
            record.state = "failed"
            record.state_detail = message
            record.error = message

    @staticmethod
    def _view(record: _WorkerRecord) -> DemoAnalysisView:
        with record.lock:
            latencies = np.asarray(record.latency_samples_ms, dtype=np.float64)
            p50 = None if not len(latencies) else float(np.percentile(latencies, 50))
            p95 = None if not len(latencies) else float(np.percentile(latencies, 95))
            if record.latest_results:
                produced = datetime.fromisoformat(
                    record.latest_results[0].produced_at_utc.replace("Z", "+00:00")
                )
                age = max(
                    0.0,
                    (datetime.now(timezone.utc) - produced).total_seconds() * 1_000.0,
                )
            else:
                age = None
            progress = min(
                1.0,
                len(record.frames) / float(STATE8_REFERENCE_SAMPLES),
            )
            return DemoAnalysisView(
                session_id=record.session_id,
                state=record.state,
                state_detail=record.state_detail,
                worker_epoch=record.worker_epoch,
                application_id=record.application_id,
                active_local_bundle_id=record.active_local_bundle_id,
                active_network_bundle_id=record.active_network_bundle_id,
                reference_id=(
                    None if record.reference is None else record.reference.reference_id
                ),
                reference_progress=progress,
                latest_observation_frame_id=record.cursor_frame_id,
                latest_observation_time_s=record.latest_observation_time_s,
                latest_analyzed_window_start_s=record.latest_analyzed_window_start_s,
                completed_window_count=record.completed_window_count,
                skipped_window_count=record.skipped_window_count,
                quality_abstention_count=record.quality_abstention_count,
                queue_depth=1 if record.state == "paused_for_training" else 0,
                latency_p50_ms=p50,
                latency_p95_ms=p95,
                result_age_ms=age,
                latest_results=record.latest_results,
                error=record.error,
            )


demo_analysis = DemoAnalysisService()
