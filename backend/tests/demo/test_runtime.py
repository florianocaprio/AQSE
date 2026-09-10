from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace

import numpy as np

from app.classical.models import MLPScoreBatch
from app.demo.bundle_models import (
    LOCAL_TASK_ID,
    NETWORK_TASK_ID,
    ActiveBundlePointer,
    BundlePredictionBatch,
    canonical_digest,
)
from app.demo.bundle_storage import AppliedBundleSet
from app.demo.runtime import DemoAnalysisService
from app.features.state8 import state8_profile, state8_profile_fingerprint
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
)
from app.network.defaults import default_network_configuration
from app.network.session import network_sessions
from app.quantum.admission import heavy_quantum_slot


class _FakeEncoder:
    def transform(self, features, *, profile):
        assert profile.profile_id in {
            LOCAL_STATE8_PROFILE_ID,
            NETWORK_STATE8_PROFILE_ID,
        }
        return np.asarray(features, dtype=np.float64) / 10.0


@dataclass
class _FakeRuntime:
    artifact: SimpleNamespace
    encoder: _FakeEncoder = _FakeEncoder()

    def predict(self, raw_features, *, sample_ids, context):
        assert context.query_acquisition_id != self.artifact.fitted_on_dataset_id
        matrix = np.asarray(raw_features, dtype=np.float64)
        classes = self.artifact.class_order
        scores = np.full((len(matrix), len(classes)), 0.1, dtype=np.float64)
        scores[:, 0] = 0.8
        score_batch = MLPScoreBatch(
            sample_ids=sample_ids,
            class_order=classes,
            scores=tuple(tuple(float(value) for value in row) for row in scores),
            predicted_classes=tuple(classes[0] for _ in sample_ids),
            displayed_classes=tuple(classes[0] for _ in sample_ids),
            uncertain=tuple(False for _ in sample_ids),
            top_scores=tuple(0.8 for _ in sample_ids),
            top_two_margins=tuple(0.7 for _ in sample_ids),
        )
        return BundlePredictionBatch(
            bundle_id=self.artifact.bundle_id,
            task_id=self.artifact.task_id,
            query_acquisition_id=context.query_acquisition_id,
            sample_ids=sample_ids,
            afse_vectors=tuple((0.25, 0.5) for _ in sample_ids),
            reconstruction_residuals=tuple(0.01 for _ in sample_ids),
            heuristic_ood=tuple(False for _ in sample_ids),
            model_scores=score_batch,
            displayed_classes=score_batch.displayed_classes,
            raw_baseline_scores=score_batch,
            observable_rule_statuses=tuple(
                "NO_OBSERVED_CHANGE" for _ in sample_ids
            ),
        )


class _FakeRuntimeCache:
    def __init__(self, bundle_set: AppliedBundleSet) -> None:
        self.bundle_set = bundle_set

    def get(self) -> AppliedBundleSet:
        return self.bundle_set

    def invalidate(self) -> None:
        pass


def _runtime(task_id: str, profile_id: str, suffix: str) -> _FakeRuntime:
    profile = state8_profile(profile_id)
    classes = (
        ("NORMAL", "CHANGE_DETECTED")
        if task_id == LOCAL_TASK_ID
        else (
            "NORMAL",
            "ENVIRONMENT_COMPATIBLE",
            "DEVICE_COMPATIBLE",
            "MIXED_OR_AMBIGUOUS",
        )
    )
    return _FakeRuntime(
        artifact=SimpleNamespace(
            bundle_id=f"aqse-demo-bundle-{suffix * 16}",
            task_id=task_id,
            profile_id=profile_id,
            profile_fingerprint=state8_profile_fingerprint(profile),
            fitted_on_dataset_id="offline-training-study",
            class_order=classes,
            afse=SimpleNamespace(
                theta_id=f"aqse-theta-{suffix * 16}",
                reference_size=2,
            ),
        )
    )


def _bundle_set(generation: int = 1) -> AppliedBundleSet:
    suffixes = ("1", "2") if generation == 1 else ("3", "4")
    freeze_suffix = "a" if generation == 1 else "b"
    pointer_payload = {
        "schema_version": "aqse.network-demo.active-bundles.v1",
        "generation": generation,
        "revision": generation,
        "selection_freeze_id": f"aqse-demo-freeze-{freeze_suffix * 16}",
        "local_bundle_id": f"aqse-demo-bundle-{suffixes[0] * 16}",
        "network_bundle_id": f"aqse-demo-bundle-{suffixes[1] * 16}",
        "previous_application_id": None,
    }
    return AppliedBundleSet(
        pointer=ActiveBundlePointer(
            **pointer_payload,
            application_id=(
                f"aqse-demo-application-{canonical_digest(pointer_payload)[:16]}"
            ),
        ),
        local=_runtime(LOCAL_TASK_ID, LOCAL_STATE8_PROFILE_ID, suffixes[0]),
        network=_runtime(NETWORK_TASK_ID, NETWORK_STATE8_PROFILE_ID, suffixes[1]),
    )


def _wait_for_state(
    service: DemoAnalysisService,
    session_id: str,
    *,
    completed: int,
    timeout_s: float = 4.0,
):
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        view = service.get(session_id)
        assert view is not None
        if view.completed_window_count >= completed or view.state == "failed":
            return view
        sleep(0.02)
    raise AssertionError(
        "analysis worker did not reach the expected bounded state: "
        f"{service.get(session_id)!r}"
    )


def test_live_worker_uses_latest_complete_window_and_observations_only() -> None:
    network_sessions.clear()
    session = network_sessions.create(default_network_configuration(1))
    # Any accidental truth-channel dependency is made fatal.
    session.truth_batch = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())  # type: ignore[method-assign]
    session.truth_snapshot = lambda: (_ for _ in ()).throw(AssertionError())  # type: ignore[method-assign]
    session.step(1_600)
    service = DemoAnalysisService(runtime_cache=_FakeRuntimeCache(_bundle_set()))  # type: ignore[arg-type]

    service.start(session.session_id)
    view = _wait_for_state(service, session.session_id, completed=1)

    assert view.state == "running"
    assert view.completed_window_count == 1
    assert view.latest_analyzed_window_start_s == 12.0
    assert len(view.latest_results) == 1
    result = view.latest_results[0]
    assert result.context_mode == "local"
    assert result.task_id == LOCAL_TASK_ID
    assert result.feature_valid
    assert result.afse_vector == (0.25, 0.5)
    assert result.displayed_class == "NORMAL"
    assert result.source_frame_ids == tuple(range(1_201, 1_601))
    service.clear()
    network_sessions.clear()


def test_training_slot_pauses_analysis_without_blocking_measurement() -> None:
    network_sessions.clear()
    session = network_sessions.create(default_network_configuration(1))
    session.step(1_200)
    service = DemoAnalysisService(runtime_cache=_FakeRuntimeCache(_bundle_set()))  # type: ignore[arg-type]
    assert heavy_quantum_slot.acquire(blocking=False)
    try:
        service.start(session.session_id)
        deadline = monotonic() + 2.0
        view = service.get(session.session_id)
        while view is not None and view.state != "paused_for_training" and monotonic() < deadline:
            sleep(0.02)
            view = service.get(session.session_id)
        assert view is not None
        assert view.state == "paused_for_training"
        previous_frame = view.latest_observation_frame_id
        session.step(100)
        sleep(0.1)
        current = service.get(session.session_id)
        assert current is not None
        assert current.latest_observation_frame_id >= previous_frame + 100
        assert current.queue_depth == 1
    finally:
        heavy_quantum_slot.release()

    resumed = _wait_for_state(service, session.session_id, completed=1)
    assert resumed.state == "running"
    assert resumed.latest_analyzed_window_start_s == 9.0
    service.clear()
    network_sessions.clear()


def test_analysis_worker_stops_when_sensor_session_stops() -> None:
    network_sessions.clear()
    session = network_sessions.create(default_network_configuration(1))
    service = DemoAnalysisService(runtime_cache=_FakeRuntimeCache(_bundle_set()))  # type: ignore[arg-type]
    service.start(session.session_id)

    session.stop()
    deadline = monotonic() + 2.0
    view = service.get(session.session_id)
    while view is not None and view.state != "stopped" and monotonic() < deadline:
        sleep(0.02)
        view = service.get(session.session_id)

    assert view is not None
    assert view.state == "stopped"
    assert view.state_detail == "Continuous analysis stopped with its sensor session."
    service.clear()
    network_sessions.clear()


def test_bundle_invalidation_discards_inflight_results_and_references() -> None:
    network_sessions.clear()
    session = network_sessions.create(default_network_configuration(1))
    session.step(1_600)
    cache = _FakeRuntimeCache(_bundle_set())
    service = DemoAnalysisService(runtime_cache=cache)  # type: ignore[arg-type]
    entered = Event()
    release = Event()
    original = service._analyze_window

    def blocking_analysis(*args, **kwargs):  # type: ignore[no-untyped-def]
        entered.set()
        assert release.wait(timeout=2.0)
        return original(*args, **kwargs)

    service._analyze_window = blocking_analysis  # type: ignore[method-assign]
    initial = service.start(session.session_id)
    assert entered.wait(timeout=2.0)

    cache.bundle_set = _bundle_set(2)
    service.invalidate_bundle_cache()
    invalidated = service.get(session.session_id)
    assert invalidated is not None
    assert invalidated.worker_epoch == initial.worker_epoch + 1
    assert invalidated.state == "awaiting_reference"
    assert invalidated.reference_id is None
    assert invalidated.application_id is None
    assert invalidated.latest_results == ()

    release.set()
    sleep(0.1)
    after_old_result = service.get(session.session_id)
    assert after_old_result is not None
    assert after_old_result.latest_results == ()
    assert after_old_result.completed_window_count == 0

    session.step(1_200)
    resumed = _wait_for_state(service, session.session_id, completed=1)
    assert resumed.application_id == _bundle_set(2).pointer.application_id
    assert resumed.reference_id is not None
    assert {item.bundle_id for item in resumed.latest_results} == {
        "aqse-demo-bundle-3333333333333333"
    }
    service.clear()
    network_sessions.clear()


def test_bundle_invalidation_preserves_stopped_worker_as_restartable() -> None:
    network_sessions.clear()
    session = network_sessions.create(default_network_configuration(1))
    session.step(900)
    cache = _FakeRuntimeCache(_bundle_set())
    service = DemoAnalysisService(runtime_cache=cache)  # type: ignore[arg-type]
    service.start(session.session_id)
    stopped = service.stop(session.session_id)
    assert stopped is not None
    assert stopped.state == "stopped"

    cache.bundle_set = _bundle_set(2)
    service.invalidate_bundle_cache()
    invalidated = service.get(session.session_id)
    assert invalidated is not None
    assert invalidated.state == "stopped"
    assert "start analysis" in invalidated.state_detail

    restarted = service.start(session.session_id)
    assert restarted.state in {"awaiting_reference", "running"}
    assert restarted.worker_epoch > invalidated.worker_epoch
    assert restarted.application_id == _bundle_set(2).pointer.application_id
    service.clear()
    network_sessions.clear()
