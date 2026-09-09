from __future__ import annotations

from threading import Event
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient

import app.api.quantum_preview as quantum_preview_api
import app.api.training as training_api
from app.main import app
from app.quantum.admission import heavy_quantum_slot
from app.training.run_models import TrainingJobState
from app.training.service import TrainingJobBusyError, TrainingJobService
from app.training.workflow import TrainingJobOutcome


def _wait_terminal(service: TrainingJobService, job_id: str):
    deadline = monotonic() + 3.0
    while monotonic() < deadline:
        view = service.get(job_id)
        if view is not None and view.state in {
            TrainingJobState.COMPLETED,
            TrainingJobState.CANCELLED,
            TrainingJobState.FAILED,
        }:
            return view
        sleep(0.005)
    raise TimeoutError("training job did not reach a terminal state")


def test_one_active_job_busy_admission_and_cooperative_cancel() -> None:
    started = Event()
    release = Event()

    def blocked_runner(cancellation: Event) -> TrainingJobOutcome:
        started.set()
        assert release.wait(timeout=2.0)
        return TrainingJobOutcome(
            run_artifact_id="aqse-qng-run-cancelled0",
            accepted_update_count=1,
            stop_reason="CANCELLED" if cancellation.is_set() else "MAX_UPDATES_REACHED",
        )

    service = TrainingJobService(blocked_runner)
    job = service.start()
    assert started.wait(timeout=1.0)
    with pytest.raises(TrainingJobBusyError, match="already active"):
        service.start()
    assert not heavy_quantum_slot.acquire(blocking=False)
    service.cancel(job.job_id)
    release.set()
    terminal = _wait_terminal(service, job.job_id)

    assert terminal.state is TrainingJobState.CANCELLED
    assert terminal.accepted_update_count == 1
    assert heavy_quantum_slot.acquire(blocking=False)
    heavy_quantum_slot.release()


def test_completed_cancel_is_harmless_and_registry_is_bounded() -> None:
    counter = 0

    def completed_runner(cancellation: Event) -> TrainingJobOutcome:
        nonlocal counter
        counter += 1
        return TrainingJobOutcome(
            run_artifact_id=f"aqse-qng-run-{counter:016x}",
            accepted_update_count=10,
            stop_reason="MAX_UPDATES_REACHED",
        )

    service = TrainingJobService(completed_runner, maximum_records=1)
    first = service.start()
    first_terminal = _wait_terminal(service, first.job_id)
    after_cancel = service.cancel(first.job_id)
    assert after_cancel == first_terminal
    second = service.start()
    _wait_terminal(service, second.job_id)
    assert service.get(first.job_id) is None


def test_failure_is_preserved_without_fabricated_completion() -> None:
    def failed_runner(cancellation: Event) -> TrainingJobOutcome:
        raise RuntimeError("controlled failure")

    service = TrainingJobService(failed_runner)
    job = service.start()
    terminal = _wait_terminal(service, job.job_id)

    assert terminal.state is TrainingJobState.FAILED
    assert terminal.run_artifact_id is None
    assert terminal.accepted_update_count == 0
    assert terminal.error == "RuntimeError: controlled failure"


def test_training_api_is_explicit_and_lightweight_health_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()

    def blocked_runner(cancellation: Event) -> TrainingJobOutcome:
        started.set()
        assert release.wait(timeout=2.0)
        return TrainingJobOutcome(
            run_artifact_id="aqse-qng-run-api0000000000000",
            accepted_update_count=0,
            stop_reason="CANCELLED" if cancellation.is_set() else "NO_ACCEPTED_UPDATE",
        )

    service = TrainingJobService(blocked_runner)
    monkeypatch.setattr(training_api, "training_jobs", service)
    client = TestClient(app)
    response = client.post("/api/training/jobs")
    assert response.status_code == 201
    job_id = response.json()["job_id"]
    assert started.wait(timeout=1.0)

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/quantum/health").status_code == 200
    assert client.get("/api/network/health").status_code == 200
    defaults = client.get("/api/network/defaults", params={"node_count": 1})
    session = client.post("/api/network/sessions", json=defaults.json())
    session_id = session.json()["status"]["session_id"]
    simulator_started = monotonic()
    simulator = client.post(
        f"/api/network/sessions/{session_id}/step",
        json={"frames": 1},
    )
    simulator_latency_s = monotonic() - simulator_started
    assert simulator.status_code == 200
    assert simulator_latency_s < 1.0
    assert client.post("/api/quantum/diagnostics").status_code == 429
    assert quantum_preview_api.preview_slot is heavy_quantum_slot
    assert client.post("/api/training/jobs").status_code == 429
    assert client.get("/api/training/jobs/unknown").status_code == 404
    cancel = client.post(f"/api/training/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    release.set()
    terminal = _wait_terminal(service, job_id)
    assert terminal.state is TrainingJobState.CANCELLED
    assert client.delete(f"/api/network/sessions/{session_id}").status_code == 204
