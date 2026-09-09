from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.demo import router
from app.demo.runtime_models import (
    DemoAnalysisView,
    DemoRegistryView,
    DemoTrainingJobView,
)

api = FastAPI()
api.include_router(router, prefix="/api")
client = TestClient(api)


def _analysis(session_id: str, state: str = "awaiting_reference") -> DemoAnalysisView:
    return DemoAnalysisView(
        session_id=session_id,
        state=state,
        state_detail="fixture",
        worker_epoch=1,
        reference_progress=0.5,
        latest_observation_frame_id=400,
        latest_observation_time_s=3.99,
        completed_window_count=0,
        skipped_window_count=0,
        quality_abstention_count=0,
        queue_depth=0,
        latest_results=(),
    )


def test_demo_registry_reports_unprepared_state_without_inventing_metrics(
    monkeypatch,
) -> None:
    view = DemoRegistryView(
        prepared=False,
        study_artifact_id=None,
        study_content_digest=None,
        selection_freeze_id=None,
        final_evaluation_id=None,
        historical_test_ledger_sha256="a" * 64,
        active=None,
        bundles=(),
        final_metrics=(),
        preparation_detail="Run make prepare-demo once.",
    )
    monkeypatch.setattr("app.api.demo.demo_registry_view", lambda: view)

    response = client.get("/api/demo/registry")

    assert response.status_code == 200
    assert response.json()["prepared"] is False
    assert response.json()["final_metrics"] == []


def test_analysis_controls_return_real_worker_state(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.demo.demo_analysis.start",
        lambda session_id: _analysis(session_id),
    )
    monkeypatch.setattr(
        "app.api.demo.demo_analysis.get",
        lambda session_id: _analysis(session_id, "running"),
    )
    monkeypatch.setattr(
        "app.api.demo.demo_analysis.stop",
        lambda session_id: _analysis(session_id, "stopped"),
    )

    started = client.post(
        "/api/demo/analysis/session-fixture/start",
        json={"acquire_reference": True},
    )
    current = client.get("/api/demo/analysis/session-fixture")
    stopped = client.post("/api/demo/analysis/session-fixture/stop")

    assert started.status_code == 200
    assert started.json()["state"] == "awaiting_reference"
    assert current.json()["state"] == "running"
    assert stopped.json()["state"] == "stopped"


def test_training_retry_uses_http_200_and_same_job(monkeypatch) -> None:
    job = DemoTrainingJobView(
        job_id="aqse-demo-training-1111111111111111",
        intent_id="aqse-demo-intent-api",
        study_artifact_id="aqse-network-study-fixture",
        state="running",
        created_at_utc="2026-01-01T00:00:00Z",
        updated_at_utc="2026-01-01T00:00:00Z",
        current_stage="fixture",
        progress=(),
    )
    calls = 0

    def start(_intent):
        nonlocal calls
        calls += 1
        return job, calls == 1

    monkeypatch.setattr("app.api.demo.demo_training_jobs.start", start)
    payload = {
        "intent_id": "aqse-demo-intent-api",
        "study_artifact_id": "aqse-network-study-fixture",
        "requested_action": "fit-two-candidate-local-and-network-bundles",
    }

    first = client.post("/api/demo/training/jobs", json=payload)
    retry = client.post("/api/demo/training/jobs", json=payload)

    assert first.status_code == 201
    assert retry.status_code == 200
    assert first.json()["job_id"] == retry.json()["job_id"]
