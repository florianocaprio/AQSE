from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.demo import router
from app.demo.runtime_models import (
    DemoAnalysisView,
    DemoRegistryView,
    DemoTrainingJobView,
)
from app.features.state8 import state8_profile, state8_profile_fingerprint
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
)

api = FastAPI()
api.include_router(router, prefix="/api")
client = TestClient(api)


def _current_local_result(
    *,
    profile_id: str = LOCAL_STATE8_PROFILE_ID,
    feature_valid: bool = True,
    source_frame_count: int = 400,
) -> SimpleNamespace:
    profile = state8_profile(profile_id)
    return SimpleNamespace(
        sensor_id="S1",
        task_id="aqse.local-change.v1",
        profile_id=profile_id,
        profile_fingerprint=state8_profile_fingerprint(profile),
        window_id="knowledge-api-window-1",
        session_id="knowledge-api-session",
        reference_id="aqse-state8-reference-1111111111111111",
        window_start_s=8.0,
        window_end_exclusive_s=12.0,
        source_frame_ids=tuple(range(1, source_frame_count + 1)),
        peer_sensor_ids=(),
        feature_values=(1.0, 0.5, 0.1, -2.0, 0.0, 0.25, 0.75, 1.0),
        node_count=1,
        feature_valid=feature_valid,
    )


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
    assert response.json()["selection_freezes"] == []


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


def test_knowledge_api_keeps_observations_and_human_labels_separate_and_idempotent(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AQSE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "app.api.demo.demo_analysis.get",
        lambda _session_id: SimpleNamespace(
            latest_results=(_current_local_result(),),
        ),
    )
    capture_payload = {
        "session_id": "knowledge-api-session",
        "sensor_id": "S1",
        "task": "local",
    }

    initial = client.get("/api/demo/knowledge")
    captured = client.post(
        "/api/demo/knowledge/observations",
        json=capture_payload,
    )
    recaptured = client.post(
        "/api/demo/knowledge/observations",
        json=capture_payload,
    )

    assert initial.status_code == 200
    assert initial.json()["observations"] == []
    assert initial.json()["reviewed_labels"] == []
    assert captured.status_code == 201
    assert captured.json()["reused"] is False
    assert recaptured.status_code == 200
    assert recaptured.json()["reused"] is True
    observation = captured.json()["observation"]
    assert observation["observation_id"] == recaptured.json()["observation"][
        "observation_id"
    ]
    assert not {
        "label",
        "prediction",
        "predicted_class",
        "truth",
    } & observation.keys()

    label_payload = {
        "observation_id": observation["observation_id"],
        "task": "local",
        "label": "NORMAL",
        "reviewer_id": "human-reviewer-api",
        "reviewed_at_utc": "2026-09-10T10:00:00Z",
    }
    labelled = client.post("/api/demo/knowledge/labels", json=label_payload)
    relabelled = client.post("/api/demo/knowledge/labels", json=label_payload)

    assert labelled.status_code == 201
    assert labelled.json()["reused"] is False
    assert relabelled.status_code == 200
    assert relabelled.json()["reused"] is True
    assert "feature" not in labelled.json()["label"]

    approval_payload = {
        "partition": "TRAIN",
        "task": "local",
        "observation_ids": [observation["observation_id"]],
        "approved_by": "principal-investigator-api",
        "approved_at_utc": "2026-09-10T11:00:00Z",
        "approval_declaration": (
            "explicitly approved for bounded TRAIN-only retraining"
        ),
    }
    approved = client.post(
        "/api/demo/knowledge/train-collections",
        json=approval_payload,
    )
    reapproved = client.post(
        "/api/demo/knowledge/train-collections",
        json=approval_payload,
    )
    registry = client.get("/api/demo/knowledge")

    assert approved.status_code == 201
    assert approved.json()["reused"] is False
    assert approved.json()["collection"]["partition"] == "TRAIN"
    assert reapproved.status_code == 200
    assert reapproved.json()["reused"] is True
    assert registry.status_code == 200
    assert len(registry.json()["observations"]) == 1
    assert len(registry.json()["reviewed_labels"]) == 1
    assert len(registry.json()["approved_train_collections"]) == 1
    assert "label" not in registry.json()["observations"][0]
    assert "feature" not in registry.json()["reviewed_labels"][0]


def test_knowledge_capture_rejects_incompatible_or_incomplete_results(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AQSE_ARTIFACT_ROOT", str(tmp_path))
    current = _current_local_result()
    monkeypatch.setattr(
        "app.api.demo.demo_analysis.get",
        lambda _session_id: SimpleNamespace(latest_results=(current,)),
    )

    incompatible_task = client.post(
        "/api/demo/knowledge/observations",
        json={
            "session_id": "knowledge-api-session",
            "sensor_id": "S1",
            "task": "network",
        },
    )
    assert incompatible_task.status_code == 409
    assert "differs" in incompatible_task.json()["detail"]

    incomplete = _current_local_result(source_frame_count=399)
    monkeypatch.setattr(
        "app.api.demo.demo_analysis.get",
        lambda _session_id: SimpleNamespace(latest_results=(incomplete,)),
    )
    incomplete_capture = client.post(
        "/api/demo/knowledge/observations",
        json={
            "session_id": "knowledge-api-session",
            "sensor_id": "S1",
            "task": "local",
        },
    )
    assert incomplete_capture.status_code == 409
    assert "complete eligible" in incomplete_capture.json()["detail"]

    incompatible_profile = _current_local_result(
        profile_id=NETWORK_STATE8_PROFILE_ID,
    )
    monkeypatch.setattr(
        "app.api.demo.demo_analysis.get",
        lambda _session_id: SimpleNamespace(latest_results=(incompatible_profile,)),
    )
    incompatible_capture = client.post(
        "/api/demo/knowledge/observations",
        json={
            "session_id": "knowledge-api-session",
            "sensor_id": "S1",
            "task": "local",
        },
    )
    registry = client.get("/api/demo/knowledge")

    assert incompatible_capture.status_code == 409
    assert "incompatible state8 profile" in incompatible_capture.json()["detail"]
    assert registry.status_code == 200
    assert registry.json()["observations"] == []
