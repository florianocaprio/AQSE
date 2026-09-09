from __future__ import annotations

from pathlib import Path
from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.api.training as training_api
from app.main import app
from app.training.benchmark import (
    EngineeringBenchmarkResult,
    EngineeringBenchmarkService,
    EngineeringBenchmarkState,
)
from app.training.canonical import file_sha256
from app.training.execution_intent import (
    EngineeringBenchmarkIntent,
    TrainingExecutionIntent,
)
from app.training.intent_storage import FileExecutionIntentStore
from app.training.run_models import EvaluationCounters, TrainingJobState
from app.training.run_storage import load_training_run_artifact
from app.training.seal import verify_canonical_test_seal
from app.training.service import TrainingJobService
from app.training.storage import artifact_root
from app.training.trajectory import (
    QngTrajectoryContent,
    build_trajectory_audit_mapping,
    build_trajectory_identity,
    derive_trajectory_identity,
)
from app.training.trajectory_storage import (
    load_trajectory_audit_mapping,
    write_trajectory_audit_mapping,
)
from app.training.workflow import TrainingJobOutcome

DESIGNATED_ID = "aqse-qng-run-f00c702ad790df2b"
DUPLICATE_ID = "aqse-qng-run-5ae026e633def66b"
INPUT_FINGERPRINT = "aqse-training-input-e853259fac7c0eba"
RUN_SHA256 = {
    DESIGNATED_ID: "770bf5533adce51ccaf596e672fe1190dea609c8f3ba1087d5ca56e9862bd106",
    DUPLICATE_ID: "c781e8e5cc5584e1e9da2c89acdd165343e7ff43449dc33103f7bec7f05f87da",
}


def _historical_runs():  # type: ignore[no-untyped-def]
    root = artifact_root()
    paths = {run_id: root / "training-runs" / run_id for run_id in RUN_SHA256}
    if not all(path.is_dir() for path in paths.values()):
        pytest.skip("historical AQSE 1D.3 run artifacts are not mounted")
    runs = {
        run_id: load_training_run_artifact(
            path,
            expected_run_id=run_id,
            expected_training_input_fingerprint=INPUT_FINGERPRINT,
        )
        for run_id, path in paths.items()
    }
    return root, paths, runs


def _wait_training(service: TrainingJobService, job_id: str):  # type: ignore[no-untyped-def]
    deadline = monotonic() + 3.0
    while monotonic() < deadline:
        job = service.get(job_id)
        if job is not None and job.state in {
            TrainingJobState.COMPLETED,
            TrainingJobState.CANCELLED,
            TrainingJobState.FAILED,
        }:
            return job
        sleep(0.005)
    raise TimeoutError("training test job did not finish")


def _wait_benchmark(service: EngineeringBenchmarkService):  # type: ignore[no-untyped-def]
    deadline = monotonic() + 3.0
    while monotonic() < deadline:
        benchmark = service.get()
        if benchmark is not None and benchmark.state in {
            EngineeringBenchmarkState.COMPLETED,
            EngineeringBenchmarkState.FAILED,
        }:
            return benchmark
        sleep(0.005)
    raise TimeoutError("benchmark test job did not finish")


def test_historical_runs_are_unchanged_and_share_one_trajectory_identity() -> None:
    root, paths, runs = _historical_runs()
    identities = {run_id: derive_trajectory_identity(run) for run_id, run in runs.items()}

    assert file_sha256(paths[DESIGNATED_ID] / "run.json") == RUN_SHA256[DESIGNATED_ID]
    assert file_sha256(paths[DUPLICATE_ID] / "run.json") == RUN_SHA256[DUPLICATE_ID]
    assert identities[DESIGNATED_ID] == identities[DUPLICATE_ID]
    manifest, ledger_digest = verify_canonical_test_seal(
        root / "aqse-development-064acca20fc788c6"
    )
    assert manifest.test_state == "sealed"
    assert ledger_digest == (
        "210077e41754c47eebe572660f07b7cdaf8653ade2945fccd368e04d64a432c6"
    )


def test_trajectory_identity_excludes_runtime_measurements() -> None:
    _, _, runs = _historical_runs()
    run = runs[DESIGNATED_ID]
    changed_steps = tuple(
        step.model_copy(update={"step_wall_time_ms": step.step_wall_time_ms + 1000.0})
        for step in run.steps
    )
    changed_runtime = run.model_copy(
        update={
            "steps": changed_steps,
            "total_wall_time_ms": run.total_wall_time_ms + 10_000.0,
            "peak_process_rss_bytes": run.peak_process_rss_bytes + 1_000_000,
            "run_id": "aqse-qng-run-0000000000000000",
            "content_digest": "0" * 64,
        }
    )

    assert derive_trajectory_identity(run) == derive_trajectory_identity(changed_runtime)


def test_scientific_theta_input_and_optimizer_each_change_trajectory_identity() -> None:
    _, _, runs = _historical_runs()
    identity = derive_trajectory_identity(runs[DESIGNATED_ID])
    content = identity.content

    def shifted(theta: tuple[float, ...]) -> tuple[float, ...]:
        return (theta[0] + 0.01, *theta[1:])

    theta_content = QngTrajectoryContent.model_validate(
        content.model_copy(
            update={
                "theta0": shifted(content.theta0),
                "steps": tuple(
                    step.model_copy(
                        update={
                            "theta_before": shifted(step.theta_before),
                            "theta_after": shifted(step.theta_after),
                        }
                    )
                    for step in content.steps
                ),
                "final_theta": shifted(content.final_theta),
            }
        ).model_dump(mode="json")
    )
    input_content = content.model_copy(
        update={
            "training_input": content.training_input.model_copy(
                update={"fingerprint_id": "aqse-training-input-0000000000000000"}
            )
        }
    )
    optimizer_content = content.model_copy(
        update={
            "optimizer": content.optimizer.model_copy(update={"learning_rate": 0.25})
        }
    )

    assert build_trajectory_identity(theta_content) != identity
    assert build_trajectory_identity(input_content) != identity
    assert build_trajectory_identity(optimizer_content) != identity


def test_trajectory_audit_mapping_is_immutable_and_loads_back(tmp_path: Path) -> None:
    _, _, runs = _historical_runs()
    trajectory = derive_trajectory_identity(runs[DESIGNATED_ID])
    mapping = build_trajectory_audit_mapping(
        trajectory,
        designated_execution_id=DESIGNATED_ID,
        equivalent_execution_ids=(DUPLICATE_ID,),
        execution_run_sha256=RUN_SHA256,
    )
    path = write_trajectory_audit_mapping(mapping, root=tmp_path)

    assert load_trajectory_audit_mapping(
        path,
        expected_trajectory_id=trajectory.trajectory_id,
    ) == mapping
    assert write_trajectory_audit_mapping(mapping, root=tmp_path) == path
    changed = mapping.model_copy(update={"mapping_digest": "0" * 64})
    with pytest.raises(ValueError, match="mapping digest"):
        write_trajectory_audit_mapping(changed, root=tmp_path)


def test_execution_intent_is_idempotent_during_after_and_across_service_restart(
    tmp_path: Path,
) -> None:
    started = Event()
    release = Event()
    calls = 0

    def runner(cancellation: Event) -> TrainingJobOutcome:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=2.0)
        return TrainingJobOutcome(
            run_artifact_id="aqse-qng-run-1111111111111111",
            accepted_update_count=10,
            stop_reason="MAX_UPDATES_REACHED",
        )

    intent = TrainingExecutionIntent(intent_id="corrective-idempotency")
    service = TrainingJobService(
        runner,
        intent_store=FileExecutionIntentStore(tmp_path),
    )
    first = service.start(intent)
    assert first.created
    assert started.wait(timeout=1.0)
    repeated_running = service.start(intent)
    assert not repeated_running.created
    assert repeated_running.job.job_id == first.job.job_id
    assert calls == 1
    release.set()
    terminal = _wait_training(service, first.job.job_id)
    repeated_terminal = service.start(intent)
    assert not repeated_terminal.created
    assert repeated_terminal.job == terminal

    restarted = TrainingJobService(
        runner,
        intent_store=FileExecutionIntentStore(tmp_path),
    )
    recovered = restarted.start(intent)
    assert not recovered.created
    assert recovered.job == terminal
    assert calls == 1

    distinct = restarted.start(
        TrainingExecutionIntent(intent_id="corrective-idempotency-distinct")
    )
    assert distinct.created
    _wait_training(restarted, distinct.job.job_id)
    assert calls == 2


def test_engineering_benchmark_intent_is_separate_and_admission_is_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()

    def benchmark_runner() -> EngineeringBenchmarkResult:
        started.set()
        assert release.wait(timeout=2.0)
        return EngineeringBenchmarkResult(
            accepted_update_count=10,
            compute_wall_time_ms=100.0,
            end_to_end_wall_time_ms=110.0,
            peak_process_rss_bytes=1,
            counters=EvaluationCounters(
                differential_calls=10,
                gram_calls=11,
                statevector_evaluations=10_912,
            ),
        )

    benchmark_service = EngineeringBenchmarkService(benchmark_runner)
    monkeypatch.setattr(
        training_api,
        "qng_responsiveness_benchmarks",
        benchmark_service,
    )
    client = TestClient(app)
    feature_payload = {
        "series": {
            "acquisition_id": "corrective-admission",
            "sensor_id": "corrective-sensor",
            "sampling_rate_hz": 100.0,
            "time_s": (np.arange(400, dtype=float) / 100.0).tolist(),
            "measured_field": np.column_stack(
                (
                    25.0
                    + 8.0
                    * np.sin(2.0 * np.pi * 8.0 * np.arange(400, dtype=float) / 100.0),
                    np.full(400, 2.0),
                    np.zeros(400),
                )
            ).tolist(),
            "field_unit": "nT",
            "temperature_k": np.full(400, 293.15).tolist(),
            "saturation_mask": np.zeros((400, 3), dtype=bool).tolist(),
        },
        "channel": "x",
        "window": {"duration_s": 1.0, "overlap_fraction": 0.5},
    }
    extracted = client.post(
        "/api/features/vector-magnetometer/extract",
        json=feature_payload,
    ).json()
    preview_payload = {
        "mode": "self_reference",
        "backend": "numpy",
        "feature_profile": extracted["profile"],
        "reference_dataset_id": "corrective-reference",
        "reference_windows": extracted["windows"][:2],
        "query_windows": [],
        "theta": [0.0] * 16,
    }
    defaults = client.get("/api/network/defaults", params={"node_count": 1})
    session = client.post("/api/network/sessions", json=defaults.json())
    session_id = session.json()["status"]["session_id"]
    intent = EngineeringBenchmarkIntent()

    with pytest.raises(ValidationError):
        TrainingExecutionIntent.model_validate(intent.model_dump(mode="json"))
    response = client.post(
        "/api/training/benchmarks/responsiveness",
        json=intent.model_dump(mode="json"),
    )
    assert response.status_code == 201
    assert started.wait(timeout=1.0)
    repeated = client.post(
        "/api/training/benchmarks/responsiveness",
        json=intent.model_dump(mode="json"),
    )
    assert repeated.status_code == 200
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/quantum/health").status_code == 200
    assert client.get("/api/network/health").status_code == 200
    assert (
        client.post(
            f"/api/network/sessions/{session_id}/step",
            json={"frames": 1},
        ).status_code
        == 200
    )
    assert client.post("/api/quantum/diagnostics").status_code == 429
    assert client.post("/api/quantum/preview", json=preview_payload).status_code == 429
    release.set()
    terminal = _wait_benchmark(benchmark_service)
    assert terminal.state is EngineeringBenchmarkState.COMPLETED
    assert terminal.result is not None
    assert not terminal.result.candidate_artifact_published
    assert client.delete(f"/api/network/sessions/{session_id}").status_code == 204
