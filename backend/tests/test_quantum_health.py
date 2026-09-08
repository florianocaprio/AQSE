from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import perf_counter

from fastapi.testclient import TestClient

import app.api.quantum_health as quantum_health_api
from app.main import app
from app.quantum.adapter import (
    QuantumInfrastructureStatus,
    TQK8Adapter,
    run_quantum_infrastructure_smoke_test,
)
from app.quantum.user_pipeline.sampler_qng import circuit_budget

client = TestClient(app)


def test_quantum_health_endpoint_is_lightweight_readiness() -> None:
    response = client.get("/api/quantum/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "engine": "AQSE TQK8",
        "adapter": "ready",
        "qiskit": "installed",
        "qiskit_version": payload["qiskit_version"],
        "numpy_reference": "available_not_executed",
        "qubits": 8,
        "features": 8,
        "trainable_parameters": 16,
        "simulation": "qiskit_statevector",
        "circuit_metadata": "available",
        "entangling_edges": 7,
    }
    assert payload["qiskit_version"]


def test_quantum_health_does_not_execute_statevector_smoke_test(monkeypatch) -> None:
    def fail_if_executed() -> QuantumInfrastructureStatus:
        raise AssertionError("GET quantum health must not execute statevectors")

    monkeypatch.setattr(
        quantum_health_api,
        "run_quantum_infrastructure_smoke_test",
        fail_if_executed,
    )

    response = client.get("/api/quantum/health")

    assert response.status_code == 200
    assert response.json()["numpy_reference"] == "available_not_executed"


def test_repeated_service_and_quantum_health_requests_remain_responsive() -> None:
    assert quantum_health_api.diagnostics_slot.acquire(blocking=False)
    try:
        started = perf_counter()
        for _ in range(20):
            assert client.get("/api/health").status_code == 200
            assert client.get("/api/quantum/health").status_code == 200
        elapsed = perf_counter() - started
    finally:
        quantum_health_api.diagnostics_slot.release()

    assert elapsed < 2.0


def test_quantum_diagnostics_endpoint_runs_real_statevector_smoke_test() -> None:
    response = client.post("/api/quantum/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["qiskit"] == "ready"
    assert payload["numpy_reference"] == "ready"
    assert payload["state_comparison"] == "passed"
    assert payload["execution_duration_ms"] > 0.0


def test_overlapping_quantum_diagnostics_return_busy(monkeypatch) -> None:
    started = Event()
    release = Event()

    def blocked_diagnostic() -> QuantumInfrastructureStatus:
        started.set()
        if not release.wait(timeout=5.0):
            raise TimeoutError("test did not release diagnostics")
        return QuantumInfrastructureStatus(
            engine="AQSE TQK8",
            qiskit="ready",
            numpy_reference="ready",
            qubits=8,
            features=8,
            trainable_parameters=16,
            simulation="statevector",
        )

    monkeypatch.setattr(
        quantum_health_api,
        "run_quantum_infrastructure_smoke_test",
        blocked_diagnostic,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        first_request = executor.submit(client.post, "/api/quantum/diagnostics")
        assert started.wait(timeout=2.0)
        busy_response = client.post("/api/quantum/diagnostics")
        release.set()
        first_response = first_request.result(timeout=5.0)

    assert busy_response.status_code == 429
    assert busy_response.json() == {
        "detail": "Quantum diagnostics are already running; retry after completion"
    }
    assert first_response.status_code == 200


def test_both_exact_state_engines_are_available() -> None:
    assert TQK8Adapter("numpy").metadata.backend_type == "numpy_statevector"
    assert TQK8Adapter("qiskit").metadata.backend_type == "qiskit_statevector"


def test_sampler_module_is_importable_from_application_package() -> None:
    assert circuit_budget(8, 4, "diagonal")["total"] == 1884


def test_smoke_test_makes_no_network_request(monkeypatch) -> None:
    def fail_on_connect(*args, **kwargs) -> None:
        raise AssertionError("The infrastructure smoke test must remain local")

    monkeypatch.setattr(socket.socket, "connect", fail_on_connect)

    result = run_quantum_infrastructure_smoke_test()

    assert result.qiskit == "ready"
