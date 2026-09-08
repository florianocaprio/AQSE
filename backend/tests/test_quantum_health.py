import socket

from fastapi.testclient import TestClient

from app.main import app
from app.quantum.adapter import TQK8Adapter, run_quantum_infrastructure_smoke_test
from app.quantum.user_pipeline.sampler_qng import circuit_budget

client = TestClient(app)


def test_quantum_health_endpoint() -> None:
    response = client.get("/api/quantum/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "engine": "AQSE TQK8",
        "qiskit": "ready",
        "numpy_reference": "ready",
        "qubits": 8,
        "features": 8,
        "trainable_parameters": 16,
        "simulation": "statevector",
    }


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
