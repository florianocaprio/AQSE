from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.features import router as features_router
from app.api.quantum_preview import router as quantum_preview_router
from app.api.workbench import router as workbench_router

api_app = FastAPI()
api_app.include_router(features_router, prefix="/api")
api_app.include_router(quantum_preview_router, prefix="/api")
api_app.include_router(workbench_router, prefix="/api")
client = TestClient(api_app)


def measured_payload() -> dict[str, object]:
    sampling_rate = 100.0
    time = np.arange(400, dtype=float) / sampling_rate
    signal = 25.0 + 8.0 * np.sin(2.0 * np.pi * 8.0 * time)
    field = np.column_stack((signal, np.full_like(signal, 2.0), np.zeros_like(signal)))
    return {
        "series": {
            "acquisition_id": "api-acquisition",
            "sensor_id": "api-sensor",
            "sampling_rate_hz": sampling_rate,
            "time_s": time.tolist(),
            "measured_field": field.tolist(),
            "field_unit": "nT",
            "temperature_k": (293.15 + 0.2 * time).tolist(),
            "saturation_mask": np.zeros((len(time), 3), dtype=bool).tolist(),
        },
        "channel": "x",
        "window": {"duration_s": 1.0, "overlap_fraction": 0.5},
    }


def test_workbench_capabilities_are_honest_about_pending_science() -> None:
    response = client.get("/api/workbench/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["quantum_preview"]["status"] == "implemented"
    assert payload["qng_training"]["status"] == "available_not_connected"
    assert payload["local_embedding_afse"]["status"] == "architecture_defined"
    assert payload["neural_model"]["status"] == "not_implemented"
    assert payload["physical_qpu"]["status"] == "not_implemented"


def test_feature_and_quantum_preview_api_flow() -> None:
    feature_response = client.post(
        "/api/features/vector-magnetometer/extract",
        json=measured_payload(),
    )
    assert feature_response.status_code == 200
    extracted = feature_response.json()
    assert extracted["valid_window_count"] >= 2

    preview_response = client.post(
        "/api/quantum/preview",
        json={
            "mode": "self_reference",
            "backend": "numpy",
            "feature_profile": extracted["profile"],
            "reference_dataset_id": "api-reference",
            "reference_windows": extracted["windows"][:2],
            "query_windows": [],
            "theta": [0.0] * 16,
        },
    )
    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert len(preview["reference_kernel"]) == 2
    assert preview["executed_theta"] == [0.0] * 16
    assert preview["scientific_scope"].endswith("no QNG training")


def test_circuit_api_reports_the_actual_tqk8_contract() -> None:
    response = client.get("/api/quantum/circuit")

    assert response.status_code == 200
    payload = response.json()
    assert payload["qubits"] == 8
    assert payload["trainable_parameters"] == 16
    assert payload["feature_uploads_per_feature"] == 2
    assert len(payload["cz_edges"]) == 7


def test_feature_api_rejects_truth_leakage() -> None:
    payload = measured_payload()
    payload["series"]["field_true_world_T"] = [[0.0, 0.0, 0.0]] * 400

    response = client.post(
        "/api/features/vector-magnetometer/extract",
        json=payload,
    )

    assert response.status_code == 422
