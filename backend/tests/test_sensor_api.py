import math
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.sensors.models import MAX_ACQUISITION_SAMPLES

client = TestClient(app)


def test_sensor_catalog_lists_quantum_magnetometer() -> None:
    response = client.get("/api/sensors")

    assert response.status_code == 200
    assert response.json() == {
        "sensors": [
            {
                "sensor_type": "quantum_magnetometer",
                "display_name": "Quantum Magnetometer",
                "simulation": True,
                "physical_unit": "nT",
            }
        ]
    }


def test_defaults_are_small_enough_for_local_visualization() -> None:
    response = client.get("/api/sensors/magnetometer/defaults")

    assert response.status_code == 200
    defaults = response.json()
    assert defaults["duration"] == 2.0
    assert defaults["sampling_rate"] == 200.0
    assert defaults["duration"] * defaults["sampling_rate"] == 400


def test_simulation_returns_acquisition_spectrum_and_features() -> None:
    response = client.post(
        "/api/sensors/magnetometer/simulate",
        json={
            "duration": 1.0,
            "sampling_rate": 100.0,
            "frequency": 5.0,
            "amplitude": 8.0,
            "noise_std": 0.0,
            "random_seed": 9,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["acquisition"]["sample_count"] == 100
    assert len(payload["acquisition"]["time"]) == 100
    assert len(payload["acquisition"]["signal"]) == 100
    assert payload["acquisition"]["physical_unit"] == "nT"
    timestamp = datetime.fromisoformat(
        payload["acquisition"]["timestamp"].replace("Z", "+00:00")
    )
    assert timestamp.utcoffset() == timedelta(0)
    assert len(payload["spectrum"]["frequencies"]) == 51
    assert len(payload["spectrum"]["power_spectral_density"]) == 51
    assert payload["features"]["names"] == [
        "amplitude",
        "phase",
        "frequency",
        "variance",
        "drift",
        "snr",
        "spectral_peak",
        "temperature",
    ]
    assert len(payload["features"]["values"]) == 8
    assert all(math.isfinite(value) for value in payload["acquisition"]["signal"])
    assert all(
        math.isfinite(value)
        for value in payload["spectrum"]["power_spectral_density"]
    )
    assert all(math.isfinite(value) for value in payload["features"]["values"])


def test_api_rejects_frequency_at_or_above_nyquist() -> None:
    response = client.post(
        "/api/sensors/magnetometer/simulate",
        json={"sampling_rate": 100.0, "frequency": 50.0},
    )

    assert response.status_code == 422
    assert "Nyquist" in response.text


def test_api_rejects_anomaly_outside_acquisition() -> None:
    response = client.post(
        "/api/sensors/magnetometer/simulate",
        json={
            "duration": 2.0,
            "anomaly_enabled": True,
            "anomaly_time": 2.0,
        },
    )

    assert response.status_code == 422
    assert "anomaly_time" in response.text


def test_api_rejects_excessive_sample_count_and_unknown_fields() -> None:
    too_many = client.post(
        "/api/sensors/magnetometer/simulate",
        json={
            "duration": 3.0,
            "sampling_rate": MAX_ACQUISITION_SAMPLES,
            "frequency": 1.0,
        },
    )
    unknown = client.post(
        "/api/sensors/magnetometer/simulate",
        json={"frequency": 8.0, "unsupported_parameter": 1},
    )

    assert too_many.status_code == 422
    assert f"at most {MAX_ACQUISITION_SAMPLES}" in too_many.text
    assert unknown.status_code == 422


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"amplitude": -1.0},
        {"noise_std": -1.0},
        {"temperature": 0.0},
        {"random_seed": -1},
        {"random_seed": 2**32},
        {"background_field": 1.0e308},
        {"phase": 1.0e308},
        {"duration": 0.01, "sampling_rate": 100.0, "frequency": 1.0},
    ],
)
def test_api_rejects_invalid_ranges(invalid_field: dict[str, float | int]) -> None:
    response = client.post(
        "/api/sensors/magnetometer/simulate",
        json=invalid_field,
    )

    assert response.status_code == 422
