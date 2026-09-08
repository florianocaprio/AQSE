from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.network.defaults import (
    default_network_configuration,
    network_preset_configuration,
)
from app.network.models import EventKind
from app.network.session import network_sessions

client = TestClient(app)


@pytest.fixture(autouse=True)
def empty_session_registry():  # type: ignore[no-untyped-def]
    network_sessions.clear()
    yield
    network_sessions.clear()


def _create(node_count: int = 1) -> tuple[str, dict[str, object]]:
    configuration = default_network_configuration(node_count).model_dump(mode="json")
    response = client.post("/api/network/sessions", json=configuration)
    assert response.status_code == 201
    return response.json()["status"]["session_id"], configuration


def test_network_discovery_contract_exposes_presets_one_two_four_and_eight() -> None:
    health = client.get("/api/network/health")
    presets = client.get("/api/network/presets")

    assert health.status_code == 200
    assert health.json()["streaming"] == "sse"
    assert health.json()["session_store"] == "in_memory_bounded"
    assert presets.status_code == 200
    counts = {item["recommended_node_count"] for item in presets.json()["presets"]}
    assert {1, 2, 4, 8}.issubset(counts)
    for node_count in range(1, 9):
        response = client.get("/api/network/defaults", params={"node_count": node_count})
        assert response.status_code == 200
        assert len(response.json()["nodes"]) == node_count

    providers = client.get("/api/network/field-providers")
    assert providers.status_code == 200
    by_id = {item["provider_id"]: item for item in providers.json()["providers"]}
    assert by_id["constant_local_field"]["configured"] is True
    assert by_id["synthetic_spatial_field"]["available"] is True
    assert by_id["world_magnetic_model"]["configured"] is False
    assert by_id["world_magnetic_model"]["available"] is False


def test_vertical_demo_presets_have_valid_reproducible_causal_timelines() -> None:
    eight_node = network_preset_configuration("eight_node_event_demo")
    assert [node.sensor_id for node in eight_node.nodes] == [
        f"S{index}" for index in range(1, 9)
    ]
    assert len(eight_node.environment.dipoles) == 1
    assert eight_node.environment.dipoles[0].source_id == "mobile-demo-dipole"
    assert eight_node.environment.dipoles[0].velocity_m_per_s != (0.0, 0.0, 0.0)

    event_ids = [event.event_id for event in eight_node.events]
    assert len(event_ids) == len(set(event_ids))
    assert all(event_ids)
    valid_sensor_ids = {node.sensor_id for node in eight_node.nodes}
    assert all(
        set(event.target_sensor_ids).issubset(valid_sensor_ids)
        for event in eight_node.events
    )
    by_id = {event.event_id: event for event in eight_node.events}
    assert by_id["demo-s3-drift"].target_sensor_ids == ("S3",)
    assert by_id["demo-s1-s2-shared-offset"].target_sensor_ids == ("S1", "S2")
    assert by_id["demo-s4-dropout"].target_sensor_ids == ("S4",)

    simultaneous = {
        event.event_id
        for event in eight_node.events
        if event.start_time_s <= 13.25 < event.start_time_s + event.duration_s
    }
    assert {
        "demo-simultaneous-physical",
        "demo-simultaneous-s4-fault",
    }.issubset(simultaneous)
    assert by_id["demo-simultaneous-physical"].kind is EventKind.WORLD_FIELD_OFFSET
    assert by_id["demo-simultaneous-s4-fault"].kind is EventKind.DROPOUT
    source = eight_node.environment.dipoles[0]
    position_at_crossing = tuple(
        source.initial_position_m[index] + source.velocity_m_per_s[index] * 8.0
        for index in range(3)
    )
    assert position_at_crossing == (0.0, -0.5, 1.0)

    single_node = network_preset_configuration("single_sensor_ambiguity_demo")
    assert [node.sensor_id for node in single_node.nodes] == ["S1"]
    assert {event.event_id for event in single_node.events} == {
        "single-world-change",
        "single-device-bias",
    }
    assert all(
        not event.target_sensor_ids or event.target_sensor_ids == ("S1",)
        for event in single_node.events
    )
    assert {event.kind for event in single_node.events} == {
        EventKind.WORLD_FIELD_OFFSET,
        EventKind.NODE_BIAS,
    }
    single_by_id = {event.event_id: event for event in single_node.events}
    world = single_by_id["single-world-change"]
    bias = single_by_id["single-device-bias"]
    assert world.start_time_s + world.duration_s < bias.start_time_s
    assert world.field_offset_T == bias.field_offset_T

    for preset_id in ("eight_node_event_demo", "single_sensor_ambiguity_demo"):
        response = client.get(f"/api/network/presets/{preset_id}")
        assert response.status_code == 200
        assert response.json()["session_name"]


def test_session_step_keeps_truth_on_a_separate_api_channel() -> None:
    session_id, _ = _create(2)
    step = client.post(
        f"/api/network/sessions/{session_id}/step",
        json={"frames": 3},
    )
    observations = client.get(f"/api/network/sessions/{session_id}/frames")
    truth = client.get(f"/api/network/sessions/{session_id}/truth/frames")

    assert step.status_code == observations.status_code == truth.status_code == 200
    assert step.json()["from_frame_id"] == 1
    assert step.json()["to_frame_id"] == 3
    assert len(observations.json()["frames"]) == 3
    assert len(truth.json()["frames"]) == 3
    assert "fields" not in observations.text
    assert "active_causes" not in observations.text
    assert "readings" not in truth.text
    timestamp = datetime.fromisoformat(
        observations.json()["frames"][0]["readings"][0]["acquisition_time"]
    )
    assert timestamp.utcoffset() is not None


def test_session_controls_are_idempotent_and_reset_allows_restart() -> None:
    session_id, _ = _create()

    started = client.post(f"/api/network/sessions/{session_id}/start")
    paused = client.post(f"/api/network/sessions/{session_id}/pause")
    paused_again = client.post(f"/api/network/sessions/{session_id}/pause")
    resumed = client.post(f"/api/network/sessions/{session_id}/resume")
    stopped = client.post(f"/api/network/sessions/{session_id}/stop")
    stopped_again = client.post(f"/api/network/sessions/{session_id}/stop")

    assert started.json()["status"]["state"] == "running"
    assert paused.json()["status"]["state"] == "paused"
    assert paused_again.json()["status"]["state"] == "paused"
    assert resumed.json()["status"]["state"] == "running"
    assert stopped.json()["status"]["state"] == "stopped"
    assert stopped_again.json()["status"]["state"] == "stopped"
    assert client.post(f"/api/network/sessions/{session_id}/start").status_code == 409

    reset = client.post(f"/api/network/sessions/{session_id}/reset")
    assert reset.status_code == 200
    assert reset.json()["status"]["state"] == "created"
    assert reset.json()["status"]["latest_frame_id"] == 0


def test_replay_reproduces_numeric_frames_and_bounded_buffer_reports_gap() -> None:
    configuration = default_network_configuration(1).model_copy(
        update={"sampling_rate_Hz": 10.0, "ui_refresh_rate_Hz": 5.0, "buffer_duration_s": 1.0}
    )
    response = client.post(
        "/api/network/sessions",
        json=configuration.model_dump(mode="json"),
    )
    session_id = response.json()["status"]["session_id"]
    first = client.post(
        f"/api/network/sessions/{session_id}/step",
        json={"frames": 15},
    )
    replay = client.post(f"/api/network/sessions/{session_id}/replay")
    after = client.get(
        f"/api/network/sessions/{session_id}/frames",
        params={"after_frame_id": 0, "limit": 20},
    )

    assert first.status_code == replay.status_code == after.status_code == 200
    assert replay.json()["status"]["state"] == "paused"
    assert after.json()["gap_detected"] is True
    assert after.json()["from_frame_id"] == 6
    assert after.json()["status"]["buffer"]["size"] == 10
    assert after.json()["status"]["buffer"]["overwritten_frames"] == 5
    first_tail = first.json()["frames"][-10:]
    assert after.json()["frames"] == first_tail


def test_sse_reconnect_cursor_reports_a_bounded_buffer_gap() -> None:
    configuration = default_network_configuration(1).model_copy(
        update={"sampling_rate_Hz": 10.0, "ui_refresh_rate_Hz": 5.0, "buffer_duration_s": 1.0}
    )
    created = client.post(
        "/api/network/sessions",
        json=configuration.model_dump(mode="json"),
    )
    session_id = created.json()["status"]["session_id"]
    client.post(f"/api/network/sessions/{session_id}/step", json={"frames": 15})
    client.post(f"/api/network/sessions/{session_id}/stop")

    with client.stream(
        "GET",
        f"/api/network/sessions/{session_id}/stream",
        params={"after_frame_id": 0},
    ) as response:
        body = "\n".join(response.iter_lines())

    assert response.status_code == 200
    assert "event: observations" in body
    assert '"gap_detected":true' in body
    assert "event: session_status" in body


def test_sse_last_event_id_overrides_the_bootstrap_query_on_reconnect() -> None:
    configuration = default_network_configuration(1).model_copy(
        update={"sampling_rate_Hz": 10.0, "ui_refresh_rate_Hz": 5.0, "buffer_duration_s": 2.0}
    )
    created = client.post(
        "/api/network/sessions",
        json=configuration.model_dump(mode="json"),
    )
    session_id = created.json()["status"]["session_id"]
    client.post(f"/api/network/sessions/{session_id}/step", json={"frames": 15})
    client.post(f"/api/network/sessions/{session_id}/stop")

    with client.stream(
        "GET",
        f"/api/network/sessions/{session_id}/stream",
        params={"after_frame_id": 0},
        headers={"Last-Event-ID": "12"},
    ) as response:
        body = "\n".join(response.iter_lines())

    assert response.status_code == 200
    assert '"from_frame_id":13' in body
    assert '"to_frame_id":15' in body
    assert '"gap_detected":false' in body


def test_event_validation_and_simulated_dropout_return_data_not_http_500() -> None:
    session_id, _ = _create()
    event = {
        "event_id": "manual-dropout",
        "kind": "dropout",
        "start_time_s": 0.0,
        "duration_s": 1.0,
        "target_sensor_ids": ["S1"],
    }
    scheduled = client.post(
        f"/api/network/sessions/{session_id}/events",
        json=event,
    )
    stepped = client.post(
        f"/api/network/sessions/{session_id}/step",
        json={"frames": 1},
    )

    assert scheduled.status_code == 201
    assert scheduled.json()["effective_frame_id"] == 1
    reading = stepped.json()["frames"][0]["readings"][0]
    assert reading["components_T"] is None
    assert reading["valid"] is False
    assert "signal_absent" in reading["quality_flags"]
    unknown = {**event, "event_id": "unknown", "target_sensor_ids": ["S9"]}
    assert (
        client.post(f"/api/network/sessions/{session_id}/events", json=unknown).status_code
        == 409
    )


def test_event_effective_boundary_and_configuration_versions_survive_replay() -> None:
    session_id, _ = _create()
    first = client.post(
        f"/api/network/sessions/{session_id}/step",
        json={"frames": 3},
    )
    event = {
        "event_id": "future-bias",
        "kind": "node_bias",
        "start_time_s": 0.03,
        "duration_s": 1.0,
        "target_sensor_ids": ["S1"],
        "field_offset_T": [1.0e-9, 0.0, 0.0],
    }
    scheduled = client.post(
        f"/api/network/sessions/{session_id}/events",
        json=event,
    )
    second = client.post(
        f"/api/network/sessions/{session_id}/step",
        json={"frames": 3},
    )
    past = client.post(
        f"/api/network/sessions/{session_id}/events",
        json={**event, "event_id": "past", "start_time_s": 0.0},
    )

    assert first.status_code == second.status_code == 200
    assert scheduled.status_code == 201
    assert scheduled.json()["effective_frame_id"] == 4
    assert [frame["configuration_version"] for frame in first.json()["frames"]] == [1, 1, 1]
    assert [frame["configuration_version"] for frame in second.json()["frames"]] == [2, 2, 2]
    assert past.status_code == 409

    before_replay = [*first.json()["frames"], *second.json()["frames"]]
    assert client.post(f"/api/network/sessions/{session_id}/replay").status_code == 200
    replayed = client.get(
        f"/api/network/sessions/{session_id}/frames",
        params={"after_frame_id": 0, "limit": 10},
    ).json()["frames"]
    assert replayed == before_replay


def test_vector_finite_api_returns_synchronized_observation_and_truth_arrays() -> None:
    scenarios = client.get("/api/sensors/vector-magnetometer/scenarios")
    assert scenarios.status_code == 200
    assert len(scenarios.json()["scenarios"]) == 6

    configuration = client.get(
        "/api/sensors/vector-magnetometer/defaults",
        params={"scenario": "combined_stress"},
    ).json()
    configuration["duration_s"] = 0.2
    simulated = client.post(
        "/api/sensors/vector-magnetometer/simulate",
        json=configuration,
    )

    assert simulated.status_code == 200
    payload = simulated.json()
    sample_count = payload["sample_count"]
    for name in (
        "time_s",
        "position_m",
        "orientation_world_to_sensor_wxyz",
        "temperature_K",
        "measured_field_T",
        "saturation_mask",
        "valid",
        "quality_flags",
    ):
        assert len(payload["measurement"][name]) == sample_count
    for name in (
        "time_s",
        "earth_field_world_T",
        "environment_field_world_T",
        "anomaly_field_world_T",
        "periodic_field_world_T",
        "ideal_field_sensor_T",
        "active_causes",
    ):
        assert len(payload["generator_truth"][name]) == sample_count
    json.loads(simulated.text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def test_api_rejects_sample_cap_extra_fields_bandwidth_and_invalid_matrices() -> None:
    finite = client.get("/api/sensors/vector-magnetometer/defaults").json()
    finite["duration_s"] = 101.0
    finite["sampling_rate_Hz"] = 100.0
    assert client.post("/api/sensors/vector-magnetometer/simulate", json=finite).status_code == 422

    configuration = default_network_configuration(1).model_dump(mode="json")
    configuration["unexpected"] = True
    assert client.post("/api/network/sessions", json=configuration).status_code == 422
    configuration.pop("unexpected")
    configuration["nodes"][0]["errors"]["bandwidth_Hz"] = 50.0
    assert client.post("/api/network/sessions", json=configuration).status_code == 422
    configuration["nodes"][0]["errors"]["bandwidth_Hz"] = None
    configuration["nodes"][0]["errors"]["soft_iron_matrix"] = [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert client.post("/api/network/sessions", json=configuration).status_code == 422


def test_observability_api_is_finite_and_does_not_claim_global_uniqueness() -> None:
    configuration = default_network_configuration(8).model_dump(mode="json")
    response = client.post(
        "/api/network/observability",
        json={"configuration": configuration},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["jacobian_shape"] == [24, 6]
    assert payload["numerical_rank"] == 6
    assert "does not guarantee" in payload["interpretation"]
    json.dumps(payload, allow_nan=False)


def test_missing_session_and_delete_have_clear_http_contracts() -> None:
    assert client.get("/api/network/sessions/missing").status_code == 404
    session_id, _ = _create()
    assert client.delete(f"/api/network/sessions/{session_id}").status_code == 204
    assert client.delete(f"/api/network/sessions/{session_id}").status_code == 404
