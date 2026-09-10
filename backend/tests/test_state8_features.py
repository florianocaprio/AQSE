from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from math import pi, sin, sqrt
from typing import get_type_hints

import numpy as np
import pytest

from app.features.state8 import (
    build_state8_reference,
    extract_latest_state8_window,
    extract_state8_features,
    require_state8_profile,
    state8_profile,
    state8_profile_fingerprint,
    validate_state8_reference,
)
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
)
from app.network.models import (
    MeasurementMode,
    ObservationFrame,
    QualityFlag,
    SensorReading,
)
from app.network.physics import quaternion_world_to_sensor_matrix

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
IDENTITY = (1.0, 0.0, 0.0, 0.0)
QUARTER_TURN_Z = (sqrt(0.5), 0.0, 0.0, sqrt(0.5))


def _default_anomaly(sensor_id: str, elapsed_s: float) -> float:
    phase = {"A": 0.0, "B": 0.15, "C": -0.1}.get(sensor_id, 0.0)
    amplitude = {"A": 3.0, "B": 2.0, "C": 4.0}.get(sensor_id, 1.0)
    return amplitude * sin(2.0 * pi * elapsed_s + phase) + 0.2 * sin(
        2.0 * pi * 10.0 * elapsed_s
    )


def _frames(
    *,
    node_ids: tuple[str, ...] = ("A",),
    count: int = 1_200,
    anomaly: Callable[[str, float], float] = _default_anomaly,
    reverse_readings: bool = False,
    orientations: dict[str, tuple[float, float, float, float]] | None = None,
) -> tuple[ObservationFrame, ...]:
    output: list[ObservationFrame] = []
    orientations = orientations or {}
    for index in range(count):
        sim_time_s = index / 100.0
        readings: list[SensorReading] = []
        for node_offset, sensor_id in enumerate(node_ids):
            reference_north_nt = 20_000.0 + node_offset * 100.0
            elapsed_s = sim_time_s - 8.0
            anomaly_nt = anomaly(sensor_id, elapsed_s) if elapsed_s >= 0.0 else 0.0
            world_field = np.asarray(
                ((reference_north_nt + anomaly_nt) * 1.0e-9, 0.0, 0.0)
            )
            quaternion = orientations.get(sensor_id, IDENTITY)
            sensor_field = quaternion_world_to_sensor_matrix(quaternion) @ world_field
            timestamp = EPOCH + timedelta(seconds=sim_time_s)
            readings.append(
                SensorReading(
                    sensor_id=sensor_id,
                    sequence_id=index + 1,
                    acquisition_time=timestamp,
                    arrival_time=timestamp,
                    measurement_mode=MeasurementMode.VECTOR,
                    components_T=tuple(float(value) for value in sensor_field),
                    saturation_mask=(False, False, False),
                    valid=True,
                    observed_temperature_K=(
                        290.0 + node_offset + (2.0 if elapsed_s >= 0.0 else 0.0)
                    ),
                    position_m=(float(node_offset), 0.0, 0.0),
                    orientation_world_to_sensor_wxyz=quaternion,
                    calibration_version="calibration-v1",
                )
            )
        if reverse_readings:
            readings.reverse()
        output.append(
            ObservationFrame(
                session_id="state8-test-session",
                configuration_version=1,
                frame_id=index + 1,
                sim_time_s=sim_time_s,
                readings=tuple(readings),
            )
        )
    return tuple(output)


def _replace_reading(
    frames: tuple[ObservationFrame, ...],
    index: int,
    sensor_id: str,
    **updates: object,
) -> tuple[ObservationFrame, ...]:
    changed = list(frames)
    frame = frames[index]
    readings = tuple(
        reading.model_copy(update=updates) if reading.sensor_id == sensor_id else reading
        for reading in frame.readings
    )
    changed[index] = frame.model_copy(update={"readings": readings})
    return tuple(changed)


def _independent_band_ratio(values: np.ndarray) -> float:
    time_s = np.arange(len(values), dtype=np.float64) / 100.0
    slope, intercept = np.polyfit(time_s, values, 1)
    windowed = (values - (slope * time_s + intercept)) * np.hanning(len(values))
    frequencies = np.fft.rfftfreq(len(values), 0.01)
    density = np.abs(np.fft.rfft(windowed)) ** 2 / (
        100.0 * np.sum(np.hanning(len(values)) ** 2)
    )
    density[1:-1] *= 2.0
    lower = np.sum(density[(frequencies >= 0.25) & (frequencies < 2.0)]) * 0.25
    upper = np.sum(density[(frequencies >= 2.0) & (frequencies <= 20.0)]) * 0.25
    return float(10.0 * np.log10(max(lower, 1.0e-12) / max(upper, 1.0e-12)))


def test_profiles_have_distinct_frozen_semantics_and_fingerprints() -> None:
    local = state8_profile(LOCAL_STATE8_PROFILE_ID)
    network = state8_profile(NETWORK_STATE8_PROFILE_ID)

    assert local.scope == "local"
    assert network.scope == "network"
    assert local.feature_names[5:7] != network.feature_names[5:7]
    assert state8_profile_fingerprint(local) != state8_profile_fingerprint(network)
    with pytest.raises(ValueError, match="incompatible"):
        require_state8_profile(local.model_copy(update={"scope": "network"}), local.profile_id)


def test_observed_reference_recovers_world_north_with_declared_pose() -> None:
    frames = _frames(orientations={"A": QUARTER_TURN_Z})

    reference = build_state8_reference(frames)

    assert reference.node_order == ("A",)
    assert reference.nodes[0].valid
    assert reference.nodes[0].north_reference_nt == pytest.approx(20_000.0)
    assert reference.nodes[0].temperature_reference_k == pytest.approx(290.0)
    sensor_components = frames[0].readings[0].components_T
    assert sensor_components is not None
    assert sensor_components[0] == pytest.approx(0.0, abs=1.0e-20)
    assert sensor_components[1] == pytest.approx(20_000.0e-9)
    validate_state8_reference(reference)
    tampered = reference.model_copy(
        update={
            "nodes": (
                reference.nodes[0].model_copy(update={"north_reference_nt": 0.0}),
            )
        }
    )
    with pytest.raises(ValueError, match="digest"):
        validate_state8_reference(tampered)


def test_local_state8_formulas_and_units_match_the_frozen_definition() -> None:
    def signal(_: str, elapsed_s: float) -> float:
        return (
            3.0
            + 0.5 * elapsed_s
            + 2.0 * sin(2.0 * pi * elapsed_s)
            + 0.25 * sin(2.0 * pi * 10.0 * elapsed_s)
        )

    frames = _frames(anomaly=signal)
    reference = build_state8_reference(frames)
    response = extract_state8_features(
        frames,
        reference=reference,
        profile_id=LOCAL_STATE8_PROFILE_ID,
    )
    record = response.records[0]
    elapsed = np.arange(400, dtype=np.float64) / 100.0
    anomaly = np.asarray([signal("A", value) for value in elapsed])
    median = np.median(anomaly)
    expected = (
        np.mean(anomaly),
        1.4826 * np.median(np.abs(anomaly - median)),
        np.polyfit(elapsed, anomaly, 1)[0],
        _independent_band_ratio(anomaly),
        2.0,
        np.sqrt(np.mean(np.diff(anomaly) ** 2)),
        np.corrcoef(anomaly[:-1], anomaly[1:])[0, 1],
        1.0,
    )

    assert record.quality.valid_for_quantum
    assert response.profile.feature_units == (
        "nT",
        "nT",
        "nT/s",
        "dB",
        "K",
        "nT",
        "dimensionless",
        "dimensionless",
    )
    np.testing.assert_allclose(record.values, expected, rtol=1.0e-10, atol=1.0e-10)


def test_first_window_is_causal_and_ignores_later_observations() -> None:
    frames = _frames(count=1_300)
    reference = build_state8_reference(frames)
    baseline = extract_state8_features(
        frames,
        reference=reference,
        profile_id=LOCAL_STATE8_PROFILE_ID,
    )
    changed = _replace_reading(
        frames,
        1_299,
        "A",
        components_T=(1.0e-3, 0.0, 0.0),
    )
    perturbed = extract_state8_features(
        changed,
        reference=reference,
        profile_id=LOCAL_STATE8_PROFILE_ID,
    )

    assert len(baseline.records) == len(perturbed.records) == 2
    assert baseline.records[0] == perturbed.records[0]
    assert baseline.records[1].values != perturbed.records[1].values
    assert baseline.records[0].source_frame_ids[-1] == 1_200


def test_latest_window_matches_batch_without_requiring_historical_windows() -> None:
    frames = _frames(count=2_000)
    reference = build_state8_reference(frames[:800])
    batch = extract_state8_features(
        frames,
        reference=reference,
        profile_id=LOCAL_STATE8_PROFILE_ID,
    )
    latest = extract_latest_state8_window(
        frames[-400:],
        reference=reference,
        profile_id=LOCAL_STATE8_PROFILE_ID,
    )

    assert len(batch.records) == 9
    assert latest.records == (batch.records[-1],)
    assert latest.records[0].start_time_s == 16.0


def test_network_features_use_sorted_leave_one_out_peer_context() -> None:
    frames = _frames(node_ids=("C", "A", "B"), reverse_readings=True)
    reference = build_state8_reference(frames)
    response = extract_state8_features(
        frames,
        reference=reference,
        profile_id=NETWORK_STATE8_PROFILE_ID,
    )

    assert reference.node_order == ("A", "B", "C")
    assert tuple(record.sensor_id for record in response.records) == ("A", "B", "C")
    focal = next(record for record in response.records if record.sensor_id == "A")
    assert focal.peer_sensor_ids == ("B", "C")
    time_s = np.arange(400, dtype=np.float64) / 100.0
    arrays = {
        node: np.asarray([_default_anomaly(node, value) for value in time_s])
        for node in ("A", "B", "C")
    }
    peer_median = np.median(np.vstack((arrays["B"], arrays["C"])), axis=0)
    expected_residual_rms = np.sqrt(np.mean((arrays["A"] - peer_median) ** 2))
    expected_correlation = np.mean(
        (
            np.corrcoef(arrays["A"], arrays["B"])[0, 1],
            np.corrcoef(arrays["A"], arrays["C"])[0, 1],
        )
    )

    assert focal.quality.valid_for_quantum
    assert focal.values[5] == pytest.approx(expected_residual_rms)
    assert focal.values[6] == pytest.approx(expected_correlation)


@pytest.mark.parametrize("node_count", (1, 2, 3))
def test_network_context_requires_three_nodes_while_local_supports_one(
    node_count: int,
) -> None:
    node_ids = tuple(chr(ord("A") + index) for index in range(node_count))
    frames = _frames(node_ids=node_ids)
    reference = build_state8_reference(frames)
    network = extract_state8_features(
        frames,
        reference=reference,
        profile_id=NETWORK_STATE8_PROFILE_ID,
        sensor_ids=("A",),
    ).records[0]

    assert network.quality.valid_for_quantum is (node_count >= 3)
    if node_count < 3:
        assert network.values[5:7] == (None, None)
        assert "insufficient_network_context" in network.quality.flags

    if node_count == 1:
        local = extract_state8_features(
            frames,
            reference=reference,
            profile_id=LOCAL_STATE8_PROFILE_ID,
        ).records[0]
        assert local.quality.valid_for_quantum


def test_network_context_excludes_invalid_peer_when_two_valid_peers_remain() -> None:
    frames = _frames(node_ids=("A", "B", "C", "D"))
    reference = build_state8_reference(frames)
    changed = _replace_reading(
        frames,
        900,
        "B",
        quality_flags=(QualityFlag.STUCK,),
    )

    record = extract_state8_features(
        changed,
        reference=reference,
        profile_id=NETWORK_STATE8_PROFILE_ID,
        sensor_ids=("A",),
    ).records[0]

    assert record.quality.valid_for_quantum
    assert record.peer_sensor_ids == ("C", "D")
    assert all(value is not None for value in record.values)


def test_network_context_rejects_window_with_fewer_than_two_valid_peers() -> None:
    frames = _frames(node_ids=("A", "B", "C"))
    reference = build_state8_reference(frames)
    changed = _replace_reading(
        frames,
        900,
        "B",
        quality_flags=(QualityFlag.CLOCK_ERROR,),
    )

    record = extract_state8_features(
        changed,
        reference=reference,
        profile_id=NETWORK_STATE8_PROFILE_ID,
        sensor_ids=("A",),
    ).records[0]

    assert not record.quality.valid_for_quantum
    assert record.peer_sensor_ids == ("C",)
    assert record.values[5:7] == (None, None)
    assert "insufficient_valid_peer_context" in record.quality.flags


@pytest.fixture(scope="module")
def three_node_frames() -> tuple[ObservationFrame, ...]:
    return _frames(node_ids=("A", "B", "C"))


@pytest.mark.parametrize(
    ("updates", "expected_flag"),
    (
        ({"saturation_mask": (True, False, False)}, "clipped"),
        ({"quality_flags": (QualityFlag.STUCK,)}, "stuck"),
        ({"quality_flags": (QualityFlag.CLOCK_ERROR,)}, "clock_error"),
        ({"calibration_version": "calibration-v2"}, "calibration_version_changed"),
    ),
)
def test_observable_quality_failures_produce_nullable_features_without_imputation(
    three_node_frames: tuple[ObservationFrame, ...],
    updates: dict[str, object],
    expected_flag: str,
) -> None:
    reference = build_state8_reference(three_node_frames)
    changed = _replace_reading(three_node_frames, 900, "A", **updates)
    record = extract_state8_features(
        changed,
        reference=reference,
        profile_id=LOCAL_STATE8_PROFILE_ID,
        sensor_ids=("A",),
    ).records[0]

    assert not record.quality.valid_for_quantum
    assert record.values[:7] == (None,) * 7
    assert record.values[7] == 1.0
    assert expected_flag in record.quality.flags


def test_missing_and_nonuniform_samples_invalidate_without_zero_semantics(
    three_node_frames: tuple[ObservationFrame, ...],
) -> None:
    reference = build_state8_reference(three_node_frames)
    missing = list(three_node_frames)
    missing_frame = missing[900]
    missing[900] = missing_frame.model_copy(
        update={
            "readings": tuple(
                reading
                for reading in missing_frame.readings
                if reading.sensor_id != "A"
            )
        }
    )
    missing_record = extract_state8_features(
        tuple(missing),
        reference=reference,
        profile_id=LOCAL_STATE8_PROFILE_ID,
        sensor_ids=("A",),
    ).records[0]
    nonuniform = list(three_node_frames)
    nonuniform[900] = nonuniform[900].model_copy(
        update={"sim_time_s": nonuniform[900].sim_time_s + 0.001}
    )
    nonuniform_record = extract_state8_features(
        tuple(nonuniform),
        reference=reference,
        profile_id=LOCAL_STATE8_PROFILE_ID,
        sensor_ids=("A",),
    ).records[0]

    assert missing_record.values[:7] == (None,) * 7
    assert missing_record.values[7] == pytest.approx(399.0 / 400.0)
    assert "missing_reading" in missing_record.quality.flags
    assert nonuniform_record.values[:7] == (None,) * 7
    assert "nonuniform_sampling" in nonuniform_record.quality.flags


def test_invalid_reference_and_duplicate_selected_ids_are_explicit() -> None:
    frames = _replace_reading(
        _frames(),
        100,
        "A",
        quality_flags=(QualityFlag.CLIPPED,),
    )
    reference = build_state8_reference(frames)

    assert not reference.nodes[0].valid
    assert reference.nodes[0].north_reference_nt is None
    assert "clipped" in reference.nodes[0].flags
    record = extract_state8_features(
        frames,
        reference=reference,
        profile_id=LOCAL_STATE8_PROFILE_ID,
    ).records[0]
    assert not record.quality.valid_for_quantum
    assert "reference_invalid" in record.quality.flags
    with pytest.raises(ValueError, match="unique"):
        extract_state8_features(
            frames,
            reference=reference,
            profile_id=LOCAL_STATE8_PROFILE_ID,
            sensor_ids=("A", "A"),
        )


def test_public_extraction_boundary_accepts_observations_not_truth() -> None:
    reference_hints = get_type_hints(build_state8_reference)
    extraction_hints = get_type_hints(extract_state8_features)

    assert reference_hints["frames"] == Sequence[ObservationFrame]
    assert extraction_hints["frames"] == Sequence[ObservationFrame]
    assert "TruthFrame" not in str(inspect.signature(build_state8_reference))
    assert "TruthFrame" not in str(inspect.signature(extract_state8_features))

    frames = _frames()
    reference_a = build_state8_reference(frames)
    reference_b = build_state8_reference(frames)
    assert reference_a == reference_b
    assert extract_state8_features(
        frames,
        reference=reference_a,
        profile_id=LOCAL_STATE8_PROFILE_ID,
    ) == extract_state8_features(
        frames,
        reference=reference_b,
        profile_id=LOCAL_STATE8_PROFILE_ID,
    )
