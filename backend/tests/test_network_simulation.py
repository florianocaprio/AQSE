from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.network.defaults import default_network_configuration, default_vector_configuration
from app.network.models import (
    DipoleSourceConfiguration,
    EnvironmentConfiguration,
    EventKind,
    GaussianAnomalyConfiguration,
    MeasurementMode,
    MotionKind,
    NetworkEventConfiguration,
    NetworkSessionConfiguration,
    NodeErrorConfiguration,
    NodeMotionConfiguration,
    QualityFlag,
    SensorNodeConfiguration,
    TemperatureDriverConfiguration,
    VectorScenario,
)
from app.network.simulation import NetworkSimulator

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _generate(
    configuration: NetworkSessionConfiguration,
    count: int,
    events: tuple[NetworkEventConfiguration, ...] = (),
):  # type: ignore[no-untyped-def]
    simulator = NetworkSimulator(configuration)
    return tuple(
        simulator.generate(
            "test-session",
            index + 1,
            index / configuration.sampling_rate_Hz,
            EPOCH,
            events,
        )
        for index in range(count)
    )


def _configuration(
    *nodes: SensorNodeConfiguration,
    sampling_rate_Hz: float = 100.0,
    environment: EnvironmentConfiguration | None = None,
    seed: int = 7,
) -> NetworkSessionConfiguration:
    return NetworkSessionConfiguration(
        random_seed=seed,
        sampling_rate_Hz=sampling_rate_Hz,
        ui_refresh_rate_Hz=min(5.0, sampling_rate_Hz),
        buffer_duration_s=min(100.0, 20_000.0 / sampling_rate_Hz),
        environment=environment or EnvironmentConfiguration(dipoles=()),
        nodes=nodes,
    )


def test_vector_response_uses_non_commuting_c_axis_gain_soft_iron_order() -> None:
    gain = ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.5))
    soft = ((1.0, 0.2, 0.0), (0.2, 1.0, 0.0), (0.0, 0.0, 1.0))
    cross = ((1.0, 0.0, 0.1), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    bias = (1.0e-6, -2.0e-6, 3.0e-6)
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(
            gain_matrix=gain,
            soft_iron_matrix=soft,
            cross_axis_matrix=cross,
            bias_T=bias,
        ),
    )
    environment = EnvironmentConfiguration(uniform_field_T=(10.0e-6, 20.0e-6, 30.0e-6))

    reading = _generate(_configuration(node, environment=environment), 1)[0].observation.readings[0]
    gain_array = np.asarray(gain)
    soft_array = np.asarray(soft)
    cross_array = np.asarray(cross)
    field_array = np.asarray(environment.uniform_field_T)
    expected_linear_response = cross_array @ gain_array @ soft_array @ field_array
    previous_incorrect_response = cross_array @ soft_array @ gain_array @ field_array
    expected = expected_linear_response + np.asarray(bias)

    assert not np.allclose(gain_array @ soft_array, soft_array @ gain_array)
    assert not np.allclose(
        expected_linear_response,
        previous_incorrect_response,
        rtol=0.0,
        atol=1.0e-18,
    )
    np.testing.assert_allclose(reading.components_T, expected, rtol=0.0, atol=1.0e-18)


def test_vector_response_keeps_hard_offset_after_linear_response() -> None:
    bias = (1.0e-6, -2.0e-6, 3.0e-6)
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(bias_T=bias),
    )
    environment = EnvironmentConfiguration(uniform_field_T=(10.0e-6, 20.0e-6, 30.0e-6))
    reading = _generate(_configuration(node, environment=environment), 1)[0].observation.readings[0]
    expected = np.asarray(environment.uniform_field_T) + np.asarray(bias)

    np.testing.assert_allclose(reading.components_T, expected, rtol=0.0, atol=1.0e-18)


def test_vector_monoaxial_and_total_field_shapes_are_not_conflated() -> None:
    errors = NodeErrorConfiguration(saturation_limit_T=10.0)
    nodes = (
        SensorNodeConfiguration(sensor_id="V", position_m=(0.0, 0.0, 0.0), errors=errors),
        SensorNodeConfiguration(
            sensor_id="M",
            position_m=(1.0, 0.0, 0.0),
            measurement_mode=MeasurementMode.MONOAXIAL,
            monoaxial_axis_sensor=(0.0, 1.0, 0.0),
            errors=errors,
        ),
        SensorNodeConfiguration(
            sensor_id="T",
            position_m=(0.0, 1.0, 0.0),
            measurement_mode=MeasurementMode.TOTAL_FIELD,
            errors=errors,
        ),
    )
    readings = _generate(
        _configuration(*nodes, environment=EnvironmentConfiguration(uniform_field_T=(3.0, 4.0, 0.0))),
        1,
    )[0].observation.readings

    assert readings[0].components_T == pytest.approx((3.0, 4.0, 0.0))
    assert readings[0].value_T is None
    assert readings[1].components_T is None
    assert readings[1].value_T == pytest.approx(4.0)
    assert readings[2].components_T is None
    assert readings[2].value_T == pytest.approx(5.0)


def test_white_noise_is_rms_per_sample_not_sampling_rate_density() -> None:
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(white_noise_std_T_per_sample=(2.0e-9,) * 3),
    )
    slow = _generate(_configuration(node, sampling_rate_Hz=100.0), 1)[0]
    fast = _generate(_configuration(node, sampling_rate_Hz=400.0), 1)[0]

    assert slow.observation.readings[0].components_T == fast.observation.readings[0].components_T


def test_static_motion_ignores_a_nonzero_velocity_parameter() -> None:
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(1.0, 2.0, 3.0),
        motion=NodeMotionConfiguration(
            kind=MotionKind.STATIC,
            velocity_m_per_s=(10.0, -5.0, 2.0),
        ),
    )

    generated = _generate(_configuration(node), 3)

    assert [frame.observation.readings[0].position_m for frame in generated] == [
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0),
    ]


@pytest.mark.parametrize(
    ("driver", "sampling_rate_hz", "expected_temperature_k"),
    [
        (
            TemperatureDriverConfiguration(
                kind="ramp",
                ramp_rate_K_per_s=10.0,
                ramp_duration_s=0.2,
            ),
            10.0,
            (300.0, 301.0, 302.0, 302.0),
        ),
        (
            TemperatureDriverConfiguration(
                kind="sinusoidal",
                sinusoidal_amplitude_K=5.0,
                sinusoidal_frequency_Hz=1.0,
            ),
            4.0,
            (300.0, 305.0, 300.0, 295.0),
        ),
    ],
)
def test_temperature_driver_is_the_target_of_device_thermal_inertia(
    driver: TemperatureDriverConfiguration,
    sampling_rate_hz: float,
    expected_temperature_k: tuple[float, ...],
) -> None:
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(
            initial_temperature_K=300.0,
            ambient_temperature_K=300.0,
            temperature_driver=driver,
            thermal_time_constant_s=1.0e-6,
        ),
    )

    generated = _generate(
        _configuration(node, sampling_rate_Hz=sampling_rate_hz),
        len(expected_temperature_k),
    )
    observed = tuple(
        frame.observation.readings[0].observed_temperature_K for frame in generated
    )

    assert observed == pytest.approx(expected_temperature_k, abs=1.0e-10)


def test_temperature_ramp_is_inactive_before_its_scheduled_onset() -> None:
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(
            initial_temperature_K=300.0,
            ambient_temperature_K=300.0,
            temperature_driver=TemperatureDriverConfiguration(
                kind="ramp",
                start_time_s=2.0,
                ramp_rate_K_per_s=10.0,
                ramp_duration_s=2.0,
            ),
            thermal_time_constant_s=1.0e-6,
        ),
    )

    generated = _generate(_configuration(node, sampling_rate_Hz=1.0), 6)
    observed = tuple(
        frame.observation.readings[0].observed_temperature_K for frame in generated
    )

    assert observed == pytest.approx(
        (300.0, 300.0, 300.0, 310.0, 320.0, 320.0),
        abs=1.0e-10,
    )


def test_streams_are_replayable_and_adding_s2_does_not_change_s1() -> None:
    base = default_network_configuration(2)
    one = NetworkSessionConfiguration.model_validate(
        {**base.model_dump(), "nodes": [base.nodes[0].model_dump()]}
    )
    first = _generate(one, 20)
    replay = _generate(one, 20)
    two = _generate(base, 20)

    first_s1 = [frame.observation.readings[0].components_T for frame in first]
    assert first_s1 == [frame.observation.readings[0].components_T for frame in replay]
    assert first_s1 == [frame.observation.readings[0].components_T for frame in two]


def test_drift_event_on_s3_does_not_perturb_s1_random_stream() -> None:
    configuration = default_network_configuration(3)
    drift = NetworkEventConfiguration(
        event_id="s3-drift",
        kind=EventKind.NODE_DRIFT,
        start_time_s=0.0,
        duration_s=1.0,
        target_sensor_ids=("S3",),
        drift_rate_T_per_s=(1.0e-6, 0.0, 0.0),
    )
    baseline = _generate(configuration, 20)
    injected = _generate(configuration, 20, (drift,))

    assert [frame.observation.readings[0] for frame in baseline] == [
        frame.observation.readings[0] for frame in injected
    ]
    assert baseline[-1].observation.readings[2].components_T != (
        injected[-1].observation.readings[2].components_T
    )


def test_random_walk_is_persistent_replayable_and_advances_with_sqrt_dt() -> None:
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(random_walk_q_T2_per_s=(1.0e-16,) * 3),
    )
    first = _generate(_configuration(node, sampling_rate_Hz=100.0), 3)
    replay = _generate(_configuration(node, sampling_rate_Hz=100.0), 3)
    walk = [frame.truth.devices[0].random_walk_bias_T for frame in first]

    assert walk[0] == (0.0, 0.0, 0.0)
    assert walk[1] != walk[0]
    assert walk[2] != walk[1]
    assert walk == [frame.truth.devices[0].random_walk_bias_T for frame in replay]


def test_faults_are_composable_data_flags_and_never_simulator_exceptions() -> None:
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(clock_offset_s=0.01),
    )
    events = (
        NetworkEventConfiguration(
            event_id="drop",
            kind=EventKind.DROPOUT,
            start_time_s=0.0,
            duration_s=1.0,
            target_sensor_ids=("S1",),
        ),
        NetworkEventConfiguration(
            event_id="noise",
            kind=EventKind.NODE_NOISE_BURST,
            start_time_s=0.0,
            duration_s=1.0,
            target_sensor_ids=("S1",),
            noise_multiplier=3.0,
        ),
    )
    reading = _generate(_configuration(node), 1, events)[0].observation.readings[0]

    assert reading.components_T is None
    assert reading.value_T is None
    assert reading.valid is False
    assert set(reading.quality_flags) == {
        QualityFlag.CLOCK_ERROR,
        QualityFlag.SIGNAL_ABSENT,
    }


def test_dropout_flags_do_not_leak_hidden_clipping_or_noise_burst_cause() -> None:
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(saturation_limit_T=1.0e-9),
    )
    dropout = NetworkEventConfiguration(
        event_id="drop",
        kind=EventKind.DROPOUT,
        start_time_s=0.0,
        duration_s=1.0,
        target_sensor_ids=("S1",),
    )
    hidden_noise = NetworkEventConfiguration(
        event_id="hidden-noise",
        kind=EventKind.NODE_NOISE_BURST,
        start_time_s=0.0,
        duration_s=1.0,
        target_sensor_ids=("S1",),
        noise_multiplier=10.0,
    )
    quiet = _generate(
        _configuration(node, environment=EnvironmentConfiguration(uniform_field_T=(0.0, 0.0, 0.0))),
        1,
        (dropout,),
    )[0].observation.readings[0]
    hidden_clipped = _generate(
        _configuration(node, environment=EnvironmentConfiguration(uniform_field_T=(1.0, 1.0, 1.0))),
        1,
        (dropout, hidden_noise),
    )[0].observation.readings[0]

    assert quiet.components_T is hidden_clipped.components_T is None
    assert quiet.quality_flags == hidden_clipped.quality_flags == (
        QualityFlag.SIGNAL_ABSENT,
    )


def test_stuck_reading_keeps_frozen_flags_without_leaking_current_hidden_field() -> None:
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(saturation_limit_T=10.0e-6),
    )
    stuck = NetworkEventConfiguration(
        event_id="stuck",
        kind=EventKind.STUCK,
        start_time_s=0.01,
        duration_s=1.0,
        target_sensor_ids=("S1",),
    )
    hidden_field = NetworkEventConfiguration(
        event_id="hidden-world-field",
        kind=EventKind.WORLD_FIELD_OFFSET,
        start_time_s=0.01,
        duration_s=1.0,
        field_offset_T=(1.0, 0.0, 0.0),
    )
    configuration = _configuration(
        node,
        environment=EnvironmentConfiguration(uniform_field_T=(1.0e-6, 0.0, 0.0)),
    )
    baseline = _generate(configuration, 2, (stuck,))[-1]
    changed_truth = _generate(configuration, 2, (stuck, hidden_field))[-1]
    baseline_reading = baseline.observation.readings[0]
    changed_reading = changed_truth.observation.readings[0]

    assert baseline_reading == changed_reading
    assert baseline_reading.quality_flags == (QualityFlag.STUCK,)
    assert baseline.truth.fields[0].field_true_world_T != (
        changed_truth.truth.fields[0].field_true_world_T
    )


def test_point_dipole_domain_crossing_becomes_null_measurement() -> None:
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(-0.2, 0.0, 0.0),
        motion=NodeMotionConfiguration(
            kind=MotionKind.LINEAR_TRANSLATION,
            velocity_m_per_s=(0.15, 0.0, 0.0),
        ),
    )
    environment = EnvironmentConfiguration(
        dipoles=(
            DipoleSourceConfiguration(
                source_id="D1",
                initial_position_m=(0.0, 0.0, 0.0),
                moment_A_m2=(0.0, 0.0, 1.0),
                minimum_distance_m=0.05,
            ),
        )
    )
    frames = _generate(_configuration(node, sampling_rate_Hz=10.0, environment=environment), 12)
    reading = frames[-1].observation.readings[0]

    assert reading.components_T is None
    assert reading.valid is False
    assert QualityFlag.SIGNAL_ABSENT in reading.quality_flags
    assert reading.quality_flags == (QualityFlag.SIGNAL_ABSENT,)
    assert frames[-1].truth.fields[0].model_valid is False


def test_gaussian_world_field_is_shared_and_rotates_only_in_sensor_frame() -> None:
    anomaly = GaussianAnomalyConfiguration(
        anomaly_id="A1",
        peak_amplitude_T=100.0e-9,
        direction_world=(1.0, 0.0, 0.0),
        center_position_m=(0.0, 0.0, 0.0),
        spatial_scale_m=0.5,
    )
    nodes = (
        SensorNodeConfiguration(sensor_id="S1", position_m=(-0.1, 0.0, 0.0)),
        SensorNodeConfiguration(
            sensor_id="S2",
            position_m=(0.1, 0.0, 0.0),
            orientation_world_to_sensor_wxyz=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    environment = EnvironmentConfiguration(
        uniform_field_T=(0.0, 0.0, 0.0),
        gaussian_anomalies=(anomaly,),
    )
    frame = _generate(_configuration(*nodes, environment=environment), 1)[0]

    assert frame.truth.fields[0].anomaly_field_world_T == pytest.approx(
        frame.truth.fields[1].anomaly_field_world_T
    )
    first = frame.truth.fields[0].ideal_field_sensor_T
    second = frame.truth.fields[1].ideal_field_sensor_T
    assert first is not None and second is not None
    assert second[0] == pytest.approx(-first[0])


def test_gaussian_crossing_temporal_width_is_spatial_scale_divided_by_speed() -> None:
    finite = default_vector_configuration(VectorScenario.LOCAL_ANOMALY_CROSSING)
    assert not finite.environment.dipoles
    assert len(finite.environment.gaussian_anomalies) == 1
    configuration = NetworkSessionConfiguration(
        random_seed=finite.random_seed,
        sampling_rate_Hz=100.0,
        ui_refresh_rate_Hz=5.0,
        buffer_duration_s=2.0,
        environment=finite.environment,
        nodes=(finite.node,),
    )
    frames = _generate(configuration, 126)
    before = frames[75].truth.fields[0].anomaly_field_world_T[2]
    peak = frames[100].truth.fields[0].anomaly_field_world_T[2]
    after = frames[125].truth.fields[0].anomaly_field_world_T[2]

    assert peak == pytest.approx(200.0e-9)
    assert before == pytest.approx(peak * np.exp(-0.5))
    assert after == pytest.approx(before)


def test_bandwidth_filter_reduces_dynamic_deterministic_response_and_persists() -> None:
    motion = NodeMotionConfiguration(
        kind=MotionKind.HIGH_DYNAMIC,
        angular_rate_rad_per_s=(0.0, 0.0, 2.0 * np.pi * 5.0),
        modulation_frequency_Hz=(0.0, 0.0, 0.0),
    )
    unfiltered_node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        motion=motion,
        errors=NodeErrorConfiguration(),
    )
    filtered_node = unfiltered_node.model_copy(
        update={"errors": NodeErrorConfiguration(bandwidth_Hz=1.0)}
    )
    environment = EnvironmentConfiguration(uniform_field_T=(1.0e-6, 0.0, 0.0))
    unfiltered = _generate(
        _configuration(unfiltered_node, environment=environment),
        300,
    )
    filtered_configuration = _configuration(filtered_node, environment=environment)
    filtered_simulator = NetworkSimulator(filtered_configuration)
    filtered = []
    for index in range(300):
        filtered.append(
            filtered_simulator.generate(
                "filter",
                index + 1,
                index / 100.0,
                EPOCH,
                (),
            )
        )

    unfiltered_x = np.asarray(
        [frame.observation.readings[0].components_T[0] for frame in unfiltered[100:]]
    )
    filtered_x = np.asarray(
        [frame.observation.readings[0].components_T[0] for frame in filtered[100:]]
    )
    assert np.std(filtered_x) < 0.3 * np.std(unfiltered_x)

    replayed = _generate(filtered_configuration, 300)
    assert [frame.observation.readings[0].components_T for frame in filtered] == [
        frame.observation.readings[0].components_T for frame in replayed
    ]


def test_bandwidth_filters_seeded_white_noise_and_replay_is_deterministic() -> None:
    filtered_node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(
            bandwidth_Hz=2.0,
            white_noise_std_T_per_sample=(2.0e-9,) * 3,
        ),
    )
    unfiltered_node = filtered_node.model_copy(
        update={
            "errors": NodeErrorConfiguration(
                bandwidth_Hz=None,
                white_noise_std_T_per_sample=(2.0e-9,) * 3,
            )
        }
    )
    zero_environment = EnvironmentConfiguration(uniform_field_T=(0.0, 0.0, 0.0))
    sample_count = 4_096
    sampling_rate_hz = 100.0
    filtered_configuration = _configuration(
        filtered_node,
        sampling_rate_Hz=sampling_rate_hz,
        environment=zero_environment,
        seed=31,
    )
    unfiltered_configuration = _configuration(
        unfiltered_node,
        sampling_rate_Hz=sampling_rate_hz,
        environment=zero_environment,
        seed=31,
    )
    filtered_noise = _generate(
        filtered_configuration,
        sample_count,
    )
    unfiltered_noise = _generate(
        unfiltered_configuration,
        sample_count,
    )

    raw_x = np.asarray(
        [frame.observation.readings[0].components_T[0] for frame in unfiltered_noise]
    )
    filtered_x = np.asarray(
        [frame.observation.readings[0].components_T[0] for frame in filtered_noise]
    )
    alpha = 1.0 - np.exp(-2.0 * np.pi * 2.0 / sampling_rate_hz)
    expected_filtered = np.empty_like(raw_x)
    expected_filtered[0] = raw_x[0]
    for index in range(1, sample_count):
        expected_filtered[index] = expected_filtered[index - 1] + alpha * (
            raw_x[index] - expected_filtered[index - 1]
        )

    np.testing.assert_allclose(filtered_x, expected_filtered, rtol=0.0, atol=1.0e-24)
    raw_std = float(np.std(raw_x[512:]))
    filtered_std = float(np.std(filtered_x[512:]))
    assert raw_std == pytest.approx(2.0e-9, rel=0.08)
    assert filtered_std < 0.3 * raw_std

    replayed = _generate(filtered_configuration, sample_count)
    assert [frame.observation.readings[0].components_T for frame in filtered_noise] == [
        frame.observation.readings[0].components_T for frame in replayed
    ]


def test_white_noise_then_bandwidth_then_saturation_uses_pre_clip_mask() -> None:
    sampling_rate_hz = 100.0
    bandwidth_hz = 1.0
    saturation_limit_t = 1.0e-6
    sample_count = 512
    zero_environment = EnvironmentConfiguration(uniform_field_T=(0.0, 0.0, 0.0))
    raw_node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(
            white_noise_std_T_per_sample=(4.0e-6,) * 3,
            saturation_limit_T=10.0,
        ),
    )
    filtered_node = raw_node.model_copy(
        update={
            "errors": NodeErrorConfiguration(
                white_noise_std_T_per_sample=(4.0e-6,) * 3,
                bandwidth_Hz=bandwidth_hz,
                saturation_limit_T=saturation_limit_t,
            )
        }
    )
    raw_frames = _generate(
        _configuration(
            raw_node,
            sampling_rate_Hz=sampling_rate_hz,
            environment=zero_environment,
            seed=59,
        ),
        sample_count,
    )
    measured_frames = _generate(
        _configuration(
            filtered_node,
            sampling_rate_Hz=sampling_rate_hz,
            environment=zero_environment,
            seed=59,
        ),
        sample_count,
    )
    raw_x = np.asarray(
        [frame.observation.readings[0].components_T[0] for frame in raw_frames]
    )
    alpha = 1.0 - np.exp(-2.0 * np.pi * bandwidth_hz / sampling_rate_hz)
    expected_pre_clip = np.empty_like(raw_x)
    expected_pre_clip[0] = raw_x[0]
    for index in range(1, sample_count):
        expected_pre_clip[index] = expected_pre_clip[index - 1] + alpha * (
            raw_x[index] - expected_pre_clip[index - 1]
        )
    expected_output = np.clip(
        expected_pre_clip,
        -saturation_limit_t,
        saturation_limit_t,
    )
    measured_x = np.asarray(
        [frame.observation.readings[0].components_T[0] for frame in measured_frames]
    )
    measured_mask_x = [
        frame.observation.readings[0].saturation_mask[0] for frame in measured_frames
    ]

    np.testing.assert_allclose(measured_x, expected_output, rtol=0.0, atol=1.0e-24)
    assert measured_mask_x == [
        bool(abs(value) > saturation_limit_t) for value in expected_pre_clip
    ]

    clipped_before_filter = np.clip(raw_x, -saturation_limit_t, saturation_limit_t)
    wrong_saturation_first = np.empty_like(raw_x)
    wrong_saturation_first[0] = clipped_before_filter[0]
    for index in range(1, sample_count):
        wrong_saturation_first[index] = wrong_saturation_first[index - 1] + alpha * (
            clipped_before_filter[index] - wrong_saturation_first[index - 1]
        )
    wrong_filter_then_noise = np.clip(raw_x, -saturation_limit_t, saturation_limit_t)

    assert float(np.max(np.abs(measured_x - wrong_saturation_first))) > 1.0e-7
    assert float(np.max(np.abs(measured_x - wrong_filter_then_noise))) > 1.0e-7


def test_bandwidth_filters_hard_offset_step() -> None:
    no_noise_node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(bandwidth_Hz=1.0),
    )
    zero_environment = EnvironmentConfiguration(uniform_field_T=(0.0, 0.0, 0.0))

    step = NetworkEventConfiguration(
        event_id="bias-step",
        kind=EventKind.NODE_BIAS,
        start_time_s=0.1,
        duration_s=1.0,
        target_sensor_ids=("S1",),
        field_offset_T=(1.0e-6, 0.0, 0.0),
    )
    response = _generate(
        _configuration(no_noise_node, sampling_rate_Hz=10.0, environment=zero_environment),
        3,
        (step,),
    )
    alpha = 1.0 - np.exp(-2.0 * np.pi * 1.0 / 10.0)
    x = [frame.observation.readings[0].components_T[0] for frame in response]

    assert x[0] == 0.0
    assert x[1] == pytest.approx(alpha * 1.0e-6)
    assert x[2] == pytest.approx((1.0 - (1.0 - alpha) ** 2) * 1.0e-6)


@pytest.mark.parametrize(
    ("scenario", "position_changes", "orientation_changes"),
    [
        (VectorScenario.STATIC_REFERENCE, False, False),
        (VectorScenario.CALIBRATION_TUMBLE, False, True),
        (VectorScenario.HIGH_DYNAMIC, False, True),
        (VectorScenario.LINEAR_TRANSLATION, True, False),
        (VectorScenario.LOCAL_ANOMALY_CROSSING, True, False),
        (VectorScenario.COMBINED_STRESS, True, True),
    ],
)
def test_scenarios_change_the_declared_pose_components(
    scenario: VectorScenario,
    position_changes: bool,
    orientation_changes: bool,
) -> None:
    finite = default_vector_configuration(scenario)
    configuration = NetworkSessionConfiguration(
        random_seed=finite.random_seed,
        sampling_rate_Hz=finite.sampling_rate_Hz,
        ui_refresh_rate_Hz=5.0,
        buffer_duration_s=finite.duration_s,
        environment=finite.environment,
        nodes=(finite.node,),
        events=finite.events,
    )
    first, *_, last = _generate(configuration, 20, finite.events)
    first_reading = first.observation.readings[0]
    last_reading = last.observation.readings[0]

    assert (first_reading.position_m != last_reading.position_m) is position_changes
    assert (
        first_reading.orientation_world_to_sensor_wxyz
        != last_reading.orientation_world_to_sensor_wxyz
    ) is orientation_changes


def test_vector_saturation_has_per_axis_mask() -> None:
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(saturation_limit_T=2.0),
    )
    reading = _generate(
        _configuration(node, environment=EnvironmentConfiguration(uniform_field_T=(3.0, 1.0, -4.0))),
        1,
    )[0].observation.readings[0]

    assert reading.components_T == (2.0, 1.0, -2.0)
    assert reading.saturation_mask == (True, False, True)
    assert reading.valid is False
    assert QualityFlag.CLIPPED in reading.quality_flags


def test_vector_saturation_supports_optional_axis_specific_limits() -> None:
    node = SensorNodeConfiguration(
        sensor_id="S1",
        position_m=(0.0, 0.0, 0.0),
        errors=NodeErrorConfiguration(
            saturation_limit_T=10.0,
            saturation_limits_T=(2.0, 3.0, 4.0),
        ),
    )
    reading = _generate(
        _configuration(
            node,
            environment=EnvironmentConfiguration(uniform_field_T=(3.0, -2.0, 5.0)),
        ),
        1,
    )[0].observation.readings[0]

    assert reading.components_T == (2.0, -2.0, 4.0)
    assert reading.saturation_mask == (True, False, True)
    assert reading.valid is False
