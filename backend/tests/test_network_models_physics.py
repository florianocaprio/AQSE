from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
from pydantic import ValidationError

from app.network.defaults import default_network_configuration
from app.network.models import (
    DipoleSourceConfiguration,
    EnvironmentConfiguration,
    EventKind,
    FiniteVectorSimulationConfiguration,
    GaussianAnomalyConfiguration,
    NetworkEventConfiguration,
    NetworkSessionConfiguration,
    NodeErrorConfiguration,
    PeriodicFieldConfiguration,
    SensorNodeConfiguration,
    TemperatureDriverConfiguration,
)
from app.network.motion import haar_pose_quaternion
from app.network.physics import (
    gaussian_anomaly_field_T,
    point_dipole_field_T,
    quaternion_world_to_sensor_matrix,
)
from app.network.providers import (
    ConstantLocalFieldProvider,
    FieldProvider,
    SyntheticSpatialFieldProvider,
    WorldMagneticModelProvider,
    dipole_position_m,
)


@pytest.mark.parametrize("node_count", range(1, 9))
def test_network_defaults_support_one_through_eight_nodes(node_count: int) -> None:
    configuration = default_network_configuration(node_count)

    assert len(configuration.nodes) == node_count
    assert len({node.sensor_id for node in configuration.nodes}) == node_count
    assert all(node.errors.bandwidth_Hz is None for node in configuration.nodes)


@pytest.mark.parametrize("node_count", [0, 9])
def test_network_defaults_reject_out_of_range_node_counts(node_count: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 8"):
        default_network_configuration(node_count)


def test_magnetostatic_gradient_must_be_symmetric_and_traceless() -> None:
    valid = EnvironmentConfiguration(
        gradient_T_per_m=((2.0, 0.5, 0.0), (0.5, -1.0, 0.25), (0.0, 0.25, -1.0))
    )
    assert np.trace(np.asarray(valid.gradient_T_per_m)) == pytest.approx(0.0)

    with pytest.raises(ValidationError, match="symmetric"):
        EnvironmentConfiguration(
            gradient_T_per_m=((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        )
    with pytest.raises(ValidationError, match="traceless"):
        EnvironmentConfiguration(
            gradient_T_per_m=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )


def test_response_matrices_and_noise_contract_are_strict() -> None:
    with pytest.raises(ValidationError, match="positive definite"):
        NodeErrorConfiguration(
            soft_iron_matrix=((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
        )
    with pytest.raises(ValidationError, match="gain_matrix must be invertible"):
        NodeErrorConfiguration(
            gain_matrix=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )
    with pytest.raises(ValidationError, match="cross_axis_matrix condition number"):
        NodeErrorConfiguration(
            cross_axis_matrix=((1.0, 0.0, 0.0), (0.0, 0.001, 0.0), (0.0, 0.0, 1.0))
        )
    with pytest.raises(ValidationError, match="white_noise_std_T_per_sample"):
        NodeErrorConfiguration(white_noise_std_T_per_sample=(-1.0, 0.0, 0.0))
    with pytest.raises(ValidationError):
        NodeErrorConfiguration(unknown_noise_density=1.0)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="diagonal"):
        NodeErrorConfiguration(
            gain_matrix=((1.0, 0.1, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
    with pytest.raises(ValidationError, match="positive"):
        NodeErrorConfiguration(
            gain_matrix=((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
    with pytest.raises(ValidationError):
        NodeErrorConfiguration(clock_offset_s=1.0e308)
    with pytest.raises(ValidationError, match="saturation_limits_T components"):
        NodeErrorConfiguration(saturation_limits_T=(1.0, 0.0, 1.0))
    with pytest.raises(ValidationError, match="ambient-temperature ramp"):
        NodeErrorConfiguration(
            ambient_temperature_K=300.0,
            temperature_driver=TemperatureDriverConfiguration(
                kind="ramp",
                ramp_rate_K_per_s=-100.0,
                ramp_duration_s=4.0,
            ),
        )


def test_bandwidth_must_be_strictly_below_nyquist() -> None:
    configuration = default_network_configuration(1)
    node = configuration.nodes[0].model_copy(
        update={"errors": configuration.nodes[0].errors.model_copy(update={"bandwidth_Hz": 50.0})}
    )

    with pytest.raises(ValidationError, match="Nyquist"):
        NetworkSessionConfiguration.model_validate(
            {**configuration.model_dump(), "nodes": [node.model_dump()]}
        )

    periodic_environment = configuration.environment.model_copy(
        update={
            "periodic_fields": (
                PeriodicFieldConfiguration(
                    source_id="aliasing",
                    amplitude_world_T=(1.0e-9, 0.0, 0.0),
                    frequency_Hz=50.0,
                ),
            )
        }
    )
    with pytest.raises(ValidationError, match="periodic field.*Nyquist"):
        NetworkSessionConfiguration.model_validate(
            {
                **configuration.model_dump(),
                "environment": periodic_environment.model_dump(),
            }
        )

    temperature_node = configuration.nodes[0].model_copy(
        update={
            "errors": configuration.nodes[0].errors.model_copy(
                update={
                    "temperature_driver": TemperatureDriverConfiguration(
                        kind="sinusoidal",
                        sinusoidal_amplitude_K=1.0,
                        sinusoidal_frequency_Hz=50.0,
                    )
                }
            )
        }
    )
    with pytest.raises(ValidationError, match="temperature-driver frequency.*Nyquist"):
        NetworkSessionConfiguration.model_validate(
            {**configuration.model_dump(), "nodes": [temperature_node.model_dump()]}
        )


def test_buffer_capacity_is_never_zero_after_rounding() -> None:
    configuration = NetworkSessionConfiguration(
        sampling_rate_Hz=0.1,
        ui_refresh_rate_Hz=0.1,
        buffer_duration_s=0.1,
        nodes=(SensorNodeConfiguration(sensor_id="S1", position_m=(0.0, 0.0, 0.0)),),
    )

    assert configuration.buffer_capacity_frames == 1


def test_initial_event_ids_must_be_nonempty_and_unique() -> None:
    with pytest.raises(ValidationError):
        NetworkEventConfiguration(
            event_id="",
            kind=EventKind.NODE_BIAS,
            start_time_s=0.0,
            duration_s=1.0,
            target_sensor_ids=("S1",),
        )
    event = NetworkEventConfiguration(
        event_id="duplicate",
        kind=EventKind.NODE_BIAS,
        start_time_s=0.0,
        duration_s=1.0,
        target_sensor_ids=("S1",),
    )
    with pytest.raises(ValidationError, match="event_id values must be unique"):
        NetworkSessionConfiguration(
            nodes=(SensorNodeConfiguration(sensor_id="S1", position_m=(0.0, 0.0, 0.0)),),
            events=(event, event),
        )
    with pytest.raises(ValidationError, match="event_id values must be unique"):
        FiniteVectorSimulationConfiguration(
            node=SensorNodeConfiguration(
                sensor_id="S1",
                position_m=(0.0, 0.0, 0.0),
            ),
            events=(event, event),
        )
    with pytest.raises(ValidationError, match="at least two target"):
        NetworkEventConfiguration(
            event_id="not-shared",
            kind=EventKind.SHARED_INSTRUMENT_OFFSET,
            start_time_s=0.0,
            duration_s=1.0,
            target_sensor_ids=("S1",),
        )


def test_quaternion_contract_is_normalized_canonical_and_orthogonal() -> None:
    with pytest.raises(ValidationError, match="canonical sign"):
        SensorNodeConfiguration(
            sensor_id="S1",
            position_m=(0.0, 0.0, 0.0),
            orientation_world_to_sensor_wxyz=(-1.0, 0.0, 0.0, 0.0),
        )

    quaternion = (sqrt(0.5), 0.0, 0.0, sqrt(0.5))
    rotation = quaternion_world_to_sensor_matrix(quaternion)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-15)
    np.testing.assert_allclose(
        rotation @ np.asarray((1.0, 0.0, 0.0)),
        (0.0, 1.0, 0.0),
        atol=1.0e-15,
    )
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_calibration_tumble_is_a_real_low_discrepancy_haar_pose_sweep() -> None:
    rotations = np.asarray(
        [
            quaternion_world_to_sensor_matrix(haar_pose_quaternion(index))
            for index in range(1, 4097)
        ]
    )
    transformed_axis = rotations @ np.asarray((0.0, 0.0, 1.0))

    assert np.max(np.abs(np.mean(transformed_axis, axis=0))) < 0.01
    np.testing.assert_allclose(
        np.mean(transformed_axis**2, axis=0),
        np.full(3, 1.0 / 3.0),
        atol=0.01,
    )


def test_point_dipole_has_inverse_cube_decay_and_no_hidden_clamp() -> None:
    near = point_dipole_field_T(
        sensor_position_m=(0.0, 0.0, 1.0),
        source_position_m=(0.0, 0.0, 0.0),
        moment_A_m2=(0.0, 0.0, 1.0),
        minimum_distance_m=0.01,
    )
    far = point_dipole_field_T(
        sensor_position_m=(0.0, 0.0, 2.0),
        source_position_m=(0.0, 0.0, 0.0),
        moment_A_m2=(0.0, 0.0, 1.0),
        minimum_distance_m=0.01,
    )

    np.testing.assert_allclose(near, 8.0 * far, atol=0.0)
    assert near[2] / far[2] == pytest.approx(8.0)
    for distance_m in (0.005, 0.01):
        with pytest.raises(ValueError, match="below the declared minimum"):
            point_dipole_field_T(
                sensor_position_m=(0.0, 0.0, distance_m),
                source_position_m=(0.0, 0.0, 0.0),
                moment_A_m2=(0.0, 0.0, 1.0),
                minimum_distance_m=0.01,
            )


def test_scheduled_dipole_is_inactive_before_onset_and_moves_from_its_onset() -> None:
    source = DipoleSourceConfiguration(
        source_id="scheduled",
        initial_position_m=(-2.0, 0.0, 0.0),
        velocity_m_per_s=(0.5, 0.0, 0.0),
        moment_A_m2=(0.0, 0.0, 1.0),
        active_start_time_s=5.0,
        active_duration_s=2.0,
    )
    provider = SyntheticSpatialFieldProvider(
        EnvironmentConfiguration(dipoles=(source,))
    )

    np.testing.assert_allclose(dipole_position_m(source, 5.0), (-2.0, 0.0, 0.0))
    np.testing.assert_allclose(dipole_position_m(source, 6.0), (-1.5, 0.0, 0.0))
    np.testing.assert_allclose(
        provider.evaluate((0.0, 0.0, 1.0), 4.999).dipole_field_world_T,
        np.zeros(3),
    )
    assert np.linalg.norm(
        provider.evaluate((0.0, 0.0, 1.0), 5.0).dipole_field_world_T
    ) > 0.0
    np.testing.assert_allclose(
        provider.evaluate((0.0, 0.0, 1.0), 7.0).dipole_field_world_T,
        np.zeros(3),
    )


def test_gaussian_anomaly_peaks_at_center_and_is_spatially_symmetric() -> None:
    anomaly = GaussianAnomalyConfiguration(
        anomaly_id="A1",
        peak_amplitude_T=200.0e-9,
        direction_world=(0.0, 1.0, 0.0),
        center_position_m=(1.0, 2.0, 3.0),
        spatial_scale_m=0.25,
    )
    center = gaussian_anomaly_field_T((1.0, 2.0, 3.0), anomaly)
    left = gaussian_anomaly_field_T((0.75, 2.0, 3.0), anomaly)
    right = gaussian_anomaly_field_T((1.25, 2.0, 3.0), anomaly)

    np.testing.assert_allclose(center, (0.0, 200.0e-9, 0.0), atol=0.0)
    np.testing.assert_allclose(left, right, atol=0.0)
    assert left[1] == pytest.approx(200.0e-9 * np.exp(-0.5))


def test_field_provider_boundary_is_real_and_wmm_is_explicitly_unconfigured() -> None:
    environment = EnvironmentConfiguration()
    constant = ConstantLocalFieldProvider(environment)
    synthetic = SyntheticSpatialFieldProvider(environment)

    assert isinstance(constant, FieldProvider)
    assert isinstance(synthetic, FieldProvider)
    assert constant.evaluate((0.0, 0.0, 0.0), 0.0).field_world_T is not None
    assert synthetic.evaluate((0.0, 0.0, 0.0), 0.0).field_world_T is not None
    with pytest.raises(RuntimeError, match="not configured"):
        WorldMagneticModelProvider().evaluate((0.0, 0.0, 0.0), 0.0)
