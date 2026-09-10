from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import blake2s
from math import exp, pi, sin, sqrt
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from app.network.models import (
    ActiveCause,
    Bool3,
    BufferedFrame,
    DeviceTruth,
    EventKind,
    FieldTruthAtNode,
    MeasurementMode,
    MotionKind,
    NetworkEventConfiguration,
    NetworkSessionConfiguration,
    NodeErrorConfiguration,
    ObservationFrame,
    QualityFlag,
    SensorNodeConfiguration,
    SensorReading,
    TemperatureDriverKind,
    TruthFrame,
    Vector3,
)
from app.network.motion import (
    haar_pose_quaternion,
    integrate_world_to_sensor_quaternion,
)
from app.network.physics import (
    as_vector,
    evaluate_environment,
    quaternion_world_to_sensor_matrix,
)


def _tuple3(values: NDArray[np.float64]) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _stable_stream_key(namespace: str) -> int:
    return int.from_bytes(
        blake2s(namespace.encode("utf-8"), digest_size=4).digest(),
        byteorder="little",
    )


def _independent_rng(seed: int, namespace: str) -> np.random.Generator:
    sequence = np.random.SeedSequence([seed, _stable_stream_key(namespace)])
    return np.random.default_rng(sequence)


def _ou_update(
    value: NDArray[np.float64],
    sigma: NDArray[np.float64],
    tau_s: float,
    dt_s: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    if dt_s <= 0.0:
        return value
    coefficient = exp(-dt_s / tau_s)
    innovation_scale = sigma * sqrt(max(0.0, 1.0 - coefficient * coefficient))
    return coefficient * value + innovation_scale * rng.normal(size=3)


def _event_is_active(event: NetworkEventConfiguration, sim_time_s: float) -> bool:
    return event.start_time_s <= sim_time_s < event.start_time_s + event.duration_s


def _ambient_temperature_target_K(
    errors: NodeErrorConfiguration,
    sim_time_s: float,
) -> float:
    """Return the bounded ambient target used by the device thermal response."""

    driver = errors.temperature_driver
    if driver is None or driver.kind is TemperatureDriverKind.CONSTANT:
        return errors.ambient_temperature_K
    if sim_time_s < driver.start_time_s:
        return errors.ambient_temperature_K
    elapsed_since_start_s = sim_time_s - driver.start_time_s
    if driver.kind is TemperatureDriverKind.RAMP:
        elapsed_s = min(elapsed_since_start_s, driver.ramp_duration_s)
        return errors.ambient_temperature_K + driver.ramp_rate_K_per_s * elapsed_s
    return errors.ambient_temperature_K + driver.sinusoidal_amplitude_K * sin(
        2.0 * pi * driver.sinusoidal_frequency_Hz * elapsed_since_start_s
        + driver.sinusoidal_phase_rad
    )


@dataclass
class _NodeRuntime:
    random_walk_bias_T: NDArray[np.float64]
    correlated_noise_T: NDArray[np.float64]
    device_temperature_K: float
    filtered_signal_T: NDArray[np.float64] | None
    last_value_T: float | None
    last_components_T: Vector3 | None
    last_saturation_mask: Bool3 | None
    last_quality_flags: tuple[QualityFlag, ...]
    position_m: Vector3
    orientation_world_to_sensor_wxyz: tuple[float, float, float, float]
    haar_shifts: Vector3
    sequence_id: int
    dropout_until_s: float
    stuck_until_s: float
    white_rng: np.random.Generator
    random_walk_rng: np.random.Generator
    ou_rng: np.random.Generator
    dropout_rng: np.random.Generator
    stuck_rng: np.random.Generator


class NetworkSimulator:
    """Stateful SI-unit simulator for one deterministic sensor-network session."""

    def __init__(self, configuration: NetworkSessionConfiguration) -> None:
        self.configuration = configuration
        seed = configuration.random_seed
        self._environment_rng = _independent_rng(seed, "environment:common-ou")
        environment_sigma = as_vector(configuration.environment.common_ou_sigma_T)
        self._common_field_T = environment_sigma * self._environment_rng.normal(size=3)
        self._nodes: dict[str, _NodeRuntime] = {}
        for node in configuration.nodes:
            sigma = as_vector(node.errors.ou_sigma_T)
            self._nodes[node.sensor_id] = _NodeRuntime(
                random_walk_bias_T=np.zeros(3, dtype=np.float64),
                correlated_noise_T=sigma
                * _independent_rng(seed, f"node:{node.sensor_id}:ou:init").normal(
                    size=3
                ),
                device_temperature_K=node.errors.initial_temperature_K,
                filtered_signal_T=None,
                last_value_T=None,
                last_components_T=None,
                last_saturation_mask=None,
                last_quality_flags=(),
                position_m=node.position_m,
                orientation_world_to_sensor_wxyz=(
                    node.orientation_world_to_sensor_wxyz
                ),
                haar_shifts=_tuple3(
                    _independent_rng(
                        seed,
                        f"node:{node.sensor_id}:haar-shifts",
                    ).random(3)
                ),
                sequence_id=0,
                dropout_until_s=-1.0,
                stuck_until_s=-1.0,
                white_rng=_independent_rng(seed, f"node:{node.sensor_id}:white"),
                random_walk_rng=_independent_rng(
                    seed,
                    f"node:{node.sensor_id}:random-walk",
                ),
                ou_rng=_independent_rng(seed, f"node:{node.sensor_id}:ou"),
                dropout_rng=_independent_rng(seed, f"node:{node.sensor_id}:dropout"),
                stuck_rng=_independent_rng(seed, f"node:{node.sensor_id}:stuck"),
            )

    def generate(
        self,
        session_id: str,
        frame_id: int,
        sim_time_s: float,
        epoch_utc: datetime,
        events: Iterable[NetworkEventConfiguration],
        configuration_version: int = 1,
    ) -> BufferedFrame:
        dt_s = 0.0 if frame_id == 1 else self.configuration.sample_interval_s
        environment = self.configuration.environment
        self._common_field_T = _ou_update(
            self._common_field_T,
            as_vector(environment.common_ou_sigma_T),
            environment.common_ou_tau_s,
            dt_s,
            self._environment_rng,
        )

        active_events = tuple(
            event for event in events if _event_is_active(event, sim_time_s)
        )
        world_event_field = sum(
            (
                as_vector(event.field_offset_T)
                for event in active_events
                if event.kind is EventKind.WORLD_FIELD_OFFSET
            ),
            start=np.zeros(3, dtype=np.float64),
        )

        readings: list[SensorReading] = []
        fields: list[FieldTruthAtNode] = []
        devices: list[DeviceTruth] = []
        for node in self.configuration.nodes:
            reading, field_truth, device_truth = self._generate_node(
                node=node,
                session_epoch_utc=epoch_utc,
                sim_time_s=sim_time_s,
                dt_s=dt_s,
                world_event_field_T=world_event_field,
                active_events=active_events,
            )
            readings.append(reading)
            fields.append(field_truth)
            devices.append(device_truth)

        observation = ObservationFrame(
            session_id=session_id,
            configuration_version=configuration_version,
            frame_id=frame_id,
            sim_time_s=sim_time_s,
            readings=tuple(readings),
        )
        truth = TruthFrame(
            session_id=session_id,
            configuration_version=configuration_version,
            frame_id=frame_id,
            sim_time_s=sim_time_s,
            fields=tuple(fields),
            devices=tuple(devices),
            active_causes=tuple(
                ActiveCause(
                    event_id=event.event_id,
                    kind=event.kind,
                    target_sensor_ids=event.target_sensor_ids,
                )
                for event in active_events
            ),
        )
        return BufferedFrame(observation=observation, truth=truth)

    def _generate_node(
        self,
        node: SensorNodeConfiguration,
        session_epoch_utc: datetime,
        sim_time_s: float,
        dt_s: float,
        world_event_field_T: NDArray[np.float64],
        active_events: tuple[NetworkEventConfiguration, ...],
    ) -> tuple[SensorReading, FieldTruthAtNode, DeviceTruth]:
        runtime = self._nodes[node.sensor_id]
        runtime.sequence_id += 1
        errors = node.errors

        translating_motion = node.motion.kind in {
            MotionKind.LINEAR_TRANSLATION,
            MotionKind.LOCAL_ANOMALY_CROSSING,
            MotionKind.COMBINED_STRESS,
        }
        runtime.position_m = _tuple3(
            as_vector(node.position_m)
            + (
                as_vector(node.motion.velocity_m_per_s) * sim_time_s
                if translating_motion
                else np.zeros(3, dtype=np.float64)
            )
        )
        if node.motion.kind is MotionKind.CALIBRATION_TUMBLE:
            runtime.orientation_world_to_sensor_wxyz = haar_pose_quaternion(
                runtime.sequence_id,
                runtime.haar_shifts,
            )
        elif node.motion.kind in {MotionKind.HIGH_DYNAMIC, MotionKind.COMBINED_STRESS}:
            modulated_rate = tuple(
                rate
                * (
                    1.0
                    + 0.25
                    * sin(
                        2.0
                        * pi
                        * node.motion.modulation_frequency_Hz[index]
                        * sim_time_s
                    )
                )
                for index, rate in enumerate(node.motion.angular_rate_rad_per_s)
            )
            runtime.orientation_world_to_sensor_wxyz = (
                integrate_world_to_sensor_quaternion(
                    runtime.orientation_world_to_sensor_wxyz,
                    modulated_rate,  # type: ignore[arg-type]
                    dt_s,
                )
            )

        random_walk_scale = np.sqrt(as_vector(errors.random_walk_q_T2_per_s) * dt_s)
        runtime.random_walk_bias_T += random_walk_scale * runtime.random_walk_rng.normal(
            size=3
        )
        runtime.correlated_noise_T = _ou_update(
            runtime.correlated_noise_T,
            as_vector(errors.ou_sigma_T),
            errors.ou_tau_s,
            dt_s,
            runtime.ou_rng,
        )
        if dt_s > 0.0:
            thermal_alpha = 1.0 - exp(-dt_s / errors.thermal_time_constant_s)
            runtime.device_temperature_K += thermal_alpha * (
                _ambient_temperature_target_K(errors, sim_time_s)
                - runtime.device_temperature_K
            )

        self._advance_random_faults(runtime, errors, sim_time_s, dt_s)
        targeted_events = tuple(
            event
            for event in active_events
            if node.sensor_id in event.target_sensor_ids
        )
        dropout_active = sim_time_s < runtime.dropout_until_s or any(
            event.kind is EventKind.DROPOUT for event in targeted_events
        )
        stuck_active = sim_time_s < runtime.stuck_until_s or any(
            event.kind is EventKind.STUCK for event in targeted_events
        )

        evaluation = evaluate_environment(
            configuration=self.configuration.environment,
            position_m=runtime.position_m,
            sim_time_s=sim_time_s,
            common_field_world_T=self._common_field_T,
            event_field_world_T=world_event_field_T,
        )
        rotation = quaternion_world_to_sensor_matrix(
            runtime.orientation_world_to_sensor_wxyz
        )
        ideal_sensor: NDArray[np.float64] | None = None
        if evaluation.field_true_world_T is not None:
            ideal_sensor = rotation @ evaluation.field_true_world_T

        quality: set[QualityFlag] = set()
        if errors.clock_offset_s != 0.0 or errors.clock_drift_ppm != 0.0:
            quality.add(QualityFlag.CLOCK_ERROR)

        value_T: float | None = None
        components_T: Vector3 | None = None
        saturation_mask: Bool3 | None = None
        if ideal_sensor is None:
            quality.add(QualityFlag.SIGNAL_ABSENT)
            # Consume the independent white stream even during an invalid frame.
            runtime.white_rng.normal(size=3)
        else:
            measured_vector = self._apply_node_response(
                node,
                runtime,
                ideal_sensor,
                sim_time_s,
                dt_s,
                targeted_events,
                quality,
            )
            value_T, components_T, saturation_mask, clipped = self._project_and_clip(
                node,
                measured_vector,
            )
            if clipped:
                quality.add(QualityFlag.CLIPPED)

        if stuck_active:
            clock_flags = {flag for flag in quality if flag is QualityFlag.CLOCK_ERROR}
            quality = set(runtime.last_quality_flags) | clock_flags | {QualityFlag.STUCK}
            value_T = runtime.last_value_T
            components_T = runtime.last_components_T
            saturation_mask = runtime.last_saturation_mask
            if value_T is None and components_T is None:
                quality.add(QualityFlag.SIGNAL_ABSENT)
        if dropout_active:
            quality = {flag for flag in quality if flag is QualityFlag.CLOCK_ERROR}
            quality.add(QualityFlag.SIGNAL_ABSENT)
            value_T = None
            components_T = None
            saturation_mask = None

        invalid_flags = {
            QualityFlag.CLIPPED,
            QualityFlag.SIGNAL_ABSENT,
            QualityFlag.STUCK,
        }
        valid = value_T is not None or components_T is not None
        valid = valid and not bool(quality & invalid_flags)
        if not dropout_active and not stuck_active and (
            value_T is not None or components_T is not None
        ):
            runtime.last_value_T = value_T
            runtime.last_components_T = components_T
            runtime.last_saturation_mask = saturation_mask
            runtime.last_quality_flags = tuple(
                sorted(quality, key=lambda flag: flag.value)
            )

        clock_scale = 1.0 + errors.clock_drift_ppm * 1.0e-6
        acquisition_time = session_epoch_utc + timedelta(
            seconds=errors.clock_offset_s + sim_time_s * clock_scale
        )
        reading = SensorReading(
            sensor_id=node.sensor_id,
            sequence_id=runtime.sequence_id,
            acquisition_time=acquisition_time,
            # Transport latency is zero in this local demonstrator. Keeping the
            # timestamp tied to simulated time makes reset/replay byte-stable.
            arrival_time=session_epoch_utc + timedelta(seconds=sim_time_s),
            measurement_mode=node.measurement_mode,
            value_T=value_T,
            components_T=components_T,
            saturation_mask=saturation_mask,
            quality_flags=tuple(sorted(quality, key=lambda flag: flag.value)),
            valid=valid,
            observed_temperature_K=runtime.device_temperature_K,
            position_m=runtime.position_m,
            orientation_world_to_sensor_wxyz=(
                runtime.orientation_world_to_sensor_wxyz
            ),
            calibration_version=node.calibration_version,
        )
        field_truth = FieldTruthAtNode(
            sensor_id=node.sensor_id,
            position_m=runtime.position_m,
            uniform_field_world_T=_tuple3(evaluation.uniform_field_world_T),
            common_field_world_T=_tuple3(evaluation.common_field_world_T),
            gradient_field_world_T=_tuple3(evaluation.gradient_field_world_T),
            dipole_field_world_T=(
                _tuple3(evaluation.dipole_field_world_T)
                if evaluation.dipole_field_world_T is not None
                else None
            ),
            anomaly_field_world_T=_tuple3(evaluation.anomaly_field_world_T),
            periodic_field_world_T=_tuple3(evaluation.periodic_field_world_T),
            event_field_world_T=_tuple3(evaluation.event_field_world_T),
            field_true_world_T=(
                _tuple3(evaluation.field_true_world_T)
                if evaluation.field_true_world_T is not None
                else None
            ),
            ideal_field_sensor_T=(
                _tuple3(ideal_sensor) if ideal_sensor is not None else None
            ),
            model_valid=evaluation.invalid_dipole is None,
        )
        device_truth = DeviceTruth(
            sensor_id=node.sensor_id,
            random_walk_bias_T=_tuple3(runtime.random_walk_bias_T),
            correlated_noise_T=_tuple3(runtime.correlated_noise_T),
            device_temperature_K=runtime.device_temperature_K,
            dropout_active=dropout_active,
            stuck_active=stuck_active,
            active_instrument_event_ids=tuple(
                event.event_id for event in targeted_events
            ),
        )
        return reading, field_truth, device_truth

    def _advance_random_faults(
        self,
        runtime: _NodeRuntime,
        errors: NodeErrorConfiguration,
        sim_time_s: float,
        dt_s: float,
    ) -> None:
        if dt_s <= 0.0:
            return
        if sim_time_s >= runtime.dropout_until_s:
            probability = 1.0 - exp(-errors.dropout_rate_per_s * dt_s)
            if runtime.dropout_rng.random() < probability:
                runtime.dropout_until_s = sim_time_s + errors.dropout_duration_s
        if sim_time_s >= runtime.stuck_until_s:
            probability = 1.0 - exp(-errors.stuck_rate_per_s * dt_s)
            if runtime.stuck_rng.random() < probability:
                runtime.stuck_until_s = sim_time_s + errors.stuck_duration_s

    def _apply_node_response(
        self,
        node: SensorNodeConfiguration,
        runtime: _NodeRuntime,
        ideal_sensor_T: NDArray[np.float64],
        sim_time_s: float,
        dt_s: float,
        active_events: tuple[NetworkEventConfiguration, ...],
        quality: set[QualityFlag],
    ) -> NDArray[np.float64]:
        errors = node.errors
        response = (
            np.asarray(errors.cross_axis_matrix, dtype=np.float64)
            @ np.asarray(errors.gain_matrix, dtype=np.float64)
            @ np.asarray(errors.soft_iron_matrix, dtype=np.float64)
            @ ideal_sensor_T
        )
        event_offset = np.zeros(3, dtype=np.float64)
        event_drift = np.zeros(3, dtype=np.float64)
        noise_multiplier = 1.0
        for event in active_events:
            if event.kind in {
                EventKind.NODE_BIAS,
                EventKind.SHARED_INSTRUMENT_OFFSET,
            }:
                event_offset += as_vector(event.field_offset_T)
            elif event.kind is EventKind.NODE_DRIFT:
                event_drift += as_vector(event.drift_rate_T_per_s) * (
                    sim_time_s - event.start_time_s
                )
            elif event.kind is EventKind.NODE_NOISE_BURST:
                noise_multiplier *= event.noise_multiplier

        thermal_bias = as_vector(errors.thermal_bias_T_per_K) * (
            runtime.device_temperature_K - errors.reference_temperature_K
        )
        deterministic_sensor_response = (
            response
            + as_vector(errors.bias_T)
            + as_vector(errors.deterministic_drift_T_per_s) * sim_time_s
            + runtime.random_walk_bias_T
            + runtime.correlated_noise_T
            + thermal_bias
            + event_offset
            + event_drift
        )
        white_sigma = as_vector(errors.white_noise_std_T_per_sample) * noise_multiplier
        white_noise = white_sigma * runtime.white_rng.normal(size=3)
        pre_filter_measurement = deterministic_sensor_response + white_noise

        if errors.bandwidth_Hz is None or runtime.filtered_signal_T is None:
            runtime.filtered_signal_T = pre_filter_measurement.copy()
        elif dt_s > 0.0:
            coefficient = 1.0 - exp(-2.0 * pi * errors.bandwidth_Hz * dt_s)
            runtime.filtered_signal_T += coefficient * (
                pre_filter_measurement - runtime.filtered_signal_T
            )

        return runtime.filtered_signal_T

    @staticmethod
    def _project_and_clip(
        node: SensorNodeConfiguration,
        measured_vector_T: NDArray[np.float64],
    ) -> tuple[float | None, Vector3 | None, Bool3 | None, bool]:
        if node.measurement_mode is MeasurementMode.VECTOR:
            limits = np.asarray(
                node.errors.saturation_limits_T
                or (node.errors.saturation_limit_T,) * 3,
                dtype=np.float64,
            )
            clipped = np.clip(measured_vector_T, -limits, limits)
            mask = tuple(
                bool(abs(value) > limits[index])
                for index, value in enumerate(measured_vector_T)
            )
            changed = any(mask)
            return None, _tuple3(clipped), mask, changed  # type: ignore[return-value]
        limit = node.errors.saturation_limit_T
        if node.measurement_mode is MeasurementMode.MONOAXIAL:
            raw_value = float(as_vector(node.monoaxial_axis_sensor) @ measured_vector_T)
            value = float(np.clip(raw_value, -limit, limit))
            return value, None, None, value != raw_value

        raw_value = float(np.linalg.norm(measured_vector_T))
        value = min(raw_value, limit)
        return value, None, None, value != raw_value
