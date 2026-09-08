from __future__ import annotations

from datetime import datetime
from enum import Enum
from math import isclose, sqrt
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"
MIN_NETWORK_NODES = 1
MAX_NETWORK_NODES = 8
MAX_BUFFER_FRAMES = 20_000
MAX_EVENTS = 128
MAX_FINITE_SIMULATION_SAMPLES = 10_000

Vector3 = tuple[float, float, float]
Bool3 = tuple[bool, bool, bool]
Matrix3 = tuple[Vector3, Vector3, Vector3]
QuaternionWXYZ = tuple[float, float, float, float]

ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)
IDENTITY_MATRIX: Matrix3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
IDENTITY_QUATERNION: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MeasurementMode(str, Enum):
    VECTOR = "vector"
    MONOAXIAL = "monoaxial"
    TOTAL_FIELD = "total_field"


class SessionState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class EventKind(str, Enum):
    WORLD_FIELD_OFFSET = "world_field_offset"
    NODE_BIAS = "node_bias"
    NODE_DRIFT = "node_drift"
    NODE_NOISE_BURST = "node_noise_burst"
    SHARED_INSTRUMENT_OFFSET = "shared_instrument_offset"
    DROPOUT = "dropout"
    STUCK = "stuck"


class VectorScenario(str, Enum):
    STATIC_REFERENCE = "static_reference"
    CALIBRATION_TUMBLE = "calibration_tumble"
    HIGH_DYNAMIC = "high_dynamic"
    LINEAR_TRANSLATION = "linear_translation"
    LOCAL_ANOMALY_CROSSING = "local_anomaly_crossing"
    COMBINED_STRESS = "combined_stress"


class MotionKind(str, Enum):
    STATIC = "static"
    CALIBRATION_TUMBLE = "calibration_tumble"
    HIGH_DYNAMIC = "high_dynamic"
    LINEAR_TRANSLATION = "linear_translation"
    LOCAL_ANOMALY_CROSSING = "local_anomaly_crossing"
    COMBINED_STRESS = "combined_stress"


class QualityFlag(str, Enum):
    CLIPPED = "clipped"
    SIGNAL_ABSENT = "signal_absent"
    STUCK = "stuck"
    CLOCK_ERROR = "clock_error"


class TemperatureDriverKind(str, Enum):
    CONSTANT = "constant"
    RAMP = "ramp"
    SINUSOIDAL = "sinusoidal"


class DipoleSourceConfiguration(StrictModel):
    source_id: str = Field(min_length=1, max_length=128)
    initial_position_m: Vector3
    velocity_m_per_s: Vector3 = ZERO_VECTOR
    moment_A_m2: Vector3
    minimum_distance_m: float = Field(default=0.05, ge=1.0e-9, le=1.0e9)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_computational_bounds(self) -> DipoleSourceConfiguration:
        if max(abs(value) for value in self.initial_position_m) > 1.0e9:
            raise ValueError("initial_position_m exceeds the local demonstrator bound")
        if max(abs(value) for value in self.velocity_m_per_s) > 1.0e6:
            raise ValueError("velocity_m_per_s exceeds the local demonstrator bound")
        if max(abs(value) for value in self.moment_A_m2) > 1.0e12:
            raise ValueError("moment_A_m2 exceeds the local demonstrator bound")
        return self


class GaussianAnomalyConfiguration(StrictModel):
    anomaly_id: str = Field(min_length=1, max_length=128)
    peak_amplitude_T: float = Field(gt=0.0, le=10.0)
    direction_world: Vector3 = (0.0, 0.0, 1.0)
    center_position_m: Vector3 = ZERO_VECTOR
    spatial_scale_m: float = Field(ge=1.0e-9, le=1.0e9)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_direction(self) -> GaussianAnomalyConfiguration:
        norm = sqrt(sum(value * value for value in self.direction_world))
        if not isclose(norm, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9):
            raise ValueError("direction_world must be a unit vector")
        if max(abs(value) for value in self.center_position_m) > 1.0e9:
            raise ValueError("center_position_m exceeds the local demonstrator bound")
        return self


class PeriodicFieldConfiguration(StrictModel):
    source_id: str = Field(min_length=1, max_length=128)
    amplitude_world_T: Vector3
    frequency_Hz: float = Field(gt=0.0, le=1_000.0)
    phase_rad: float = Field(default=0.0, ge=-1.0e6, le=1.0e6)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_amplitude(self) -> PeriodicFieldConfiguration:
        if max(abs(value) for value in self.amplitude_world_T) > 10.0:
            raise ValueError("amplitude_world_T exceeds the local demonstrator bound")
        return self


class EnvironmentConfiguration(StrictModel):
    uniform_field_T: Vector3 = (20.0e-6, 0.0, 45.0e-6)
    gradient_reference_position_m: Vector3 = ZERO_VECTOR
    gradient_T_per_m: Matrix3 = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    common_ou_sigma_T: Vector3 = ZERO_VECTOR
    common_ou_tau_s: float = Field(default=1.0, gt=0.0, le=1.0e9)
    dipoles: tuple[DipoleSourceConfiguration, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )
    gaussian_anomalies: tuple[GaussianAnomalyConfiguration, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )
    periodic_fields: tuple[PeriodicFieldConfiguration, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_magnetostatic_gradient(self) -> EnvironmentConfiguration:
        matrix = self.gradient_T_per_m
        magnitude = max(abs(value) for row in matrix for value in row)
        tolerance = max(1.0e-18, magnitude * 1.0e-9)
        for row in range(3):
            for column in range(3):
                if not isclose(
                    matrix[row][column],
                    matrix[column][row],
                    rel_tol=1.0e-9,
                    abs_tol=tolerance,
                ):
                    raise ValueError(
                        "gradient_T_per_m must be symmetric for the declared "
                        "source-free magnetostatic model"
                    )
        if abs(sum(matrix[index][index] for index in range(3))) > tolerance:
            raise ValueError(
                "gradient_T_per_m must be traceless for the declared "
                "source-free magnetostatic model"
            )
        if any(value < 0.0 for value in self.common_ou_sigma_T):
            raise ValueError("common_ou_sigma_T components cannot be negative")
        source_ids = [source.source_id for source in self.dipoles]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("dipole source_id values must be unique")
        anomaly_ids = [anomaly.anomaly_id for anomaly in self.gaussian_anomalies]
        if len(anomaly_ids) != len(set(anomaly_ids)):
            raise ValueError("Gaussian anomaly_id values must be unique")
        periodic_ids = [source.source_id for source in self.periodic_fields]
        if len(periodic_ids) != len(set(periodic_ids)):
            raise ValueError("periodic field source_id values must be unique")
        if max(abs(value) for value in self.uniform_field_T) > 10.0:
            raise ValueError("uniform_field_T exceeds the local demonstrator bound")
        if max(abs(value) for value in self.gradient_reference_position_m) > 1.0e9:
            raise ValueError(
                "gradient_reference_position_m exceeds the local demonstrator bound"
            )
        if max(abs(value) for row in self.gradient_T_per_m for value in row) > 10.0:
            raise ValueError("gradient_T_per_m exceeds the local demonstrator bound")
        if max(abs(value) for value in self.common_ou_sigma_T) > 10.0:
            raise ValueError("common_ou_sigma_T exceeds the local demonstrator bound")
        return self


class TemperatureDriverConfiguration(StrictModel):
    """Optional ambient-temperature variation around the legacy constant target."""

    kind: TemperatureDriverKind = TemperatureDriverKind.CONSTANT
    ramp_rate_K_per_s: float = Field(default=0.0, ge=-1_000.0, le=1_000.0)
    ramp_duration_s: float = Field(default=1.0, gt=0.0, le=1.0e9)
    sinusoidal_amplitude_K: float = Field(default=0.0, ge=0.0, le=2_500.0)
    sinusoidal_frequency_Hz: float = Field(default=0.1, gt=0.0, le=1_000.0)
    sinusoidal_phase_rad: float = Field(default=0.0, ge=-1.0e6, le=1.0e6)


class NodeErrorConfiguration(StrictModel):
    gain_matrix: Matrix3 = IDENTITY_MATRIX
    soft_iron_matrix: Matrix3 = IDENTITY_MATRIX
    cross_axis_matrix: Matrix3 = IDENTITY_MATRIX
    bias_T: Vector3 = ZERO_VECTOR
    deterministic_drift_T_per_s: Vector3 = ZERO_VECTOR
    white_noise_std_T_per_sample: Vector3 = ZERO_VECTOR
    random_walk_q_T2_per_s: Vector3 = ZERO_VECTOR
    ou_sigma_T: Vector3 = ZERO_VECTOR
    ou_tau_s: float = Field(default=1.0, gt=0.0, le=1.0e9)
    initial_temperature_K: float = Field(default=293.15, gt=0.0, le=5_000.0)
    ambient_temperature_K: float = Field(default=293.15, gt=0.0, le=5_000.0)
    temperature_driver: TemperatureDriverConfiguration | None = None
    thermal_time_constant_s: float = Field(default=5.0, gt=0.0, le=1.0e9)
    thermal_bias_T_per_K: Vector3 = ZERO_VECTOR
    reference_temperature_K: float = Field(default=293.15, gt=0.0, le=5_000.0)
    bandwidth_Hz: float | None = Field(default=None, gt=0.0, le=1_000.0)
    saturation_limit_T: float = Field(default=800.0e-6, gt=0.0, le=10.0)
    saturation_limits_T: Vector3 | None = None
    clock_offset_s: float = Field(default=0.0, ge=-86_400.0, le=86_400.0)
    clock_drift_ppm: float = Field(default=0.0, ge=-1.0e6, le=1.0e6)
    dropout_rate_per_s: float = Field(default=0.0, ge=0.0, le=1.0e6)
    dropout_duration_s: float = Field(default=0.5, gt=0.0, le=1.0e9)
    stuck_rate_per_s: float = Field(default=0.0, ge=0.0, le=1.0e6)
    stuck_duration_s: float = Field(default=0.5, gt=0.0, le=1.0e9)

    @model_validator(mode="after")
    def validate_nonnegative_components(self) -> NodeErrorConfiguration:
        named_vectors = {
            "white_noise_std_T_per_sample": self.white_noise_std_T_per_sample,
            "random_walk_q_T2_per_s": self.random_walk_q_T2_per_s,
            "ou_sigma_T": self.ou_sigma_T,
        }
        for name, values in named_vectors.items():
            if any(value < 0.0 for value in values):
                raise ValueError(f"{name} components cannot be negative")
        bounded_vectors = {
            "bias_T": self.bias_T,
            "deterministic_drift_T_per_s": self.deterministic_drift_T_per_s,
            "white_noise_std_T_per_sample": self.white_noise_std_T_per_sample,
            "ou_sigma_T": self.ou_sigma_T,
            "thermal_bias_T_per_K": self.thermal_bias_T_per_K,
        }
        for name, values in bounded_vectors.items():
            if max(abs(value) for value in values) > 10.0:
                raise ValueError(f"{name} exceeds the local demonstrator bound")
        if max(self.random_walk_q_T2_per_s) > 100.0:
            raise ValueError("random_walk_q_T2_per_s exceeds the local demonstrator bound")
        if self.saturation_limits_T is not None:
            if any(value <= 0.0 for value in self.saturation_limits_T):
                raise ValueError("saturation_limits_T components must be positive")
            if max(self.saturation_limits_T) > 10.0:
                raise ValueError("saturation_limits_T exceeds the local demonstrator bound")
        if self.temperature_driver is not None:
            driver = self.temperature_driver
            if driver.kind is TemperatureDriverKind.RAMP:
                final_temperature = (
                    self.ambient_temperature_K
                    + driver.ramp_rate_K_per_s * driver.ramp_duration_s
                )
                if not 0.0 < final_temperature <= 5_000.0:
                    raise ValueError(
                        "the bounded ambient-temperature ramp must remain within "
                        "(0, 5000] K"
                    )
            elif driver.kind is TemperatureDriverKind.SINUSOIDAL:
                lower = self.ambient_temperature_K - driver.sinusoidal_amplitude_K
                upper = self.ambient_temperature_K + driver.sinusoidal_amplitude_K
                if lower <= 0.0 or upper > 5_000.0:
                    raise ValueError(
                        "the sinusoidal ambient-temperature driver must remain within "
                        "(0, 5000] K"
                    )
        for name, matrix in (
            ("gain_matrix", self.gain_matrix),
            ("soft_iron_matrix", self.soft_iron_matrix),
            ("cross_axis_matrix", self.cross_axis_matrix),
        ):
            if max(abs(value) for row in matrix for value in row) > 1.0e6:
                raise ValueError(f"{name} exceeds the local demonstrator bound")
        soft_iron = np.asarray(self.soft_iron_matrix, dtype=np.float64)
        if not np.allclose(soft_iron, soft_iron.T, rtol=1.0e-9, atol=1.0e-12):
            raise ValueError("soft_iron_matrix must be symmetric")
        if float(np.min(np.linalg.eigvalsh(soft_iron))) <= 0.0:
            raise ValueError("soft_iron_matrix must be positive definite")
        if float(np.linalg.cond(soft_iron)) > 100.0:
            raise ValueError("soft_iron_matrix condition number cannot exceed 100")
        for name, matrix in (
            ("gain_matrix", self.gain_matrix),
            ("cross_axis_matrix", self.cross_axis_matrix),
        ):
            response = np.asarray(matrix, dtype=np.float64)
            if abs(float(np.linalg.det(response))) <= np.finfo(np.float64).eps:
                raise ValueError(f"{name} must be invertible")
            if float(np.linalg.cond(response)) > 100.0:
                raise ValueError(f"{name} condition number cannot exceed 100")
        gain = np.asarray(self.gain_matrix, dtype=np.float64)
        if not np.allclose(gain, np.diag(np.diag(gain)), rtol=0.0, atol=1.0e-15):
            raise ValueError("gain_matrix must be diagonal")
        if np.any(np.diag(gain) <= 0.0):
            raise ValueError("gain_matrix diagonal entries must be positive")
        combined = (
            np.asarray(self.cross_axis_matrix, dtype=np.float64)
            @ np.asarray(self.gain_matrix, dtype=np.float64)
            @ soft_iron
        )
        if float(np.linalg.cond(combined)) > 100.0:
            raise ValueError("combined linear response condition number cannot exceed 100")
        return self


class NodeMotionConfiguration(StrictModel):
    kind: MotionKind = MotionKind.STATIC
    velocity_m_per_s: Vector3 = ZERO_VECTOR
    angular_rate_rad_per_s: Vector3 = ZERO_VECTOR
    modulation_frequency_Hz: Vector3 = (0.31, 0.47, 0.59)

    @model_validator(mode="after")
    def validate_motion(self) -> NodeMotionConfiguration:
        if any(frequency < 0.0 for frequency in self.modulation_frequency_Hz):
            raise ValueError("modulation_frequency_Hz components cannot be negative")
        if max(abs(value) for value in self.velocity_m_per_s) > 1.0e6:
            raise ValueError("velocity_m_per_s exceeds the local demonstrator bound")
        if max(abs(value) for value in self.angular_rate_rad_per_s) > 1.0e6:
            raise ValueError("angular_rate_rad_per_s exceeds the local demonstrator bound")
        if max(self.modulation_frequency_Hz) > 1_000.0:
            raise ValueError("modulation_frequency_Hz exceeds the local demonstrator bound")
        return self


class SensorNodeConfiguration(StrictModel):
    sensor_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    role: Literal["sensor", "remote_reference"] = "sensor"
    profile: Literal["synthetic_magnetometer"] = "synthetic_magnetometer"
    measurement_mode: MeasurementMode = MeasurementMode.VECTOR
    position_m: Vector3
    orientation_world_to_sensor_wxyz: QuaternionWXYZ = IDENTITY_QUATERNION
    monoaxial_axis_sensor: Vector3 = (1.0, 0.0, 0.0)
    calibration_version: str = Field(default="synthetic-calibration-v1", min_length=1)
    motion: NodeMotionConfiguration = Field(default_factory=NodeMotionConfiguration)
    errors: NodeErrorConfiguration = Field(default_factory=NodeErrorConfiguration)

    @model_validator(mode="after")
    def validate_orientation_and_axis(self) -> SensorNodeConfiguration:
        quaternion_norm = sqrt(
            sum(value * value for value in self.orientation_world_to_sensor_wxyz)
        )
        if not isclose(quaternion_norm, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9):
            raise ValueError(
                "orientation_world_to_sensor_wxyz must be a normalized "
                "Hamilton quaternion in [w, x, y, z] order"
            )
        if self.orientation_world_to_sensor_wxyz[0] < 0.0:
            raise ValueError(
                "orientation_world_to_sensor_wxyz must use canonical sign w >= 0"
            )
        axis_norm = sqrt(sum(value * value for value in self.monoaxial_axis_sensor))
        if not isclose(axis_norm, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9):
            raise ValueError("monoaxial_axis_sensor must be a unit vector")
        if max(abs(value) for value in self.position_m) > 1.0e9:
            raise ValueError("position_m exceeds the local demonstrator bound")
        return self


class NetworkEventConfiguration(StrictModel):
    event_id: str = Field(
        default_factory=lambda: f"event-{uuid4().hex}",
        min_length=1,
        max_length=128,
    )
    kind: EventKind
    start_time_s: float = Field(ge=0.0, le=1.0e9)
    duration_s: float = Field(gt=0.0, le=1.0e9)
    target_sensor_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    field_offset_T: Vector3 = ZERO_VECTOR
    drift_rate_T_per_s: Vector3 = ZERO_VECTOR
    noise_multiplier: float = Field(default=1.0, ge=1.0, le=1.0e6)

    @model_validator(mode="after")
    def validate_event_scope(self) -> NetworkEventConfiguration:
        node_scoped = self.kind is not EventKind.WORLD_FIELD_OFFSET
        if node_scoped and not self.target_sensor_ids:
            raise ValueError(f"{self.kind.value} requires at least one target sensor")
        if not node_scoped and self.target_sensor_ids:
            raise ValueError("world_field_offset cannot target individual sensors")
        if (
            self.kind is EventKind.SHARED_INSTRUMENT_OFFSET
            and len(self.target_sensor_ids) < 2
        ):
            raise ValueError("shared_instrument_offset requires at least two target sensors")
        if (
            self.kind is EventKind.NODE_NOISE_BURST
            and self.noise_multiplier <= 1.0
        ):
            raise ValueError("node_noise_burst requires noise_multiplier greater than 1")
        if len(self.target_sensor_ids) != len(set(self.target_sensor_ids)):
            raise ValueError("target_sensor_ids cannot contain duplicates")
        if max(abs(value) for value in self.field_offset_T) > 10.0:
            raise ValueError("field_offset_T exceeds the local demonstrator bound")
        if max(abs(value) for value in self.drift_rate_T_per_s) > 10.0:
            raise ValueError("drift_rate_T_per_s exceeds the local demonstrator bound")
        return self


class NetworkSessionConfiguration(StrictModel):
    session_name: str = Field(default="AQSE sensor network", min_length=1, max_length=128)
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    sampling_rate_Hz: float = Field(default=100.0, ge=1.0e-3, le=2_000.0)
    ui_refresh_rate_Hz: float = Field(default=5.0, ge=1.0e-3, le=60.0)
    time_scale: float = Field(default=1.0, gt=0.0, le=100.0)
    buffer_duration_s: float = Field(default=120.0, gt=0.0, le=600.0)
    environment: EnvironmentConfiguration = Field(default_factory=EnvironmentConfiguration)
    nodes: tuple[SensorNodeConfiguration, ...] = Field(
        min_length=MIN_NETWORK_NODES,
        max_length=MAX_NETWORK_NODES,
    )
    events: tuple[NetworkEventConfiguration, ...] = Field(
        default_factory=tuple,
        max_length=MAX_EVENTS,
    )

    @property
    def sample_interval_s(self) -> float:
        return 1.0 / self.sampling_rate_Hz

    @property
    def buffer_capacity_frames(self) -> int:
        return max(1, int(round(self.sampling_rate_Hz * self.buffer_duration_s)))

    @model_validator(mode="after")
    def validate_network(self) -> NetworkSessionConfiguration:
        if self.ui_refresh_rate_Hz > self.sampling_rate_Hz:
            raise ValueError("ui_refresh_rate_Hz cannot exceed sampling_rate_Hz")
        if self.buffer_capacity_frames > MAX_BUFFER_FRAMES:
            raise ValueError(
                "sampling_rate_Hz and buffer_duration_s may retain at most "
                f"{MAX_BUFFER_FRAMES} frames"
            )

        sensor_ids = [node.sensor_id for node in self.nodes]
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError("sensor_id values must be unique within a session")
        known_sensors = set(sensor_ids)
        for event in self.events:
            unknown = set(event.target_sensor_ids) - known_sensors
            if unknown:
                raise ValueError(
                    f"event {event.event_id} targets unknown sensors: "
                    f"{', '.join(sorted(unknown))}"
                )
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique within a session")
        for node in self.nodes:
            if (
                node.errors.bandwidth_Hz is not None
                and node.errors.bandwidth_Hz >= self.sampling_rate_Hz / 2.0
            ):
                raise ValueError(
                    f"sensor {node.sensor_id} bandwidth_Hz must be below "
                    "the Nyquist frequency"
                )
            temperature_driver = node.errors.temperature_driver
            if (
                temperature_driver is not None
                and temperature_driver.kind is TemperatureDriverKind.SINUSOIDAL
                and temperature_driver.sinusoidal_frequency_Hz
                >= self.sampling_rate_Hz / 2.0
            ):
                raise ValueError(
                    f"sensor {node.sensor_id} temperature-driver frequency must be "
                    "below the Nyquist frequency"
                )
            for source in self.environment.dipoles:
                if not source.enabled:
                    continue
                distance = sqrt(
                    sum(
                        (node.position_m[index] - source.initial_position_m[index])
                        ** 2
                        for index in range(3)
                    )
                )
                if distance <= source.minimum_distance_m:
                    raise ValueError(
                        f"sensor {node.sensor_id} starts inside minimum distance "
                        f"of dipole {source.source_id}"
                    )
        for source in self.environment.periodic_fields:
            if source.enabled and source.frequency_Hz >= self.sampling_rate_Hz / 2.0:
                raise ValueError(
                    f"periodic field {source.source_id} frequency_Hz must be below "
                    "the Nyquist frequency"
                )
        return self


class NetworkUnits(StrictModel):
    magnetic_field: Literal["T"] = "T"
    distance: Literal["m"] = "m"
    time: Literal["s"] = "s"
    magnetic_moment: Literal["A·m²"] = "A·m²"
    gradient: Literal["T/m"] = "T/m"
    temperature: Literal["K"] = "K"
    orientation: Literal["dimensionless quaternion wxyz"] = (
        "dimensionless quaternion wxyz"
    )


class SensorReading(StrictModel):
    sensor_id: str
    sequence_id: int = Field(ge=0)
    acquisition_time: datetime
    arrival_time: datetime
    measurement_mode: MeasurementMode
    value_T: float | None = None
    components_T: Vector3 | None = None
    saturation_mask: Bool3 | None = None
    quality_flags: tuple[QualityFlag, ...] = Field(default_factory=tuple)
    valid: bool
    observed_temperature_K: float
    position_m: Vector3
    orientation_world_to_sensor_wxyz: QuaternionWXYZ
    calibration_version: str

    @model_validator(mode="after")
    def validate_measurement_shape(self) -> SensorReading:
        for name, timestamp in (
            ("acquisition_time", self.acquisition_time),
            ("arrival_time", self.arrival_time),
        ):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        missing = self.value_T is None and self.components_T is None
        if missing:
            if self.valid:
                raise ValueError("a missing reading cannot be valid")
            return self
        if self.measurement_mode is MeasurementMode.VECTOR:
            if self.components_T is None or self.value_T is not None:
                raise ValueError("vector readings require components_T only")
            if self.saturation_mask is None:
                raise ValueError("vector readings require a saturation_mask")
        elif self.value_T is None or self.components_T is not None:
            raise ValueError("scalar readings require value_T only")
        return self


class ObservationFrame(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session_id: str
    configuration_version: int = Field(ge=1)
    frame_id: int = Field(ge=1)
    sim_time_s: float = Field(ge=0.0)
    readings: tuple[SensorReading, ...]


class FieldTruthAtNode(StrictModel):
    sensor_id: str
    position_m: Vector3
    uniform_field_world_T: Vector3
    common_field_world_T: Vector3
    gradient_field_world_T: Vector3
    dipole_field_world_T: Vector3 | None
    anomaly_field_world_T: Vector3
    periodic_field_world_T: Vector3
    event_field_world_T: Vector3
    field_true_world_T: Vector3 | None
    ideal_field_sensor_T: Vector3 | None
    model_valid: bool


class DeviceTruth(StrictModel):
    sensor_id: str
    random_walk_bias_T: Vector3
    correlated_noise_T: Vector3
    device_temperature_K: float
    dropout_active: bool
    stuck_active: bool
    active_instrument_event_ids: tuple[str, ...]


class ActiveCause(StrictModel):
    event_id: str
    kind: EventKind
    target_sensor_ids: tuple[str, ...]


class TruthFrame(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session_id: str
    configuration_version: int = Field(ge=1)
    frame_id: int = Field(ge=1)
    sim_time_s: float = Field(ge=0.0)
    fields: tuple[FieldTruthAtNode, ...]
    devices: tuple[DeviceTruth, ...]
    active_causes: tuple[ActiveCause, ...]


class BufferedFrame(StrictModel):
    observation: ObservationFrame
    truth: TruthFrame


class BufferStatus(StrictModel):
    size: int = Field(ge=0)
    capacity: int = Field(gt=0)
    oldest_frame_id: int | None
    newest_frame_id: int | None
    overwritten_frames: int = Field(ge=0)


class SessionStatus(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session_id: str
    state: SessionState
    configuration_version: int = Field(ge=1)
    sim_time_s: float = Field(ge=0.0)
    latest_frame_id: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    simulation_lag_s: float = Field(ge=0.0)
    buffer: BufferStatus

    @model_validator(mode="after")
    def validate_timestamps(self) -> SessionStatus:
        for name, timestamp in (
            ("created_at", self.created_at),
            ("updated_at", self.updated_at),
        ):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        return self


class SessionView(StrictModel):
    status: SessionStatus
    configuration: NetworkSessionConfiguration


class StepRequest(StrictModel):
    frames: int = Field(default=1, ge=1, le=500)


class FrameBatch(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session_id: str
    from_frame_id: int | None
    to_frame_id: int | None
    gap_detected: bool
    frames: tuple[ObservationFrame, ...]
    status: SessionStatus


class TruthFrameBatch(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session_id: str
    from_frame_id: int | None
    to_frame_id: int | None
    gap_detected: bool
    frames: tuple[TruthFrame, ...]
    status: SessionStatus


class ObservationSnapshot(StrictModel):
    status: SessionStatus
    observation: ObservationFrame | None


class TruthSnapshot(StrictModel):
    status: SessionStatus
    truth: TruthFrame | None


class ScheduledEventResponse(StrictModel):
    event: NetworkEventConfiguration
    configuration_version: int
    effective_frame_id: int


class NetworkHealth(StrictModel):
    status: Literal["ok"] = "ok"
    service: Literal["AQSE Sensor Network"] = "AQSE Sensor Network"
    session_store: Literal["in_memory_bounded"] = "in_memory_bounded"
    streaming: Literal["sse"] = "sse"
    active_sessions: int = Field(ge=0)
    maximum_sessions: int = Field(gt=0)


class NetworkPresetDescriptor(StrictModel):
    preset_id: str
    display_name: str
    recommended_node_count: int = Field(ge=1, le=8)
    description: str


class NetworkPresetCatalog(StrictModel):
    presets: tuple[NetworkPresetDescriptor, ...]


class FieldProviderStatus(StrictModel):
    provider_id: str
    display_name: str
    available: bool
    configured: bool
    description: str


class FieldProviderCatalog(StrictModel):
    providers: tuple[FieldProviderStatus, ...]


class VectorScenarioDescriptor(StrictModel):
    scenario_id: VectorScenario
    display_name: str
    description: str


class VectorScenarioCatalog(StrictModel):
    scenarios: tuple[VectorScenarioDescriptor, ...]


class FiniteVectorSimulationConfiguration(StrictModel):
    scenario: VectorScenario = VectorScenario.STATIC_REFERENCE
    duration_s: float = Field(default=2.0, gt=0.0, le=300.0)
    sampling_rate_Hz: float = Field(default=100.0, ge=1.0e-3, le=2_000.0)
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    environment: EnvironmentConfiguration = Field(default_factory=EnvironmentConfiguration)
    node: SensorNodeConfiguration
    events: tuple[NetworkEventConfiguration, ...] = Field(
        default_factory=tuple,
        max_length=MAX_EVENTS,
    )

    @property
    def sample_count(self) -> int:
        return int(round(self.duration_s * self.sampling_rate_Hz))

    @model_validator(mode="after")
    def validate_finite_simulation(self) -> FiniteVectorSimulationConfiguration:
        if self.node.measurement_mode is not MeasurementMode.VECTOR:
            raise ValueError("finite vector simulation requires measurement_mode=vector")
        if self.sample_count < 16:
            raise ValueError("finite simulation must contain at least 16 samples")
        if self.sample_count > MAX_FINITE_SIMULATION_SAMPLES:
            raise ValueError(
                "finite simulation may contain at most "
                f"{MAX_FINITE_SIMULATION_SAMPLES} samples"
            )
        known_sensor = {self.node.sensor_id}
        for event in self.events:
            unknown = set(event.target_sensor_ids) - known_sensor
            if unknown:
                raise ValueError(
                    f"event {event.event_id} targets unknown sensors: "
                    f"{', '.join(sorted(unknown))}"
                )
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique within a finite simulation")
        for source in self.environment.dipoles:
            if not source.enabled:
                continue
            distance = sqrt(
                sum(
                    (self.node.position_m[index] - source.initial_position_m[index])
                    ** 2
                    for index in range(3)
                )
            )
            if distance <= source.minimum_distance_m:
                raise ValueError(
                    f"sensor {self.node.sensor_id} starts inside minimum distance "
                    f"of dipole {source.source_id}"
                )
        if (
            self.node.errors.bandwidth_Hz is not None
            and self.node.errors.bandwidth_Hz >= self.sampling_rate_Hz / 2.0
        ):
            raise ValueError("node bandwidth_Hz must be below the Nyquist frequency")
        temperature_driver = self.node.errors.temperature_driver
        if (
            temperature_driver is not None
            and temperature_driver.kind is TemperatureDriverKind.SINUSOIDAL
            and temperature_driver.sinusoidal_frequency_Hz
            >= self.sampling_rate_Hz / 2.0
        ):
            raise ValueError(
                "node temperature-driver frequency must be below the Nyquist frequency"
            )
        for source in self.environment.periodic_fields:
            if source.enabled and source.frequency_Hz >= self.sampling_rate_Hz / 2.0:
                raise ValueError(
                    f"periodic field {source.source_id} frequency_Hz must be below "
                    "the Nyquist frequency"
                )
        return self


class FiniteMeasurementSeries(StrictModel):
    sensor_id: str
    measurement_mode: Literal[MeasurementMode.VECTOR] = MeasurementMode.VECTOR
    time_s: tuple[float, ...]
    position_m: tuple[Vector3, ...]
    orientation_world_to_sensor_wxyz: tuple[QuaternionWXYZ, ...]
    temperature_K: tuple[float, ...]
    measured_field_T: tuple[Vector3 | None, ...]
    saturation_mask: tuple[Bool3, ...]
    valid: tuple[bool, ...]
    quality_flags: tuple[tuple[QualityFlag, ...], ...]


class FiniteGeneratorTruthSeries(StrictModel):
    time_s: tuple[float, ...]
    earth_field_world_T: tuple[Vector3, ...]
    environment_field_world_T: tuple[Vector3 | None, ...]
    anomaly_field_world_T: tuple[Vector3, ...]
    periodic_field_world_T: tuple[Vector3, ...]
    ideal_field_sensor_T: tuple[Vector3 | None, ...]
    active_causes: tuple[tuple[ActiveCause, ...], ...]


class FiniteVectorSimulationResponse(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    configuration: FiniteVectorSimulationConfiguration
    sample_count: int = Field(ge=16, le=MAX_FINITE_SIMULATION_SAMPLES)
    measurement: FiniteMeasurementSeries
    generator_truth: FiniteGeneratorTruthSeries


class ObservabilityRequest(StrictModel):
    configuration: NetworkSessionConfiguration
    source_id: str | None = None
    numerical_step: float = Field(default=1.0e-5, gt=0.0, le=1.0e-2)
    noise_floor_T: float = Field(default=1.0e-12, gt=0.0)
    condition_warning_threshold: float = Field(default=1.0e6, gt=1.0)


class ObservabilityDiagnostics(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    source_id: str
    parameter_names: tuple[str, ...]
    measurement_dimension: int = Field(ge=1)
    jacobian_shape: tuple[int, int]
    weighted_nondimensional_jacobian: tuple[tuple[float, ...], ...]
    singular_values: tuple[float, ...]
    numerical_rank: int = Field(ge=0)
    full_column_rank: bool
    condition_number: float | None
    parameter_scales: tuple[float, ...]
    noise_floor_T: float = Field(gt=0.0)
    warnings: tuple[str, ...]
    interpretation: str


NodeCount = Annotated[int, Field(ge=MIN_NETWORK_NODES, le=MAX_NETWORK_NODES)]
