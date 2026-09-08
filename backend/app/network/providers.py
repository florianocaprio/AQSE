from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from app.network.models import (
    DipoleSourceConfiguration,
    EnvironmentConfiguration,
    FieldProviderCatalog,
    FieldProviderStatus,
    GaussianAnomalyConfiguration,
    Vector3,
)

MU0_OVER_4PI = 1.0e-7


class DipoleDomainError(ValueError):
    """Raised when the point-dipole approximation is outside its domain."""

    def __init__(self, source_id: str, distance_m: float, minimum_distance_m: float):
        super().__init__(
            f"distance {distance_m:g} m from dipole {source_id} is at or below "
            f"the declared minimum {minimum_distance_m:g} m"
        )
        self.source_id = source_id
        self.distance_m = distance_m
        self.minimum_distance_m = minimum_distance_m


@dataclass(frozen=True)
class ProviderEvaluation:
    field_world_T: NDArray[np.float64] | None
    uniform_field_world_T: NDArray[np.float64]
    gradient_field_world_T: NDArray[np.float64]
    dipole_field_world_T: NDArray[np.float64] | None
    anomaly_field_world_T: NDArray[np.float64]
    periodic_field_world_T: NDArray[np.float64]
    invalid_dipole: DipoleDomainError | None = None


@runtime_checkable
class FieldProvider(Protocol):
    provider_id: str

    def evaluate(
        self,
        position_m: Vector3 | NDArray[np.float64],
        sim_time_s: float,
    ) -> ProviderEvaluation: ...


def _vector(values: Vector3 | NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def dipole_position_m(
    source: DipoleSourceConfiguration,
    sim_time_s: float,
) -> NDArray[np.float64]:
    return _vector(source.initial_position_m) + sim_time_s * _vector(
        source.velocity_m_per_s
    )


def point_dipole_field_T(
    sensor_position_m: Vector3 | NDArray[np.float64],
    source_position_m: Vector3 | NDArray[np.float64],
    moment_A_m2: Vector3 | NDArray[np.float64],
    minimum_distance_m: float,
    source_id: str = "dipole",
) -> NDArray[np.float64]:
    """Evaluate the quasistatic point-dipole field without singularity clamps."""

    displacement = _vector(sensor_position_m) - _vector(source_position_m)
    distance = float(np.linalg.norm(displacement))
    if distance <= minimum_distance_m:
        raise DipoleDomainError(source_id, distance, minimum_distance_m)
    direction = displacement / distance
    moment = _vector(moment_A_m2)
    return (MU0_OVER_4PI / distance**3) * (
        3.0 * direction * float(moment @ direction) - moment
    )


def gaussian_anomaly_field_T(
    sensor_position_m: Vector3 | NDArray[np.float64],
    anomaly: GaussianAnomalyConfiguration,
) -> NDArray[np.float64]:
    """Evaluate the approved smooth phenomenological world-frame anomaly."""

    displacement = _vector(sensor_position_m) - _vector(anomaly.center_position_m)
    exponent = -float(displacement @ displacement) / (
        2.0 * anomaly.spatial_scale_m**2
    )
    return (
        anomaly.peak_amplitude_T
        * np.exp(exponent)
        * _vector(anomaly.direction_world)
    )


class ConstantLocalFieldProvider:
    provider_id = "constant_local_field"

    def __init__(self, configuration: EnvironmentConfiguration) -> None:
        self._configuration = configuration

    def evaluate(
        self,
        position_m: Vector3 | NDArray[np.float64],
        sim_time_s: float,
    ) -> ProviderEvaluation:
        del position_m, sim_time_s
        uniform = _vector(self._configuration.uniform_field_T)
        zero = np.zeros(3, dtype=np.float64)
        return ProviderEvaluation(
            field_world_T=uniform,
            uniform_field_world_T=uniform,
            gradient_field_world_T=zero.copy(),
            dipole_field_world_T=zero.copy(),
            anomaly_field_world_T=zero.copy(),
            periodic_field_world_T=zero.copy(),
        )


class SyntheticSpatialFieldProvider:
    provider_id = "synthetic_spatial_field"

    def __init__(self, configuration: EnvironmentConfiguration) -> None:
        self._configuration = configuration

    def evaluate(
        self,
        position_m: Vector3 | NDArray[np.float64],
        sim_time_s: float,
    ) -> ProviderEvaluation:
        position = _vector(position_m)
        reference = _vector(self._configuration.gradient_reference_position_m)
        gradient = (
            np.asarray(self._configuration.gradient_T_per_m, dtype=np.float64)
            @ (position - reference)
        )
        anomaly_total = sum(
            (
                gaussian_anomaly_field_T(position, anomaly)
                for anomaly in self._configuration.gaussian_anomalies
                if anomaly.enabled
            ),
            start=np.zeros(3, dtype=np.float64),
        )
        dipole_total = np.zeros(3, dtype=np.float64)
        periodic_total = sum(
            (
                _vector(source.amplitude_world_T)
                * sin(2.0 * pi * source.frequency_Hz * sim_time_s + source.phase_rad)
                for source in self._configuration.periodic_fields
                if source.enabled
            ),
            start=np.zeros(3, dtype=np.float64),
        )
        for source in self._configuration.dipoles:
            if not source.enabled:
                continue
            try:
                dipole_total += point_dipole_field_T(
                    sensor_position_m=position,
                    source_position_m=dipole_position_m(source, sim_time_s),
                    moment_A_m2=source.moment_A_m2,
                    minimum_distance_m=source.minimum_distance_m,
                    source_id=source.source_id,
                )
            except DipoleDomainError as exc:
                return ProviderEvaluation(
                    field_world_T=None,
                    uniform_field_world_T=np.zeros(3, dtype=np.float64),
                    gradient_field_world_T=gradient,
                    dipole_field_world_T=None,
                    anomaly_field_world_T=anomaly_total,
                    periodic_field_world_T=periodic_total,
                    invalid_dipole=exc,
                )
        zero = np.zeros(3, dtype=np.float64)
        return ProviderEvaluation(
            field_world_T=gradient + dipole_total + anomaly_total + periodic_total,
            uniform_field_world_T=zero,
            gradient_field_world_T=gradient,
            dipole_field_world_T=dipole_total,
            anomaly_field_world_T=anomaly_total,
            periodic_field_world_T=periodic_total,
        )


class WorldMagneticModelProvider:
    """Reserved provider boundary; no WMM coefficients are bundled in AQSE."""

    provider_id = "world_magnetic_model"
    available = False
    configured = False

    def evaluate(
        self,
        position_m: Vector3 | NDArray[np.float64],
        sim_time_s: float,
    ) -> ProviderEvaluation:
        del position_m, sim_time_s
        raise RuntimeError("World Magnetic Model provider is not configured")


def field_provider_catalog() -> FieldProviderCatalog:
    return FieldProviderCatalog(
        providers=(
            FieldProviderStatus(
                provider_id=ConstantLocalFieldProvider.provider_id,
                display_name="Constant local field",
                available=True,
                configured=True,
                description="Explicit local NED field in tesla; not labelled as WMM.",
            ),
            FieldProviderStatus(
                provider_id=SyntheticSpatialFieldProvider.provider_id,
                display_name="Synthetic spatial field",
                available=True,
                configured=True,
                description=(
                    "Gradient, point-dipole, Gaussian and periodic synthetic providers."
                ),
            ),
            FieldProviderStatus(
                provider_id=WorldMagneticModelProvider.provider_id,
                display_name="World Magnetic Model",
                available=False,
                configured=False,
                description="Future adapter boundary; coefficients are not configured.",
            ),
        )
    )
