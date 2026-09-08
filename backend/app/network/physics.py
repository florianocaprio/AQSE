from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.network.models import (
    EnvironmentConfiguration,
    QuaternionWXYZ,
    Vector3,
)
from app.network.providers import (
    ConstantLocalFieldProvider,
    DipoleDomainError,
    SyntheticSpatialFieldProvider,
)
from app.network.providers import (
    dipole_position_m as dipole_position_m,
)
from app.network.providers import (
    gaussian_anomaly_field_T as gaussian_anomaly_field_T,
)
from app.network.providers import (
    point_dipole_field_T as point_dipole_field_T,
)


@dataclass(frozen=True)
class EnvironmentEvaluation:
    uniform_field_world_T: NDArray[np.float64]
    common_field_world_T: NDArray[np.float64]
    gradient_field_world_T: NDArray[np.float64]
    dipole_field_world_T: NDArray[np.float64] | None
    anomaly_field_world_T: NDArray[np.float64]
    periodic_field_world_T: NDArray[np.float64]
    event_field_world_T: NDArray[np.float64]
    field_true_world_T: NDArray[np.float64] | None
    invalid_dipole: DipoleDomainError | None


def as_vector(values: Vector3) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def quaternion_world_to_sensor_matrix(
    quaternion_wxyz: QuaternionWXYZ,
) -> NDArray[np.float64]:
    """Return R_world_to_sensor for a normalized Hamilton wxyz quaternion."""

    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def evaluate_environment(
    configuration: EnvironmentConfiguration,
    position_m: Vector3,
    sim_time_s: float,
    common_field_world_T: NDArray[np.float64],
    event_field_world_T: NDArray[np.float64],
) -> EnvironmentEvaluation:
    constant = ConstantLocalFieldProvider(configuration).evaluate(
        position_m,
        sim_time_s,
    )
    spatial = SyntheticSpatialFieldProvider(configuration).evaluate(
        position_m,
        sim_time_s,
    )
    if spatial.field_world_T is None:
        return EnvironmentEvaluation(
            uniform_field_world_T=constant.uniform_field_world_T,
            common_field_world_T=common_field_world_T.copy(),
            gradient_field_world_T=spatial.gradient_field_world_T,
            dipole_field_world_T=None,
            anomaly_field_world_T=spatial.anomaly_field_world_T,
            periodic_field_world_T=spatial.periodic_field_world_T,
            event_field_world_T=event_field_world_T.copy(),
            field_true_world_T=None,
            invalid_dipole=spatial.invalid_dipole,
        )

    total = (
        constant.field_world_T
        + common_field_world_T
        + spatial.field_world_T
        + event_field_world_T
    )
    return EnvironmentEvaluation(
        uniform_field_world_T=constant.uniform_field_world_T,
        common_field_world_T=common_field_world_T.copy(),
        gradient_field_world_T=spatial.gradient_field_world_T,
        dipole_field_world_T=spatial.dipole_field_world_T,
        anomaly_field_world_T=spatial.anomaly_field_world_T,
        periodic_field_world_T=spatial.periodic_field_world_T,
        event_field_world_T=event_field_world_T.copy(),
        field_true_world_T=total,
        invalid_dipole=None,
    )
