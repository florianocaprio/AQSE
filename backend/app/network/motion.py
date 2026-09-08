from __future__ import annotations

from math import cos, pi, sin, sqrt

import numpy as np
from numpy.typing import NDArray

from app.network.models import QuaternionWXYZ, Vector3


def canonical_quaternion(values: NDArray[np.float64]) -> QuaternionWXYZ:
    """Normalize a Hamilton quaternion and select the API's canonical w >= 0 sign."""

    normalized = values / float(np.linalg.norm(values))
    if normalized[0] < 0.0:
        normalized = -normalized
    return tuple(float(value) for value in normalized)  # type: ignore[return-value]


def quaternion_multiply(
    left_wxyz: QuaternionWXYZ,
    right_wxyz: QuaternionWXYZ,
) -> QuaternionWXYZ:
    lw, lx, ly, lz = left_wxyz
    rw, rx, ry, rz = right_wxyz
    return canonical_quaternion(
        np.asarray(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dtype=np.float64,
        )
    )


def integrate_world_to_sensor_quaternion(
    quaternion_wxyz: QuaternionWXYZ,
    angular_rate_rad_per_s: Vector3,
    dt_s: float,
) -> QuaternionWXYZ:
    """Apply a continuous sensor-frame rotation increment to world-to-sensor q."""

    rotation_vector = np.asarray(angular_rate_rad_per_s, dtype=np.float64) * dt_s
    angle = float(np.linalg.norm(rotation_vector))
    if angle == 0.0:
        return quaternion_wxyz
    axis = rotation_vector / angle
    half_angle = 0.5 * angle
    delta = canonical_quaternion(
        np.asarray(
            (cos(half_angle), *(axis * sin(half_angle))),
            dtype=np.float64,
        )
    )
    return quaternion_multiply(delta, quaternion_wxyz)


def _van_der_corput(index: int, base: int) -> float:
    result = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        result += remainder / denominator
    return result


def haar_pose_quaternion(
    sequence_index: int,
    shifts: Vector3 = (0.0, 0.0, 0.0),
) -> QuaternionWXYZ:
    """Return a deterministic low-discrepancy Haar pose sample on SO(3).

    The Shoemake transform maps three uniform coordinates to Haar measure on
    unit quaternions. Radical-inverse coordinates provide a replayable pose
    sweep; optional Cranley shifts let a session seed select another sweep.
    """

    index = max(1, sequence_index)
    u1 = (_van_der_corput(index, 2) + shifts[0]) % 1.0
    u2 = (_van_der_corput(index, 3) + shifts[1]) % 1.0
    u3 = (_van_der_corput(index, 5) + shifts[2]) % 1.0
    radius_a = sqrt(1.0 - u1)
    radius_b = sqrt(u1)
    x = radius_a * sin(2.0 * pi * u2)
    y = radius_a * cos(2.0 * pi * u2)
    z = radius_b * sin(2.0 * pi * u3)
    w = radius_b * cos(2.0 * pi * u3)
    return canonical_quaternion(np.asarray((w, x, y, z), dtype=np.float64))
