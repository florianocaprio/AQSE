from __future__ import annotations

from itertools import combinations
from math import sin

import numpy as np
from numpy.typing import NDArray

from app.network.models import (
    DipoleSourceConfiguration,
    MeasurementMode,
    ObservabilityDiagnostics,
    ObservabilityRequest,
)
from app.network.physics import (
    gaussian_anomaly_field_T,
    point_dipole_field_T,
    quaternion_world_to_sensor_matrix,
)

_PARAMETER_NAMES = (
    "source_position_north_m",
    "source_position_east_m",
    "source_position_down_m",
    "moment_north_A_m2",
    "moment_east_A_m2",
    "moment_down_A_m2",
)


def calculate_observability(request: ObservabilityRequest) -> ObservabilityDiagnostics:
    """Compute a local, weighted and nondimensionalized numerical Jacobian.

    This diagnostic evaluates only instantaneous dipole position and moment. It
    is a local conditioning indicator, never a guarantee of global uniqueness.
    """

    configuration = request.configuration
    source = _select_source(request)
    source_position = np.asarray(source.initial_position_m, dtype=np.float64)
    moment = np.asarray(source.moment_A_m2, dtype=np.float64)
    physical_parameters = np.concatenate((source_position, moment))

    positions = [np.asarray(node.position_m, dtype=np.float64) for node in configuration.nodes]
    baselines = [
        float(np.linalg.norm(left - right)) for left, right in combinations(positions, 2)
    ]
    position_scale = max([source.minimum_distance_m, 0.1, *baselines])
    moment_scale = max(float(np.linalg.norm(moment)), 1.0e-6)
    scales = np.asarray((position_scale,) * 3 + (moment_scale,) * 3)
    weights = _measurement_noise_scales(request)

    columns: list[NDArray[np.float64]] = []
    for parameter_index, scale in enumerate(scales):
        delta = request.numerical_step * scale
        lower = physical_parameters.copy()
        upper = physical_parameters.copy()
        lower[parameter_index] -= delta
        upper[parameter_index] += delta
        lower_prediction = _predict_measurements(request, source, lower)
        upper_prediction = _predict_measurements(request, source, upper)
        # derivative with respect to x/scale, then whiten by per-sample RMS noise
        columns.append(
            ((upper_prediction - lower_prediction) / (2.0 * request.numerical_step))
            / weights
        )
    jacobian = np.column_stack(columns)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = (
        max(jacobian.shape)
        * np.finfo(np.float64).eps
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    rank = int(np.sum(singular_values > tolerance))
    full_rank = rank == len(_PARAMETER_NAMES)
    condition_number: float | None = None
    if full_rank and singular_values[-1] > 0.0:
        condition_number = float(singular_values[0] / singular_values[-1])

    warnings: list[str] = ["instantaneous_velocity_not_observable"]
    if jacobian.shape[0] < len(_PARAMETER_NAMES):
        warnings.append("geometry_insufficient")
    if not full_rank:
        warnings.append("parameters_not_identifiable")
    elif condition_number is not None and condition_number > request.condition_warning_threshold:
        warnings.append("ill_conditioned")

    return ObservabilityDiagnostics(
        source_id=source.source_id,
        parameter_names=_PARAMETER_NAMES,
        measurement_dimension=jacobian.shape[0],
        jacobian_shape=jacobian.shape,
        weighted_nondimensional_jacobian=tuple(
            tuple(float(value) for value in row) for row in jacobian
        ),
        singular_values=tuple(float(value) for value in singular_values),
        numerical_rank=rank,
        full_column_rank=full_rank,
        condition_number=condition_number,
        parameter_scales=tuple(float(value) for value in scales),
        noise_floor_T=request.noise_floor_T,
        warnings=tuple(warnings),
        interpretation=(
            "Local instantaneous Jacobian for dipole position and moment, weighted by "
            "declared per-sample RMS noise (with the explicit noise_floor_T) and "
            "nondimensionalized by array/source scales. "
            "Full local rank does not guarantee a globally unique inverse solution."
        ),
    )


def _select_source(request: ObservabilityRequest) -> DipoleSourceConfiguration:
    enabled = [source for source in request.configuration.environment.dipoles if source.enabled]
    if request.source_id is None:
        if len(enabled) != 1:
            raise ValueError("source_id is required unless exactly one dipole is enabled")
        return enabled[0]
    for source in enabled:
        if source.source_id == request.source_id:
            return source
    raise ValueError(f"enabled dipole source not found: {request.source_id}")


def _measurement_noise_scales(request: ObservabilityRequest) -> NDArray[np.float64]:
    scales: list[float] = []
    for node in request.configuration.nodes:
        component_std = np.maximum(
            np.asarray(node.errors.white_noise_std_T_per_sample, dtype=np.float64),
            request.noise_floor_T,
        )
        if node.measurement_mode is MeasurementMode.VECTOR:
            scales.extend(float(value) for value in component_std)
        elif node.measurement_mode is MeasurementMode.MONOAXIAL:
            axis = np.asarray(node.monoaxial_axis_sensor, dtype=np.float64)
            scales.append(float(np.sqrt(np.sum((axis * component_std) ** 2))))
        else:
            scales.append(float(np.linalg.norm(component_std) / np.sqrt(3.0)))
    return np.asarray(scales, dtype=np.float64)


def _predict_measurements(
    request: ObservabilityRequest,
    selected_source: DipoleSourceConfiguration,
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    environment = request.configuration.environment
    selected_position = parameters[:3]
    selected_moment = parameters[3:]
    predictions: list[float] = []
    for node in request.configuration.nodes:
        node_position = np.asarray(node.position_m, dtype=np.float64)
        world_field = np.asarray(environment.uniform_field_T, dtype=np.float64)
        gradient = np.asarray(environment.gradient_T_per_m, dtype=np.float64)
        reference = np.asarray(environment.gradient_reference_position_m, dtype=np.float64)
        world_field = world_field + gradient @ (node_position - reference)
        for anomaly in environment.gaussian_anomalies:
            if anomaly.enabled:
                world_field = world_field + gaussian_anomaly_field_T(
                    node_position,
                    anomaly,
                )
        for periodic in environment.periodic_fields:
            if periodic.enabled:
                world_field = world_field + np.asarray(
                    periodic.amplitude_world_T,
                    dtype=np.float64,
                ) * sin(periodic.phase_rad)
        for source in environment.dipoles:
            if not source.enabled:
                continue
            source_position = (
                selected_position
                if source.source_id == selected_source.source_id
                else np.asarray(source.initial_position_m, dtype=np.float64)
            )
            source_moment = (
                selected_moment
                if source.source_id == selected_source.source_id
                else np.asarray(source.moment_A_m2, dtype=np.float64)
            )
            world_field = world_field + point_dipole_field_T(
                node_position,
                source_position,
                source_moment,
                source.minimum_distance_m,
                source.source_id,
            )
        rotation = quaternion_world_to_sensor_matrix(
            node.orientation_world_to_sensor_wxyz
        )
        response = (
            np.asarray(node.errors.cross_axis_matrix, dtype=np.float64)
            @ np.asarray(node.errors.soft_iron_matrix, dtype=np.float64)
            @ np.asarray(node.errors.gain_matrix, dtype=np.float64)
            @ rotation
            @ world_field
        )
        if node.measurement_mode is MeasurementMode.VECTOR:
            predictions.extend(float(value) for value in response)
        elif node.measurement_mode is MeasurementMode.MONOAXIAL:
            axis = np.asarray(node.monoaxial_axis_sensor, dtype=np.float64)
            predictions.append(float(axis @ response))
        else:
            predictions.append(float(np.linalg.norm(response)))
    return np.asarray(predictions, dtype=np.float64)
