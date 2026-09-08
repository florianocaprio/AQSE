from __future__ import annotations

import json

import numpy as np

from app.network.defaults import default_network_configuration
from app.network.models import NodeErrorConfiguration, ObservabilityRequest
from app.network.observability import calculate_observability


def test_single_node_diagnostic_reports_insufficient_local_geometry() -> None:
    diagnostics = calculate_observability(
        ObservabilityRequest(configuration=default_network_configuration(1))
    )

    assert diagnostics.measurement_dimension == 3
    assert diagnostics.numerical_rank <= 3
    assert diagnostics.full_column_rank is False
    assert diagnostics.condition_number is None
    assert "geometry_insufficient" in diagnostics.warnings
    assert "parameters_not_identifiable" in diagnostics.warnings
    assert "does not guarantee" in diagnostics.interpretation


def test_non_coplanar_eight_node_preset_has_finite_local_diagnostics() -> None:
    diagnostics = calculate_observability(
        ObservabilityRequest(configuration=default_network_configuration(8))
    )

    assert diagnostics.jacobian_shape == (24, 6)
    assert diagnostics.numerical_rank == 6
    assert diagnostics.full_column_rank is True
    assert diagnostics.condition_number is not None
    assert np.all(np.isfinite(diagnostics.singular_values))
    assert np.all(np.isfinite(diagnostics.weighted_nondimensional_jacobian))
    json.dumps(diagnostics.model_dump(mode="json"), allow_nan=False)


def test_observability_requires_unambiguous_enabled_source() -> None:
    configuration = default_network_configuration(4)
    extra_source = configuration.environment.dipoles[0].model_copy(
        update={"source_id": "second-source", "initial_position_m": (0.0, 0.0, 2.0)}
    )
    configuration = configuration.model_copy(
        update={
            "environment": configuration.environment.model_copy(
                update={"dipoles": (*configuration.environment.dipoles, extra_source)}
            )
        }
    )

    try:
        calculate_observability(ObservabilityRequest(configuration=configuration))
    except ValueError as exc:
        assert "source_id is required" in str(exc)
    else:
        raise AssertionError("ambiguous dipoles must require source_id")


def test_observability_uses_c_axis_gain_soft_iron_order() -> None:
    configuration = default_network_configuration(1)
    gain = np.diag((2.0, 1.0, 0.5))
    soft = np.asarray(((1.0, 0.2, 0.0), (0.2, 1.0, 0.0), (0.0, 0.0, 1.0)))
    cross = np.asarray(((1.0, 0.0, 0.1), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    noise = (1.0e-9,) * 3
    baseline_node = configuration.nodes[0].model_copy(
        update={"errors": NodeErrorConfiguration(white_noise_std_T_per_sample=noise)}
    )
    transformed_node = baseline_node.model_copy(
        update={
            "errors": NodeErrorConfiguration(
                gain_matrix=tuple(tuple(float(value) for value in row) for row in gain),
                soft_iron_matrix=tuple(tuple(float(value) for value in row) for row in soft),
                cross_axis_matrix=tuple(tuple(float(value) for value in row) for row in cross),
                white_noise_std_T_per_sample=noise,
            )
        }
    )
    baseline = calculate_observability(
        ObservabilityRequest(
            configuration=configuration.model_copy(update={"nodes": (baseline_node,)})
        )
    )
    transformed = calculate_observability(
        ObservabilityRequest(
            configuration=configuration.model_copy(update={"nodes": (transformed_node,)})
        )
    )
    baseline_jacobian = np.asarray(baseline.weighted_nondimensional_jacobian)
    transformed_jacobian = np.asarray(transformed.weighted_nondimensional_jacobian)
    expected = cross @ gain @ soft @ baseline_jacobian
    previous_incorrect = cross @ soft @ gain @ baseline_jacobian

    np.testing.assert_allclose(transformed_jacobian, expected, rtol=1.0e-6, atol=1.0e-6)
    assert not np.allclose(transformed_jacobian, previous_incorrect, rtol=1.0e-4, atol=1.0e-4)
