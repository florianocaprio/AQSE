from __future__ import annotations

import json

import numpy as np

from app.network.defaults import default_network_configuration
from app.network.models import ObservabilityRequest
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
