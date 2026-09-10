"""Regression tests for isolated Milestone 1D.0 evidence probes."""

from __future__ import annotations

import importlib.util

import numpy as np

from research_audit.probes import (
    run_constant_offset_probe,
    run_legacy_phase_scaling_probe,
    run_observation_equivalence_probe,
    run_training_iteration_budget_probe,
    run_vqc_phase_probe,
)


def test_legacy_angle_scaler_treats_nearby_wrapped_phases_as_distant() -> None:
    result = run_legacy_phase_scaling_probe()

    assert result.seed == 104_729
    assert result.physical_circular_separation_rad < 0.003
    assert result.legacy_encoded_linear_separation_rad > 1.0


def test_actual_vqc_is_periodic_and_continuous_at_the_phase_seam() -> None:
    results = run_vqc_phase_probe()

    assert {result.engine for result in results} >= {"numpy"}
    if importlib.util.find_spec("qiskit") is not None:
        assert {result.engine for result in results} == {"numpy", "qiskit"}
    for result in results:
        assert result.theta_l2_norm > 0.5
        assert result.periodic_density_max_abs_error < 2.0e-12
        assert result.periodic_kernel_abs_error < 2.0e-12
        assert result.seam_density_max_abs_error < 2.0e-4
        assert result.seam_kernel_abs_error < 2.0e-4

    if len(results) == 2:
        numpy_result, qiskit_result = results
        assert numpy_result.engine == "numpy"
        assert qiskit_result.engine == "qiskit"
        np.testing.assert_allclose(
            [
                numpy_result.periodic_density_max_abs_error,
                numpy_result.periodic_kernel_abs_error,
                numpy_result.seam_density_max_abs_error,
                numpy_result.seam_kernel_abs_error,
            ],
            [
                qiskit_result.periodic_density_max_abs_error,
                qiskit_result.periodic_kernel_abs_error,
                qiskit_result.seam_density_max_abs_error,
                qiskit_result.seam_kernel_abs_error,
            ],
            atol=2.0e-12,
            rtol=1.0e-8,
        )


def test_current_linear_channel_features_omit_a_constant_field_offset() -> None:
    result = run_constant_offset_probe()

    assert result.selected_channel == "x"
    assert result.constant_offset_nT == 375.0
    np.testing.assert_allclose(
        result.shifted_feature_values,
        result.baseline_feature_values,
        atol=1.0e-8,
        rtol=1.0e-10,
    )


def test_world_field_offset_and_instrument_bias_can_be_observationally_equivalent() -> None:
    result = run_observation_equivalence_probe()

    assert result.sample_count == 400
    assert result.maximum_observation_delta_T < 1.0e-18
    assert result.maximum_feature_delta < 1.0e-8
    assert result.physical_active_cause == "world_field_offset"
    assert result.instrumental_active_cause == "node_bias"
    assert result.physical_truth_event_field_T != (0.0, 0.0, 0.0)
    assert result.instrumental_truth_event_field_T == (0.0, 0.0, 0.0)


def test_one_toy_training_iteration_has_a_bounded_local_budget() -> None:
    result = run_training_iteration_budget_probe()

    assert result.backend == "numpy_statevector"
    assert result.training_samples == 4
    assert result.requested_optimizer_steps == 1
    assert result.accepted_optimizer_steps == 1
    assert 33 * result.training_samples < result.exact_state_evaluations
    assert result.exact_state_evaluations <= result.maximum_state_evaluation_budget
    assert result.theoretical_core_array_bytes < 1_000_000
    assert result.measured_peak_tracemalloc_bytes < 64 * 1024 * 1024
    assert result.elapsed_seconds < 10.0
