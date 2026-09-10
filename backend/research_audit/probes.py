"""Reproducible Milestone 1D.0 probes isolated from production behavior.

These probes call the existing feature extractor, simulator and author-supplied
TQK8 implementation. They do not activate a new encoder or training service.
"""

from __future__ import annotations

import importlib.util
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from math import pi
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from app.features import (
    FeatureExtractionRequest,
    FeatureWindowConfiguration,
    MeasuredVectorSeries,
    extract_windowed_magnetometer_features,
)
from app.network.models import (
    EnvironmentConfiguration,
    EventKind,
    NetworkEventConfiguration,
    NetworkSessionConfiguration,
    NodeErrorConfiguration,
    PeriodicFieldConfiguration,
    SensorNodeConfiguration,
)
from app.network.simulation import NetworkSimulator
from app.quantum.user_pipeline.tqk8 import (
    N_PARAMS,
    N_QUBITS,
    AngleScaler,
    StateEngine,
    fit_qng,
)

PHASE_FEATURE_INDEX = 1
PHASE_AUDIT_SEED = 104_729
OFFSET_AUDIT_SEED = 240_517
EQUIVALENCE_AUDIT_SEED = 240_523
TRAINING_BUDGET_SEED = 240_529
STATEVECTOR_COMPLEX_BYTES = 16
STATEVECTOR_DIMENSION = 1 << N_QUBITS


@dataclass(frozen=True)
class LegacyPhaseScalingProbe:
    seed: int
    seam_epsilon_rad: float
    physical_circular_separation_rad: float
    legacy_encoded_phase_left_rad: float
    legacy_encoded_phase_right_rad: float
    legacy_encoded_linear_separation_rad: float


@dataclass(frozen=True)
class VQCPhaseProbe:
    engine: str
    seed: int
    seam_epsilon_rad: float
    theta_l2_norm: float
    periodic_density_max_abs_error: float
    periodic_kernel_abs_error: float
    seam_density_max_abs_error: float
    seam_kernel_abs_error: float


@dataclass(frozen=True)
class ConstantOffsetProbe:
    seed: int
    selected_channel: str
    constant_offset_nT: float
    baseline_feature_values: tuple[float, ...]
    shifted_feature_values: tuple[float, ...]
    maximum_absolute_feature_delta: float


@dataclass(frozen=True)
class ObservationEquivalenceProbe:
    seed: int
    sample_count: int
    maximum_observation_delta_T: float
    maximum_feature_delta: float
    physical_active_cause: str
    instrumental_active_cause: str
    physical_truth_event_field_T: tuple[float, float, float]
    instrumental_truth_event_field_T: tuple[float, float, float]


@dataclass(frozen=True)
class TrainingIterationBudgetProbe:
    seed: int
    backend: str
    training_samples: int
    requested_optimizer_steps: int
    accepted_optimizer_steps: int
    exact_state_evaluations: int
    maximum_state_evaluation_budget: int
    theoretical_core_array_bytes: int
    measured_peak_tracemalloc_bytes: int
    elapsed_seconds: float


@dataclass(frozen=True)
class AuditReport:
    legacy_phase_scaling: LegacyPhaseScalingProbe
    vqc_phase: tuple[VQCPhaseProbe, ...]
    constant_offset: ConstantOffsetProbe
    observation_equivalence: ObservationEquivalenceProbe
    training_iteration_budget: TrainingIterationBudgetProbe


def _circular_separation(left: float, right: float) -> float:
    return abs(float(np.angle(np.exp(1.0j * (left - right)))))


def run_legacy_phase_scaling_probe(
    *, seed: int = PHASE_AUDIT_SEED, seam_epsilon_rad: float = 1.0e-3
) -> LegacyPhaseScalingProbe:
    """Measure the legacy linear scaling gap around the physical phase seam."""

    rng = np.random.default_rng(seed)
    training = rng.normal(size=(8, N_QUBITS))
    training[:, PHASE_FEATURE_INDEX] = np.array(
        [
            -pi + seam_epsilon_rad,
            -pi + 2.0 * seam_epsilon_rad,
            -pi + 3.0 * seam_epsilon_rad,
            -pi + 4.0 * seam_epsilon_rad,
            pi - 4.0 * seam_epsilon_rad,
            pi - 3.0 * seam_epsilon_rad,
            pi - 2.0 * seam_epsilon_rad,
            pi - seam_epsilon_rad,
        ]
    )
    query = np.repeat(training[:1], 2, axis=0)
    left_phase = -pi + seam_epsilon_rad
    right_phase = pi - seam_epsilon_rad
    query[:, PHASE_FEATURE_INDEX] = (left_phase, right_phase)

    scaler = AngleScaler.fit(training)
    encoded = scaler.transform(query)
    encoded_left = float(encoded[0, PHASE_FEATURE_INDEX])
    encoded_right = float(encoded[1, PHASE_FEATURE_INDEX])
    return LegacyPhaseScalingProbe(
        seed=seed,
        seam_epsilon_rad=seam_epsilon_rad,
        physical_circular_separation_rad=_circular_separation(left_phase, right_phase),
        legacy_encoded_phase_left_rad=encoded_left,
        legacy_encoded_phase_right_rad=encoded_right,
        legacy_encoded_linear_separation_rad=abs(encoded_left - encoded_right),
    )


def _density_max_abs_error(left: NDArray[np.complex128], right: NDArray[np.complex128]) -> float:
    left_density = np.outer(left, left.conj())
    right_density = np.outer(right, right.conj())
    return float(np.max(np.abs(left_density - right_density)))


def _kernel_value(left: NDArray[np.complex128], right: NDArray[np.complex128]) -> float:
    return float(abs(np.vdot(left, right)) ** 2)


def _available_exact_engines() -> tuple[str, ...]:
    engines = ["numpy"]
    if importlib.util.find_spec("qiskit") is not None:
        engines.append("qiskit")
    return tuple(engines)


def run_vqc_phase_probe(
    *, seed: int = PHASE_AUDIT_SEED, seam_epsilon_rad: float = 1.0e-4
) -> tuple[VQCPhaseProbe, ...]:
    """Probe direct phase periodicity and seam continuity in the actual VQC."""

    rng = np.random.default_rng(seed)
    base = rng.uniform(-1.0, 1.0, N_QUBITS)
    anchor = rng.uniform(-1.0, 1.0, N_QUBITS)
    theta = rng.uniform(-0.8, 0.8, N_PARAMS)
    base[PHASE_FEATURE_INDEX] = 0.731
    periodic = base.copy()
    periodic[PHASE_FEATURE_INDEX] += 2.0 * pi
    seam_left = base.copy()
    seam_right = base.copy()
    seam_left[PHASE_FEATURE_INDEX] = -pi + seam_epsilon_rad
    seam_right[PHASE_FEATURE_INDEX] = pi - seam_epsilon_rad

    results: list[VQCPhaseProbe] = []
    for engine_name in _available_exact_engines():
        engine = StateEngine(engine_name)
        base_state = np.asarray(engine.state(base, theta), dtype=np.complex128)
        periodic_state = np.asarray(engine.state(periodic, theta), dtype=np.complex128)
        seam_left_state = np.asarray(engine.state(seam_left, theta), dtype=np.complex128)
        seam_right_state = np.asarray(engine.state(seam_right, theta), dtype=np.complex128)
        anchor_state = np.asarray(engine.state(anchor, theta), dtype=np.complex128)
        results.append(
            VQCPhaseProbe(
                engine=engine_name,
                seed=seed,
                seam_epsilon_rad=seam_epsilon_rad,
                theta_l2_norm=float(np.linalg.norm(theta)),
                periodic_density_max_abs_error=_density_max_abs_error(base_state, periodic_state),
                periodic_kernel_abs_error=abs(
                    _kernel_value(anchor_state, base_state)
                    - _kernel_value(anchor_state, periodic_state)
                ),
                seam_density_max_abs_error=_density_max_abs_error(
                    seam_left_state, seam_right_state
                ),
                seam_kernel_abs_error=abs(
                    _kernel_value(anchor_state, seam_left_state)
                    - _kernel_value(anchor_state, seam_right_state)
                ),
            )
        )
    return tuple(results)


def _measured_series(
    signal_nT: NDArray[np.float64],
    *,
    acquisition_id: str,
    sampling_rate_hz: float,
) -> MeasuredVectorSeries:
    sample_count = signal_nT.size
    time_s = np.arange(sample_count, dtype=np.float64) / sampling_rate_hz
    field = np.column_stack(
        (
            signal_nT,
            np.full(sample_count, 11.0, dtype=np.float64),
            np.full(sample_count, -7.0, dtype=np.float64),
        )
    )
    return MeasuredVectorSeries(
        acquisition_id=acquisition_id,
        sensor_id="audit-S1",
        sampling_rate_hz=sampling_rate_hz,
        time_s=time_s.tolist(),
        measured_field=field.tolist(),
        field_unit="nT",
        temperature_k=np.full(sample_count, 293.15, dtype=np.float64).tolist(),
        saturation_mask=np.zeros((sample_count, 3), dtype=bool).tolist(),
    )


def _extract_single_window(series: MeasuredVectorSeries, *, duration_s: float) -> tuple[float, ...]:
    response = extract_windowed_magnetometer_features(
        FeatureExtractionRequest(
            series=series,
            channel="x",
            window=FeatureWindowConfiguration(
                duration_s=duration_s,
                overlap_fraction=0.0,
            ),
        )
    )
    if len(response.windows) != 1:
        raise RuntimeError("audit fixture must produce exactly one complete window")
    return tuple(float(value) for value in response.windows[0].features.values)


def run_constant_offset_probe(
    *, seed: int = OFFSET_AUDIT_SEED, constant_offset_nT: float = 375.0
) -> ConstantOffsetProbe:
    """Show that the current x-channel features omit a constant field offset."""

    rng = np.random.default_rng(seed)
    sampling_rate_hz = 100.0
    duration_s = 4.0
    time_s = np.arange(int(sampling_rate_hz * duration_s)) / sampling_rate_hz
    shared_noise_nT = rng.normal(0.0, 0.05, time_s.size)
    baseline_signal_nT = (
        25.0 + 8.0 * np.sin(2.0 * pi * 5.0 * time_s + 0.37) + 0.15 * time_s + shared_noise_nT
    )
    shifted_signal_nT = baseline_signal_nT + constant_offset_nT
    baseline_features = _extract_single_window(
        _measured_series(
            baseline_signal_nT,
            acquisition_id="audit-offset-baseline",
            sampling_rate_hz=sampling_rate_hz,
        ),
        duration_s=duration_s,
    )
    shifted_features = _extract_single_window(
        _measured_series(
            shifted_signal_nT,
            acquisition_id="audit-offset-shifted",
            sampling_rate_hz=sampling_rate_hz,
        ),
        duration_s=duration_s,
    )
    maximum_delta = float(
        np.max(np.abs(np.asarray(baseline_features) - np.asarray(shifted_features)))
    )
    return ConstantOffsetProbe(
        seed=seed,
        selected_channel="x",
        constant_offset_nT=constant_offset_nT,
        baseline_feature_values=baseline_features,
        shifted_feature_values=shifted_features,
        maximum_absolute_feature_delta=maximum_delta,
    )


def _equivalence_configuration(seed: int) -> NetworkSessionConfiguration:
    return NetworkSessionConfiguration(
        session_name="AQSE 1D.0 observation-equivalence audit",
        random_seed=seed,
        sampling_rate_Hz=100.0,
        ui_refresh_rate_Hz=5.0,
        buffer_duration_s=4.0,
        environment=EnvironmentConfiguration(
            uniform_field_T=(20.0e-6, -3.0e-6, 45.0e-6),
            dipoles=(),
            periodic_fields=(
                PeriodicFieldConfiguration(
                    source_id="audit-harmonic",
                    amplitude_world_T=(8.0e-9, 0.0, 0.0),
                    frequency_Hz=5.0,
                    phase_rad=0.37,
                ),
            ),
        ),
        nodes=(
            SensorNodeConfiguration(
                sensor_id="S1",
                position_m=(0.0, 0.0, 0.0),
                errors=NodeErrorConfiguration(),
            ),
        ),
    )


def _series_from_components(
    components_T: NDArray[np.float64], *, acquisition_id: str
) -> MeasuredVectorSeries:
    sample_count = components_T.shape[0]
    return MeasuredVectorSeries(
        acquisition_id=acquisition_id,
        sensor_id="S1",
        sampling_rate_hz=100.0,
        time_s=(np.arange(sample_count, dtype=np.float64) / 100.0).tolist(),
        measured_field=components_T.tolist(),
        field_unit="T",
        temperature_k=np.full(sample_count, 293.15, dtype=np.float64).tolist(),
        saturation_mask=np.zeros((sample_count, 3), dtype=bool).tolist(),
    )


def run_observation_equivalence_probe(
    *, seed: int = EQUIVALENCE_AUDIT_SEED
) -> ObservationEquivalenceProbe:
    """Construct two distinct causes with identical one-node observations."""

    configuration = _equivalence_configuration(seed)
    offset_T = (7.0e-9, -2.0e-9, 3.0e-9)
    physical_event = NetworkEventConfiguration(
        event_id="audit-world-field-offset",
        kind=EventKind.WORLD_FIELD_OFFSET,
        start_time_s=0.0,
        duration_s=4.0,
        field_offset_T=offset_T,
    )
    instrument_event = NetworkEventConfiguration(
        event_id="audit-instrument-bias",
        kind=EventKind.NODE_BIAS,
        start_time_s=0.0,
        duration_s=4.0,
        target_sensor_ids=("S1",),
        field_offset_T=offset_T,
    )
    physical_simulator = NetworkSimulator(configuration)
    instrument_simulator = NetworkSimulator(configuration)
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sample_count = 400
    physical_components: list[tuple[float, float, float]] = []
    instrument_components: list[tuple[float, float, float]] = []
    physical_last = None
    instrument_last = None
    for index in range(sample_count):
        sim_time_s = index / configuration.sampling_rate_Hz
        physical_last = physical_simulator.generate(
            "audit-physical",
            index + 1,
            sim_time_s,
            epoch,
            (physical_event,),
        )
        instrument_last = instrument_simulator.generate(
            "audit-instrument",
            index + 1,
            sim_time_s,
            epoch,
            (instrument_event,),
        )
        physical_reading = physical_last.observation.readings[0].components_T
        instrument_reading = instrument_last.observation.readings[0].components_T
        if physical_reading is None or instrument_reading is None:
            raise RuntimeError("audit fixture unexpectedly produced a missing vector reading")
        physical_components.append(physical_reading)
        instrument_components.append(instrument_reading)

    if physical_last is None or instrument_last is None:
        raise RuntimeError("audit fixture did not produce observations")
    physical_array = np.asarray(physical_components, dtype=np.float64)
    instrument_array = np.asarray(instrument_components, dtype=np.float64)
    physical_features = _extract_single_window(
        _series_from_components(
            physical_array,
            acquisition_id="audit-equivalence-physical",
        ),
        duration_s=4.0,
    )
    instrument_features = _extract_single_window(
        _series_from_components(
            instrument_array,
            acquisition_id="audit-equivalence-instrument",
        ),
        duration_s=4.0,
    )
    return ObservationEquivalenceProbe(
        seed=seed,
        sample_count=sample_count,
        maximum_observation_delta_T=float(np.max(np.abs(physical_array - instrument_array))),
        maximum_feature_delta=float(
            np.max(np.abs(np.asarray(physical_features) - np.asarray(instrument_features)))
        ),
        physical_active_cause=physical_last.truth.active_causes[0].kind.value,
        instrumental_active_cause=instrument_last.truth.active_causes[0].kind.value,
        physical_truth_event_field_T=physical_last.truth.fields[0].event_field_world_T,
        instrumental_truth_event_field_T=(instrument_last.truth.fields[0].event_field_world_T),
    )


class _CountingStateEngine(StateEngine):
    def __init__(self) -> None:
        super().__init__("numpy")
        self.state_evaluations = 0

    def state(self, x: object, theta: object) -> NDArray[np.complex128]:
        self.state_evaluations += 1
        return np.asarray(super().state(x, theta), dtype=np.complex128)


def _theoretical_core_array_bytes(training_samples: int) -> int:
    states = training_samples * STATEVECTOR_DIMENSION * STATEVECTOR_COMPLEX_BYTES
    derivatives = training_samples * N_PARAMS * STATEVECTOR_DIMENSION * STATEVECTOR_COMPLEX_BYTES
    kernel = training_samples * training_samples * np.dtype(np.float64).itemsize
    metric = N_PARAMS * N_PARAMS * np.dtype(np.float64).itemsize
    return int(states + derivatives + kernel + metric)


def run_training_iteration_budget_probe(
    *, seed: int = TRAINING_BUDGET_SEED, training_samples: int = 4
) -> TrainingIterationBudgetProbe:
    """Measure one bounded toy QNG step without persisting or promoting a model."""

    if training_samples != 4:
        raise ValueError("the 1D.0 audit fixture is intentionally fixed at four samples")
    rng = np.random.default_rng(seed)
    encoded_inputs = rng.uniform(-1.1, 1.1, size=(training_samples, N_QUBITS))
    labels = np.asarray((-1.0, 1.0, -1.0, 1.0), dtype=np.float64)
    theta = rng.uniform(-0.8, 0.8, size=N_PARAMS)
    engine = _CountingStateEngine()

    tracemalloc.start()
    started = perf_counter()
    try:
        _, history = fit_qng(
            engine,
            encoded_inputs,
            labels,
            theta,
            steps=1,
            learning_rate=0.2,
            damping=1.0e-3,
            max_step=0.4,
            verbose=False,
        )
        elapsed_seconds = perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    maximum_state_evaluation_budget = training_samples * (1 + 2 * N_PARAMS + 12)
    return TrainingIterationBudgetProbe(
        seed=seed,
        backend="numpy_statevector",
        training_samples=training_samples,
        requested_optimizer_steps=1,
        accepted_optimizer_steps=len(history),
        exact_state_evaluations=engine.state_evaluations,
        maximum_state_evaluation_budget=maximum_state_evaluation_budget,
        theoretical_core_array_bytes=_theoretical_core_array_bytes(training_samples),
        measured_peak_tracemalloc_bytes=peak_bytes,
        elapsed_seconds=elapsed_seconds,
    )


def run_all_probes() -> AuditReport:
    """Run every bounded 1D.0 audit probe and return in-memory evidence only."""

    return AuditReport(
        legacy_phase_scaling=run_legacy_phase_scaling_probe(),
        vqc_phase=run_vqc_phase_probe(),
        constant_offset=run_constant_offset_probe(),
        observation_equivalence=run_observation_equivalence_probe(),
        training_iteration_budget=run_training_iteration_budget_probe(),
    )
