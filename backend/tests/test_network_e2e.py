from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.features.models import (
    FeatureExtractionRequest,
    FeatureWindowConfiguration,
    MeasuredVectorSeries,
)
from app.features.windowed import extract_windowed_magnetometer_features
from app.network.defaults import network_preset_configuration
from app.network.simulation import NetworkSimulator
from app.quantum.preview import QuantumPreviewRequest, run_quantum_preview


def test_explicit_harmonic_preset_reaches_fixed_theta_preview_end_to_end() -> None:
    configuration = network_preset_configuration("quantum_preview_signal")
    assert len(configuration.environment.periodic_fields) == 1
    simulator = NetworkSimulator(configuration)
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frames = [
        simulator.generate(
            "network-e2e",
            index + 1,
            index / configuration.sampling_rate_Hz,
            epoch,
            configuration.events,
        )
        for index in range(400)
    ]
    readings = [frame.observation.readings[0] for frame in frames]
    assert all(reading.components_T is not None for reading in readings)
    assert all(reading.saturation_mask is not None for reading in readings)
    series = MeasuredVectorSeries(
        acquisition_id="network-e2e-acquisition",
        sensor_id=configuration.nodes[0].sensor_id,
        sampling_rate_hz=configuration.sampling_rate_Hz,
        time_s=[frame.observation.sim_time_s for frame in frames],
        measured_field=[reading.components_T for reading in readings],
        field_unit="T",
        temperature_k=[reading.observed_temperature_K for reading in readings],
        saturation_mask=[reading.saturation_mask for reading in readings],
    )
    extracted = extract_windowed_magnetometer_features(
        FeatureExtractionRequest(
            series=series,
            channel="magnitude",
            window=FeatureWindowConfiguration(
                duration_s=1.0,
                overlap_fraction=0.5,
            ),
        )
    )
    valid_windows = [
        window for window in extracted.windows if window.quality.valid_for_quantum
    ]

    assert len(extracted.windows) == 7
    assert len(valid_windows) == 7
    preview = run_quantum_preview(
        QuantumPreviewRequest(
            backend="numpy",
            feature_profile=extracted.profile,
            reference_dataset_id="network-harmonic-reference-v1",
            reference_windows=valid_windows[:4],
            theta=tuple(np.linspace(-0.6, 0.6, 16)),
        )
    )

    assert len(preview.reference_kernel) == 4
    assert preview.scientific_scope.endswith("no QNG training")
