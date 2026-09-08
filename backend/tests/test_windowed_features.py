from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from app.features import (
    FeatureExtractionRequest,
    FeatureWindowConfiguration,
    MeasuredVectorSeries,
    extract_windowed_magnetometer_features,
)
from app.preprocessing.features import (
    MAGNETOMETER_FEATURE_NAMES,
    MAGNETOMETER_FEATURE_UNITS,
)


def measured_series(
    *,
    field_unit: str = "nT",
    clipped_index: int | None = None,
    constant: bool = False,
) -> MeasuredVectorSeries:
    sampling_rate = 100.0
    time = np.arange(400, dtype=float) / sampling_rate
    x = (
        np.full_like(time, 25.0)
        if constant
        else 25.0 + 8.0 * np.sin(2.0 * np.pi * 8.0 * time + 0.3)
    )
    field = np.column_stack((x, np.full_like(x, 10.0), np.full_like(x, -4.0)))
    if field_unit == "T":
        field = field / 1.0e9
    mask = np.zeros((len(time), 3), dtype=bool)
    if clipped_index is not None:
        mask[clipped_index, 0] = True
    return MeasuredVectorSeries(
        acquisition_id="acquisition-001",
        sensor_id="sensor-001",
        sampling_rate_hz=sampling_rate,
        time_s=time.tolist(),
        measured_field=field.tolist(),
        field_unit=field_unit,
        temperature_k=(293.15 + 0.2 * time).tolist(),
        saturation_mask=mask.tolist(),
    )


def request_for(series: MeasuredVectorSeries, channel: str = "x") -> FeatureExtractionRequest:
    return FeatureExtractionRequest(
        series=series,
        channel=channel,
        window=FeatureWindowConfiguration(duration_s=1.0, overlap_fraction=0.5),
    )


def test_windowed_features_preserve_exact_legacy_contract() -> None:
    response = extract_windowed_magnetometer_features(request_for(measured_series()))

    assert response.profile.feature_names == MAGNETOMETER_FEATURE_NAMES
    assert response.profile.feature_units == MAGNETOMETER_FEATURE_UNITS
    assert response.profile.window_samples == 100
    assert response.profile.hop_samples == 50
    assert len(response.windows) == 7
    assert response.valid_window_count == 7
    assert response.discarded_sample_count == 0
    for window in response.windows:
        assert window.features.names == MAGNETOMETER_FEATURE_NAMES
        assert window.features.units == MAGNETOMETER_FEATURE_UNITS
        assert len(window.features.values) == 8
        assert all(math.isfinite(value) for value in window.features.values)
        assert window.features.values[2] == pytest.approx(8.0, abs=0.05)


def test_tesla_and_nanotesla_inputs_produce_the_same_features() -> None:
    nt = extract_windowed_magnetometer_features(request_for(measured_series()))
    tesla = extract_windowed_magnetometer_features(
        request_for(measured_series(field_unit="T"))
    )

    np.testing.assert_allclose(
        nt.windows[0].features.values,
        tesla.windows[0].features.values,
        atol=1.0e-8,
        rtol=1.0e-10,
    )


@pytest.mark.parametrize("channel", ["x", "y", "z", "magnitude"])
def test_channel_selection_is_explicit_and_profiled(channel: str) -> None:
    response = extract_windowed_magnetometer_features(
        request_for(measured_series(), channel=channel)
    )

    assert response.profile.channel == channel
    assert response.profile.readout == "measured_field"


def test_clipping_is_flagged_and_excluded_from_quantum_preview() -> None:
    response = extract_windowed_magnetometer_features(
        request_for(measured_series(clipped_index=10))
    )

    first = response.windows[0]
    assert first.quality.status == "invalid"
    assert first.quality.valid_for_quantum is False
    assert "clipped" in first.quality.flags
    assert first.quality.saturation_fraction == pytest.approx(0.01)
    assert first.quality.per_feature_valid == (
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    )


def test_almost_constant_windows_return_finite_values_with_invalid_flags() -> None:
    response = extract_windowed_magnetometer_features(
        request_for(measured_series(constant=True))
    )

    assert response.valid_window_count == 0
    for window in response.windows:
        assert window.quality.valid_for_quantum is False
        assert "almost_constant" in window.quality.flags
        assert all(math.isfinite(value) for value in window.features.values)


def test_observable_schema_rejects_generator_truth_and_nonuniform_time() -> None:
    payload = measured_series().model_dump()
    payload["earth_field_world"] = [[0.0, 0.0, 0.0]] * 400
    with pytest.raises(ValidationError, match="extra_forbidden"):
        MeasuredVectorSeries.model_validate(payload)

    payload.pop("earth_field_world")
    payload["time_s"][20] += 0.001
    with pytest.raises(ValidationError, match="uniformly spaced"):
        MeasuredVectorSeries.model_validate(payload)


def test_window_requires_at_least_sixteen_samples() -> None:
    request = request_for(measured_series())
    request = request.model_copy(
        update={
            "window": FeatureWindowConfiguration(
                duration_s=0.1,
                overlap_fraction=0.0,
            )
        }
    )

    with pytest.raises(ValueError, match="at least 16"):
        extract_windowed_magnetometer_features(request)


def test_excessive_overlapping_window_count_is_rejected() -> None:
    sampling_rate = 100.0
    sample_count = 3_000
    time = np.arange(sample_count, dtype=float) / sampling_rate
    field = np.column_stack(
        (
            np.sin(2.0 * np.pi * 8.0 * time),
            np.zeros(sample_count),
            np.zeros(sample_count),
        )
    )
    series = MeasuredVectorSeries(
        acquisition_id="large-overlap",
        sensor_id="sensor-001",
        sampling_rate_hz=sampling_rate,
        time_s=time.tolist(),
        measured_field=field.tolist(),
        temperature_k=np.full(sample_count, 293.15).tolist(),
        saturation_mask=np.zeros((sample_count, 3), dtype=bool).tolist(),
    )
    request = FeatureExtractionRequest(
        series=series,
        channel="x",
        window=FeatureWindowConfiguration(
            duration_s=1.0,
            overlap_fraction=0.99,
        ),
    )

    with pytest.raises(ValueError, match="more than 2048"):
        extract_windowed_magnetometer_features(request)


def test_clipping_rejection_cannot_be_disabled_by_api_input() -> None:
    payload = FeatureWindowConfiguration().model_dump()
    payload["reject_clipped"] = False

    with pytest.raises(ValidationError):
        FeatureWindowConfiguration.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_cycles", 0.0),
        ("minimum_peak_prominence_db", 0.0),
        ("minimum_snr_db", -300.0),
    ],
)
def test_canonical_quality_thresholds_cannot_be_weakened(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        FeatureWindowConfiguration.model_validate({field: value})
