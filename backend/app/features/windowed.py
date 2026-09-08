from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.features.models import (
    FeatureExtractionRequest,
    FeatureExtractionResponse,
    FeatureProfile,
    FeatureQuality,
    WindowFeatureRecord,
)
from app.features.provenance import sign_window_record
from app.preprocessing import compute_frequency_spectrum, extract_magnetometer_features
from app.sensors.models import SensorAcquisition

TESLA_TO_NANOTESLA = 1.0e9
MINIMUM_WINDOW_SAMPLES = 16
MAXIMUM_FEATURE_WINDOWS = 2_048
CONSTANT_SIGNAL_ABSOLUTE_TOLERANCE_NT = 1.0e-9


def _channel_values(request: FeatureExtractionRequest) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(request.series.measured_field, dtype=np.float64)
    if request.series.field_unit == "T":
        field = field * TESLA_TO_NANOTESLA

    channel_index = {"x": 0, "y": 1, "z": 2}.get(request.channel)
    if channel_index is None:
        signal = np.linalg.norm(field, axis=1)
        clipped = np.any(np.asarray(request.series.saturation_mask, dtype=bool), axis=1)
    else:
        signal = field[:, channel_index]
        clipped = np.asarray(request.series.saturation_mask, dtype=bool)[:, channel_index]
    return signal, clipped


def _peak_prominence_db(power: np.ndarray) -> float:
    non_dc = power[1:]
    if non_dc.size == 0:
        return -300.0
    peak = float(np.max(non_dc))
    baseline = float(np.median(non_dc))
    scale = max(peak, baseline, 1.0)
    floor = np.finfo(np.float64).eps * scale
    if peak <= floor:
        return -300.0
    return float(np.clip(10.0 * np.log10(peak / max(baseline, floor)), -300.0, 300.0))


def _quality(
    *,
    signal: np.ndarray,
    clipped: np.ndarray,
    frequency_hz: float,
    snr_db: float,
    prominence_db: float,
    duration_s: float,
    request: FeatureExtractionRequest,
) -> FeatureQuality:
    flags: list[str] = []
    standard_deviation = float(np.std(signal, ddof=0))
    numerical_tolerance = max(
        CONSTANT_SIGNAL_ABSOLUTE_TOLERANCE_NT,
        np.finfo(np.float64).eps * max(float(np.max(np.abs(signal))), 1.0) * 64.0,
    )
    saturation_fraction = float(np.mean(clipped))
    cycles = max(0.0, float(frequency_hz * duration_s))

    if standard_deviation <= numerical_tolerance:
        flags.append("almost_constant")
    if cycles < request.window.minimum_cycles:
        flags.append("insufficient_cycles")
    if prominence_db < request.window.minimum_peak_prominence_db:
        flags.append("low_peak_prominence")
    if snr_db < request.window.minimum_snr_db:
        flags.append("low_harmonic_snr")
    if saturation_fraction > 0.0:
        flags.append("clipped")

    harmonic_invalid_flags = {
        "almost_constant",
        "insufficient_cycles",
        "low_peak_prominence",
        "low_harmonic_snr",
    }
    harmonic_valid = not any(flag in harmonic_invalid_flags for flag in flags)
    clipped_valid = saturation_fraction == 0.0
    per_feature_valid = (
        harmonic_valid and clipped_valid,  # amplitude
        harmonic_valid and clipped_valid,  # phase
        harmonic_valid and clipped_valid,  # frequency
        clipped_valid,  # variance
        clipped_valid,  # drift
        harmonic_valid and clipped_valid,  # snr
        clipped_valid,  # spectral peak
        True,  # observed temperature
    )
    valid = all(per_feature_valid)
    status = "valid" if valid and not flags else "warning" if valid else "invalid"
    return FeatureQuality(
        status=status,
        valid_for_quantum=valid,
        flags=tuple(flags),
        per_feature_valid=per_feature_valid,
        sample_count=signal.size,
        saturation_fraction=saturation_fraction,
        cycles_in_window=cycles,
        peak_prominence_db=prominence_db,
        signal_standard_deviation_nt=standard_deviation,
    )


def extract_windowed_magnetometer_features(
    request: FeatureExtractionRequest,
) -> FeatureExtractionResponse:
    """Extract one legacy-compatible 8D vector per complete causal window."""

    sampling_rate = request.series.sampling_rate_hz
    window_samples = int(round(request.window.duration_s * sampling_rate))
    if window_samples < MINIMUM_WINDOW_SAMPLES:
        raise ValueError(
            f"window duration must contain at least {MINIMUM_WINDOW_SAMPLES} samples"
        )
    if window_samples > len(request.series.time_s):
        raise ValueError("window duration exceeds the measured acquisition")
    hop_samples = max(
        1,
        int(round(window_samples * (1.0 - request.window.overlap_fraction))),
    )
    window_count = 1 + (len(request.series.time_s) - window_samples) // hop_samples
    if window_count > MAXIMUM_FEATURE_WINDOWS:
        raise ValueError(
            "window configuration would produce more than "
            f"{MAXIMUM_FEATURE_WINDOWS} feature vectors"
        )

    signal, clipped = _channel_values(request)
    time = np.asarray(request.series.time_s, dtype=np.float64)
    temperature = np.asarray(request.series.temperature_k, dtype=np.float64)
    records: list[WindowFeatureRecord] = []
    latest_end = 0
    profile = FeatureProfile(
        channel=request.channel,
        sampling_rate_hz=sampling_rate,
        window_duration_s=request.window.duration_s,
        overlap_fraction=request.window.overlap_fraction,
        window_samples=window_samples,
        hop_samples=hop_samples,
        minimum_cycles=request.window.minimum_cycles,
        minimum_peak_prominence_db=request.window.minimum_peak_prominence_db,
        minimum_snr_db=request.window.minimum_snr_db,
    )

    for start in range(0, len(time) - window_samples + 1, hop_samples):
        end = start + window_samples
        latest_end = end
        local_time = time[start:end] - time[start]
        local_signal = signal[start:end]
        acquisition = SensorAcquisition(
            sensor_id=request.series.sensor_id,
            sensor_type="quantum_magnetometer",
            timestamp=datetime.now(timezone.utc),
            sampling_rate=sampling_rate,
            time=local_time.tolist(),
            signal=local_signal.tolist(),
            time_unit="s",
            physical_unit="nT",
            configuration={},
            metadata={"temperature": float(np.mean(temperature[start:end]))},
        )
        spectrum = compute_frequency_spectrum(acquisition)
        features = extract_magnetometer_features(acquisition, spectrum)
        by_name = dict(zip(features.names, features.values))
        prominence = _peak_prominence_db(
            np.asarray(spectrum.power_spectral_density, dtype=np.float64)
        )
        duration_s = window_samples / sampling_rate
        quality = _quality(
            signal=local_signal,
            clipped=clipped[start:end],
            frequency_hz=float(by_name["frequency"]),
            snr_db=float(by_name["snr"]),
            prominence_db=prominence,
            duration_s=duration_s,
            request=request,
        )
        record = WindowFeatureRecord(
            window_id=(
                f"{request.series.acquisition_id}:{request.channel}:{start}:{end}"
            ),
            acquisition_id=request.series.acquisition_id,
            sensor_id=request.series.sensor_id,
            start_index=start,
            end_index=end,
            start_time_s=float(time[start]),
            end_time_s=float(time[end - 1]),
            center_time_s=float((time[start] + time[end - 1]) / 2.0),
            features=features,
            quality=quality,
            provenance_token="",
        )
        records.append(
            record.model_copy(
                update={"provenance_token": sign_window_record(profile, record)}
            )
        )

    valid_count = sum(record.quality.valid_for_quantum for record in records)
    discarded = len(time) - latest_end if records else len(time)
    warnings = []
    if discarded:
        warnings.append(
            f"{discarded} trailing samples were excluded because only complete windows are used"
        )
    return FeatureExtractionResponse(
        profile=profile,
        windows=records,
        valid_window_count=valid_count,
        invalid_window_count=len(records) - valid_count,
        discarded_sample_count=discarded,
        warnings=warnings,
    )
