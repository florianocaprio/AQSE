from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.sensors.models import FrequencySpectrum, SensorAcquisition


def _raw_arrays(
    acquisition: SensorAcquisition,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    time = np.asarray(acquisition.time, dtype=np.float64)
    signal = np.asarray(acquisition.signal, dtype=np.float64)
    if time.ndim != 1 or signal.ndim != 1:
        raise ValueError("sensor time and signal must be one-dimensional")
    if time.size != signal.size or time.size < 2:
        raise ValueError("sensor time and signal must contain matching samples")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        raise ValueError("sensor time and signal must contain finite values")
    expected_period = 1.0 / acquisition.sampling_rate
    tolerance = max(1.0e-12, expected_period * 1.0e-9)
    if not np.allclose(
        np.diff(time),
        expected_period,
        rtol=1.0e-9,
        atol=tolerance,
    ):
        raise ValueError(
            "frequency analysis requires uniformly spaced time samples "
            "consistent with sampling_rate"
        )
    return time, signal


def remove_linear_trend(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Remove the least-squares intercept and linear trend from a signal."""

    centered_time = time - np.mean(time)
    design = np.column_stack((np.ones(time.size), centered_time))
    coefficients, *_ = np.linalg.lstsq(design, signal, rcond=None)
    return signal - design @ coefficients


def compute_frequency_spectrum(acquisition: SensorAcquisition) -> FrequencySpectrum:
    """Calculate a Hann-windowed, one-sided PSD after linear detrending."""

    time, signal = _raw_arrays(acquisition)
    detrended = remove_linear_trend(time, signal)
    sample_count = signal.size
    window = np.hanning(sample_count) if sample_count >= 4 else np.ones(sample_count)
    window_power = float(np.sum(window**2))
    if window_power == 0.0:
        window = np.ones(sample_count)
        window_power = float(sample_count)

    transform = np.fft.rfft(detrended * window)
    power = np.abs(transform) ** 2 / (acquisition.sampling_rate * window_power)
    if sample_count % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0

    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / acquisition.sampling_rate)
    return FrequencySpectrum(
        frequencies=frequencies.tolist(),
        power_spectral_density=power.astype(np.float64).tolist(),
    )


def estimate_dominant_frequency(spectrum: FrequencySpectrum) -> float:
    """Estimate the non-DC PSD peak frequency with parabolic interpolation."""

    frequencies = np.asarray(spectrum.frequencies, dtype=np.float64)
    power = np.asarray(spectrum.power_spectral_density, dtype=np.float64)
    if frequencies.size < 2 or power.size < 2:
        return 0.0

    peak_index = int(np.argmax(power[1:])) + 1
    peak_power = float(power[peak_index])
    scale = max(float(np.max(power)), 1.0)
    if peak_power <= np.finfo(np.float64).eps * scale:
        return 0.0

    frequency = float(frequencies[peak_index])
    if 0 < peak_index < power.size - 1:
        left = float(power[peak_index - 1])
        center = peak_power
        right = float(power[peak_index + 1])
        denominator = left - 2.0 * center + right
        if abs(denominator) > np.finfo(np.float64).eps * scale:
            offset = 0.5 * (left - right) / denominator
            offset = float(np.clip(offset, -0.5, 0.5))
            bin_width = float(frequencies[peak_index + 1] - frequencies[peak_index])
            frequency += offset * bin_width
    return frequency
