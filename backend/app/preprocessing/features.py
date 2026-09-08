from __future__ import annotations

import numpy as np

from app.preprocessing.signal import compute_frequency_spectrum, estimate_dominant_frequency
from app.sensors.models import FeatureVector, FrequencySpectrum, SensorAcquisition

MAGNETOMETER_FEATURE_NAMES = (
    "amplitude",
    "phase",
    "frequency",
    "variance",
    "drift",
    "snr",
    "spectral_peak",
    "temperature",
)
MAGNETOMETER_FEATURE_UNITS = (
    "nT",
    "rad",
    "Hz",
    "nT²",
    "nT/s",
    "dB",
    "nT²/Hz",
    "K",
)
MAX_ABSOLUTE_SNR_DB = 300.0


def _harmonic_fit(
    time: np.ndarray,
    signal: np.ndarray,
    frequency: float,
) -> tuple[float, float, float, float, float]:
    centered_time = time - np.mean(time)
    if frequency <= 0.0:
        design = np.column_stack((np.ones(time.size), centered_time))
        coefficients, *_ = np.linalg.lstsq(design, signal, rcond=None)
        fitted = design @ coefficients
        residual_power = float(np.mean((signal - fitted) ** 2))
        return 0.0, 0.0, float(coefficients[1]), 0.0, residual_power

    angular_frequency = 2.0 * np.pi * frequency
    sine = np.sin(angular_frequency * time)
    cosine = np.cos(angular_frequency * time)
    design = np.column_stack((np.ones(time.size), centered_time, sine, cosine))
    coefficients, *_ = np.linalg.lstsq(design, signal, rcond=None)
    fitted = design @ coefficients
    sine_coefficient = float(coefficients[2])
    cosine_coefficient = float(coefficients[3])
    amplitude = float(np.hypot(sine_coefficient, cosine_coefficient))
    phase = float(np.arctan2(cosine_coefficient, sine_coefficient))
    drift = float(coefficients[1])
    periodic = sine_coefficient * sine + cosine_coefficient * cosine
    periodic_power = float(np.mean(periodic**2))
    residual_power = float(np.mean((signal - fitted) ** 2))
    return amplitude, phase, drift, periodic_power, residual_power


def _finite_snr_db(periodic_power: float, residual_power: float) -> float:
    reference_power = max(periodic_power, residual_power, 1.0)
    floor = np.finfo(np.float64).eps * reference_power
    if periodic_power <= floor and residual_power <= floor:
        return 0.0

    ratio = max(periodic_power, floor) / max(residual_power, floor)
    return float(
        np.clip(
            10.0 * np.log10(ratio),
            -MAX_ABSOLUTE_SNR_DB,
            MAX_ABSOLUTE_SNR_DB,
        )
    )


def extract_magnetometer_features(
    acquisition: SensorAcquisition,
    spectrum: FrequencySpectrum | None = None,
) -> FeatureVector:
    """Estimate the ordered 8-feature magnetometer representation.

    No normalization or quantum-side scaling is applied here.
    """

    if acquisition.sensor_type != "quantum_magnetometer":
        raise ValueError("magnetometer feature extraction requires a magnetometer")

    spectrum = spectrum or compute_frequency_spectrum(acquisition)
    frequency = estimate_dominant_frequency(spectrum)
    time = np.asarray(acquisition.time, dtype=np.float64)
    signal = np.asarray(acquisition.signal, dtype=np.float64)
    amplitude, phase, drift, periodic_power, residual_power = _harmonic_fit(
        time,
        signal,
        frequency,
    )
    variance = float(np.var(signal, ddof=0))
    snr = _finite_snr_db(periodic_power, residual_power)
    spectral_peak = float(max(spectrum.power_spectral_density[1:], default=0.0))

    temperature = acquisition.metadata.get("temperature")
    if not isinstance(temperature, (int, float)):
        raise ValueError("magnetometer acquisition metadata must include temperature")

    return FeatureVector(
        names=MAGNETOMETER_FEATURE_NAMES,
        values=(
            amplitude,
            phase,
            frequency,
            variance,
            drift,
            snr,
            spectral_peak,
            float(temperature),
        ),
        units=MAGNETOMETER_FEATURE_UNITS,
    )
