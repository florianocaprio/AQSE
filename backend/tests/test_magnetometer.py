import numpy as np
import pytest
from pydantic import ValidationError

from app.preprocessing import (
    MAGNETOMETER_FEATURE_NAMES,
    compute_frequency_spectrum,
    extract_magnetometer_features,
)
from app.sensors.magnetometer import QuantumMagnetometerSimulator
from app.sensors.models import FeatureVector, MagnetometerConfiguration


def configuration(**overrides: object) -> MagnetometerConfiguration:
    values = MagnetometerConfiguration().model_dump()
    values.update(overrides)
    return MagnetometerConfiguration.model_validate(values)


def test_generated_sample_count_matches_duration_and_sampling_rate() -> None:
    acquisition = QuantumMagnetometerSimulator().acquire(
        configuration(duration=0.5, sampling_rate=100.0, frequency=8.0)
    )

    assert acquisition.sample_count == 50
    assert len(acquisition.time) == len(acquisition.signal) == 50
    assert acquisition.time[0] == 0.0
    assert acquisition.time[-1] == pytest.approx(0.49)


def test_same_seed_repeats_signal_and_different_seed_changes_noise() -> None:
    simulator = QuantumMagnetometerSimulator()
    first = simulator.acquire(configuration(random_seed=7, noise_std=3.0))
    repeated = simulator.acquire(configuration(random_seed=7, noise_std=3.0))
    different = simulator.acquire(configuration(random_seed=8, noise_std=3.0))

    np.testing.assert_array_equal(first.signal, repeated.signal)
    assert not np.array_equal(first.signal, different.signal)


def test_zero_noise_signal_matches_the_configured_analytic_model() -> None:
    config = configuration(
        duration=1.0,
        sampling_rate=128.0,
        background_field=25.0,
        amplitude=3.0,
        frequency=4.0,
        phase=0.3,
        drift_rate=0.4,
        noise_std=0.0,
        anomaly_enabled=False,
    )
    acquisition = QuantumMagnetometerSimulator().acquire(config)
    time = np.asarray(acquisition.time)
    expected = (
        config.background_field
        + config.amplitude * np.sin(2.0 * np.pi * config.frequency * time + config.phase)
        + config.drift_rate * time
    )

    np.testing.assert_allclose(acquisition.signal, expected, atol=1e-12, rtol=0.0)


def test_optional_anomaly_adds_a_gaussian_transient_at_configured_time() -> None:
    common = {
        "duration": 1.0,
        "sampling_rate": 200.0,
        "amplitude": 0.0,
        "drift_rate": 0.0,
        "noise_std": 0.0,
        "anomaly_time": 0.5,
        "anomaly_amplitude": 25.0,
    }
    simulator = QuantumMagnetometerSimulator()
    baseline = simulator.acquire(configuration(**common, anomaly_enabled=False))
    anomalous = simulator.acquire(configuration(**common, anomaly_enabled=True))
    difference = np.asarray(anomalous.signal) - np.asarray(baseline.signal)
    peak_index = int(np.argmax(difference))

    assert anomalous.time[peak_index] == pytest.approx(0.5)
    assert difference[peak_index] == pytest.approx(25.0)


def test_drift_is_recovered_from_a_noise_free_linear_signal() -> None:
    config = configuration(
        background_field=10.0,
        amplitude=0.0,
        drift_rate=-1.25,
        noise_std=0.0,
    )
    acquisition = QuantumMagnetometerSimulator().acquire(config)
    features = extract_magnetometer_features(acquisition)
    drift_index = features.names.index("drift")

    assert features.values[drift_index] == pytest.approx(config.drift_rate, abs=1e-10)


def test_zero_signal_has_no_arbitrary_frequency_and_all_features_are_finite() -> None:
    acquisition = QuantumMagnetometerSimulator().acquire(
        configuration(
            background_field=0.0,
            amplitude=0.0,
            drift_rate=0.0,
            noise_std=0.0,
        )
    )
    features = extract_magnetometer_features(acquisition)

    assert features.values[features.names.index("frequency")] == 0.0
    assert features.values[features.names.index("spectral_peak")] == 0.0
    assert features.values[features.names.index("snr")] == 0.0
    assert np.all(np.isfinite(features.values))


def test_harmonic_fit_recovers_amplitude_phase_and_drift() -> None:
    config = configuration(
        duration=4.0,
        sampling_rate=200.0,
        amplitude=7.5,
        frequency=10.0,
        phase=0.7,
        drift_rate=-0.35,
        noise_std=0.0,
    )
    acquisition = QuantumMagnetometerSimulator().acquire(config)
    features = extract_magnetometer_features(acquisition)
    by_name = dict(zip(features.names, features.values))

    assert by_name["amplitude"] == pytest.approx(config.amplitude, abs=1e-6)
    assert by_name["phase"] == pytest.approx(config.phase, abs=1e-6)
    assert by_name["drift"] == pytest.approx(config.drift_rate, abs=1e-6)


def test_off_bin_frequency_and_fft_peak_are_recovered() -> None:
    config = configuration(
        duration=4.0,
        sampling_rate=200.0,
        amplitude=12.0,
        frequency=12.37,
        drift_rate=2.0,
        noise_std=0.1,
        random_seed=19,
    )
    acquisition = QuantumMagnetometerSimulator().acquire(config)
    spectrum = compute_frequency_spectrum(acquisition)
    features = extract_magnetometer_features(acquisition, spectrum)
    frequency_index = features.names.index("frequency")
    fft_peak_index = int(np.argmax(spectrum.power_spectral_density[1:])) + 1
    bin_width = spectrum.frequencies[1] - spectrum.frequencies[0]
    spectral_peak_index = features.names.index("spectral_peak")

    assert features.values[frequency_index] == pytest.approx(config.frequency, abs=0.05)
    assert abs(spectrum.frequencies[fft_peak_index] - config.frequency) <= bin_width / 2
    assert features.values[spectral_peak_index] == max(
        spectrum.power_spectral_density[1:]
    )


def test_snr_is_finite_and_decreases_as_noise_increases() -> None:
    common = {
        "duration": 4.0,
        "sampling_rate": 200.0,
        "amplitude": 10.0,
        "frequency": 9.37,
        "drift_rate": 0.2,
        "random_seed": 2026,
    }
    simulator = QuantumMagnetometerSimulator()
    noise_free = extract_magnetometer_features(
        simulator.acquire(configuration(**common, noise_std=0.0))
    )
    low_noise = extract_magnetometer_features(
        simulator.acquire(configuration(**common, noise_std=0.5))
    )
    high_noise = extract_magnetometer_features(
        simulator.acquire(configuration(**common, noise_std=5.0))
    )
    snr_index = noise_free.names.index("snr")

    assert np.isfinite(noise_free.values[snr_index])
    assert low_noise.values[snr_index] > high_noise.values[snr_index]


def test_frequency_analysis_rejects_non_uniform_time_samples() -> None:
    acquisition = QuantumMagnetometerSimulator().acquire(configuration())
    non_uniform_time = list(acquisition.time)
    non_uniform_time[10] += 0.001
    non_uniform_acquisition = acquisition.model_copy(
        update={"time": non_uniform_time}
    )

    with pytest.raises(ValueError, match="uniformly spaced"):
        compute_frequency_spectrum(non_uniform_acquisition)


def test_variance_is_calculated_from_the_raw_signal() -> None:
    acquisition = QuantumMagnetometerSimulator().acquire(
        configuration(noise_std=1.3, random_seed=2026)
    )
    features = extract_magnetometer_features(acquisition)
    variance_index = features.names.index("variance")

    assert features.values[variance_index] == pytest.approx(
        np.var(acquisition.signal, ddof=0)
    )


def test_feature_vector_has_exactly_eight_named_finite_values() -> None:
    acquisition = QuantumMagnetometerSimulator().acquire(configuration(noise_std=0.0))
    features = extract_magnetometer_features(acquisition)

    assert features.names == MAGNETOMETER_FEATURE_NAMES
    assert len(features.values) == len(features.units) == 8
    assert np.all(np.isfinite(features.values))

    for size in (7, 9):
        with pytest.raises(ValidationError, match="exactly 8"):
            FeatureVector(
                names=tuple(f"f{index}" for index in range(size)),
                values=tuple(float(index) for index in range(size)),
                units=("unit",) * size,
            )
