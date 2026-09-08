# Quantum Magnetometer Simulator

## Scope

The Milestone 1B magnetometer is a configurable research simulator. It produces
a physically interpretable synthetic magnetic-field time series, but it does
not reproduce a specific commercial instrument and must not be treated as an
experimentally calibrated model.

The simulator is deliberately isolated from the AQSE TQK8 engine. Its output is
not normalized, injected into a VQC, used for training, or sent to a physical
QPU.

## Signal model

For sample time `t` in seconds, the generated magnetic field is:

```text
B(t) = background
     + amplitude × sin(2π × frequency × t + phase)
     + drift_rate × t
     + Gaussian noise
     + optional transient anomaly
```

The transient anomaly is a Gaussian pulse:

```text
anomaly(t) = anomaly_amplitude × exp(-0.5 × ((t - anomaly_time) / σ)²)
σ = max(3 / sampling_rate, 0.02 × duration)
```

The random component is generated from `random_seed`, so equal configurations
with the same seed produce equal raw signals. Acquisition timestamps remain
independent and therefore differ between runs.

## Units

| Quantity | Unit |
| --- | --- |
| Time and duration | second (`s`) |
| Sampling rate and frequency | hertz (`Hz`) |
| Magnetic field, amplitude, noise and anomaly | nanotesla (`nT`) |
| Phase | radian (`rad`) |
| Linear drift | nanotesla per second (`nT/s`) |
| Temperature | kelvin (`K`) |
| Variance | square nanotesla (`nT²`) |
| Spectral density | square nanotesla per hertz (`nT²/Hz`) |
| Signal-to-noise ratio | decibel (`dB`) |

Nanotesla is an SI-compatible submultiple of tesla and keeps the simulator and
visualization values readable.

## Processing pipeline

The implementation keeps the following stages separate:

```text
Magnetometer model → raw acquisition → signal analysis → FeatureVector
```

Frequency analysis removes a preliminary linear trend, applies a Hann window,
and computes a one-sided power spectral density (PSD) with NumPy's real FFT.
The dominant non-DC bin provides the frequency candidate. Three-point parabolic
interpolation around that PSD maximum refines off-bin frequency estimates. A
joint least-squares fit then models the raw samples with an intercept, centered
linear trend, sine, and cosine at that refined frequency.

## Feature definitions

The output order is fixed and contains exactly eight unnormalized values:

| Position | Name | Calculation | Unit |
| ---: | --- | --- | --- |
| `f0` | `amplitude` | Magnitude of the fitted sine/cosine coefficients | `nT` |
| `f1` | `phase` | `atan2` phase derived from the fitted coefficients, in `[-π, π]` and referenced to `t=0` | `rad` |
| `f2` | `frequency` | Dominant non-DC frequency detected from the detrended spectrum | `Hz` |
| `f3` | `variance` | Population variance of the measured raw signal | `nT²` |
| `f4` | `drift` | Linear coefficient of the joint regression | `nT/s` |
| `f5` | `snr` | `10 log10(P_signal / P_residual)` from fitted sinusoid and regression residual, with a numerical floor and clamp to `[-300, 300]` that keep JSON finite | `dB` |
| `f6` | `spectral_peak` | Maximum one-sided PSD above DC | `nT²/Hz` |
| `f7` | `temperature` | Acquisition metadata supplied by the simulator configuration | `K` |

Except for temperature metadata, extracted features are estimated from the
generated time series rather than copied from simulator parameters. A
numerically constant signal uses the documented finite convention of zero for
amplitude, phase, frequency, SNR, and spectral peak.

## Validation and limits

- Frequency must be greater than zero and lower than the Nyquist frequency.
- Duration is at most 300 s and sampling rate is at most 100,000 Hz. The sample
  count is `round(duration × sampling_rate)` and must be from 16 to 20,000.
- Sinusoidal amplitude and noise standard deviation cannot be negative.
  Background field, drift and transient anomaly amplitude may be signed.
- Magnetic quantities are capped at an absolute value of `10¹²` in their
  documented units, phase at `10⁶ rad`, and temperature at `10⁶ K`. These are
  computational safety limits, not instrument specifications.
- Temperature must be greater than zero.
- If an anomaly is enabled, its center must fall inside the acquisition.
- Random seed is an integer from 0 to `2³² - 1`.
- Unknown configuration fields are rejected.
- Time and signal arrays always have equal length, and timestamps are UTC.
- Frequency analysis accepts only uniformly spaced samples consistent with the
  acquisition sampling rate.

The default acquisition is intentionally small enough for an interactive local
visualization. Real-time streaming, persistent storage, sensor networks, AFSE,
classification, and Sensor-to-Quantum execution belong to later milestones.

## API

- `GET /api/sensors` lists available simulator types.
- `GET /api/sensors/magnetometer/defaults` returns the validated defaults.
- `POST /api/sensors/magnetometer/simulate` returns acquisition metadata, raw
  time/signal vectors, the one-sided spectrum, and the ordered feature vector.

The default request fields are:

| Field | Default | Unit / meaning |
| --- | ---: | --- |
| `sensor_id` | `magnetometer-001` | acquisition identifier |
| `duration` | `2.0` | `s` |
| `sampling_rate` | `200.0` | `Hz` |
| `background_field` | `50000.0` | `nT` |
| `amplitude` | `100.0` | `nT` |
| `frequency` | `8.0` | `Hz` |
| `phase` | `0.0` | `rad` |
| `drift_rate` | `0.5` | `nT/s` |
| `noise_std` | `2.0` | `nT` |
| `temperature` | `293.15` | `K` |
| `anomaly_enabled` | `false` | enable Gaussian transient |
| `anomaly_time` | `1.0` | `s`, nullable while disabled |
| `anomaly_amplitude` | `40.0` | signed `nT` |
| `random_seed` | `42` | deterministic PRNG seed |

Successful simulation responses have this shape:

```text
{
  acquisition: {
    sensor_id, sensor_type, timestamp, sampling_rate,
    time: number[], signal: number[], time_unit, physical_unit,
    configuration, metadata, sample_count
  },
  spectrum: {
    frequencies: number[], power_spectral_density: number[],
    frequency_unit, power_unit
  },
  features: { names: string[8], values: number[8], units: string[8] }
}
```

Invalid bodies, unknown fields, unsafe ranges, Nyquist violations and invalid
anomaly timing return HTTP `422` with FastAPI/Pydantic validation details. The
interactive OpenAPI schema is available at `http://localhost:8000/docs`.
