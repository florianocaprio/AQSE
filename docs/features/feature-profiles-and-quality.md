# AQSE Feature Profiles, Windowing, and Quality

## Purpose

AQSE always sends eight finite numeric values to TQK8, but vector length alone
does not make two feature sets semantically compatible. Every vector therefore
belongs to a versioned FeatureProfile that fixes source type, readout, selected
channel, names, units, window policy, extractor version, and quality policy.

This document distinguishes the validated legacy profile, the finite-vector
legacy-compatible profile approved for Milestone 1C, and a separate candidate
continuous-network profile. The candidate network profile is not silently
activated and is not a replacement for the legacy scientific contract. The
implemented continuous simulator/session API does not make the 4 s / 1 s-hop
network feature profile implemented.

## Profile metadata

A current Milestone 1C `FeatureProfile` records:

    profile_id
    extractor_version
    sensor_type
    readout and selected channel
    feature_names[8]
    feature_units[8]
    sampling_rate_hz
    window_duration_s
    window_samples
    overlap_fraction and hop_samples
    phase_reference
    quality_policy_version
    fixed quality thresholds

Each current window record adds acquisition and sensor identifiers, sample-index
and time boundaries, features, quality, and a required provenance token. Session,
observation-schema, and calibration lineage must be added before this contract
is used as a durable multi-session training artifact. Profile identity is part
of model compatibility, not a display label.

## Legacy scalar profile

The validated Milestone 1B order is immutable:

| Index | Name | Meaning | Unit |
| ---: | --- | --- | --- |
| f0 | amplitude | Magnitude of fitted sine and cosine coefficients | nT |
| f1 | phase | Fitted harmonic phase referenced to window time zero | rad |
| f2 | frequency | Dominant non-DC frequency | Hz |
| f3 | variance | Population variance of the measured channel | nT² |
| f4 | drift | Linear coefficient of the joint fit | nT/s |
| f5 | snr | Fitted periodic power relative to residual power | dB |
| f6 | spectral_peak | Maximum non-DC one-sided PSD value | nT²/Hz |
| f7 | temperature | Observed acquisition temperature | K |

In particular, spectral_peak is a PSD magnitude. It is not the frequency at
which the peak occurs. Frequency remains f2.

The scalar implementation works on a finite acquisition and applies no quantum
normalization. Its current formulas and numerical finite-value conventions are
documented in [Quantum Magnetometer Simulator](../sensors/magnetometer.md).

## Finite-vector legacy-compatible profile

Milestone 1C applies the same eight meanings to one explicitly selected measured
channel from a vector acquisition:

- X sensor-frame component;
- Y sensor-frame component;
- Z sensor-frame component;
- measured magnitude |B| = sqrt(Bx² + By² + Bz²).

It never averages X, Y, and Z, concatenates them into 24 values, or implicitly
flattens a vector. Magnitude is calculated from measured components after sensor
effects; it is not copied from environment truth.

The approved default finite-vector window policy is:

- duration: 1 s;
- overlap: 50%;
- complete windows only;
- sample count: round(sampling_rate × 1 s);
- hop: round(window_samples × 0.5), with a minimum of one sample;
- timestamp: centre of the accepted window.

No padding is applied at the start or end. Dropped remainder samples are counted
in batch metadata. A different duration or overlap creates a different profile
instance even when the eight names remain unchanged.

New network physics uses tesla internally. Before invoking the legacy-compatible
extractor, the selected field channel is converted once and explicitly to nT.
The resulting feature units therefore remain the legacy units above. The input
unit is explicit on the request, but the current output profile does not retain a
separate source-unit/conversion field; durable lineage must add it.

For a varying temperature trace, f7 is the arithmetic mean of observed device
temperature within the feature window. It is not simulator truth and not merely
the configured setpoint.

## Causal window construction

An online window ending at simulated time t may contain only observations whose
acquisition times are no later than t. AQSE must not use a centred smoother,
future samples, or final-session statistics to compute an online feature.

The implemented finite-vector extractor:

1. accepts one already selected node and declared measured readout;
2. preserves the submitted sample order;
3. rejects a series whose timestamps are not strictly increasing, uniformly
   spaced, and consistent with the declared sample rate;
4. constructs complete windows without padding;
5. passes only measured samples and observed temperature to the extractor;
6. records quality and signed provenance alongside the values.

The current request schema accepts complete finite arrays only. Gap-tolerant
windowing, null samples, duplicate/frozen classification, clock-anomaly handling,
and historical calibration selection are future extensions and must not be
presented as current reason codes.

Correlation or relational features require aligned, compatible world-frame
quantities and at least two informative samples. Two raw sensor axes must not be
correlated as if they represented the same physical direction unless pose and
calibration transformations have made that comparison valid.

## Feature quality contract

A finite fallback prevents NaN and Infinity in JSON, but a fallback is not a
physical measurement. Each window therefore carries structured quality:

    status: valid | warning | invalid
    flags: composable reason codes
    per_feature_valid: boolean[8]
    valid_for_quantum: boolean
    metrics:
      sample_count
      saturation_fraction
      cycles_in_window
      peak_prominence_db
      signal_standard_deviation_nt

The current quality-policy reason codes are:

- almost_constant;
- insufficient_cycles;
- low_peak_prominence;
- low_harmonic_snr;
- clipped.

Non-finite, too-short, non-uniform, or otherwise malformed inputs are rejected by
the request/window contract before a window record is produced. Missing, frozen,
transient-dominated, broadband-dominated, stale-calibration, and unavailable
relational-context states require future, versioned quality policies.

### Approved finite-vector eligibility policy

The Milestone 1C `aqse.harmonic-quality.v1` policy is fixed by the API schema and
is not request-tunable. Its sample bound and eligibility thresholds require:

- at least 16 samples in every complete finite window;
- at least 2 estimated cycles within the window for harmonic features;
- spectral peak prominence of at least 6 dB;
- estimated SNR of at least 0 dB;
- zero clipped samples.

Peak prominence is calculated from measured data as

    10 log10(P_peak / P_background),

where P_peak is the dominant non-DC PSD value and, in the implemented v1 policy,
P_background is the median of all non-DC PSD bins. The numerical floor is tied
to floating-point scale. Excluding a guard neighbourhood or using a robust
spectral background estimator would require a new quality-policy version.

Any clipping makes the initial window ineligible for quantum preview. It can
still be displayed with the `clipped` flag. Later research
may approve a different clipping threshold under a new policy version.

If no reliable harmonic exists, amplitude, phase, frequency, and SNR are marked
invalid even though the serialized vector contains a documented finite fallback.
Variance, drift, or temperature may remain individually valid. The UI displays
their validity separately and does not interpret a fallback zero as a measured
zero.

Transient or broadband dominance is not classified by the current policy. A
future policy may derive it only from observations, for example through
versioned crest-factor and spectral-concentration metrics. The selected scenario
name or hidden event cause must never determine the flag.

## Missing data

At the network observation boundary an absent sample is null with quality
metadata, not zero. The current legacy-compatible feature request does not
accept null or partial samples, so such data must be rejected before extraction
and cannot produce a quantum-eligible feature record.

Any future partially observed profile must state:

- minimum valid fraction;
- permitted interpolation method, if any;
- maximum gap duration;
- mask or indicator behavior;
- which features remain meaningful.

Such interpolation must use only causal neighbours in online operation. Imputation
parameters are learned on the reference or training data and frozen for
evaluation. Missing relational context from a single-node or degenerate network
is not interpreted as zero anomaly.

## Candidate continuous-network profile

The continuous network uses an operational starting point of 4 s windows with a
1 s hop. It requires a distinct profile because its eight proposed quantities
do not have the legacy harmonic semantics.

| Index | Candidate meaning | Example unit |
| ---: | --- | --- |
| f0 | Mean calibrated channel anomaly relative to a declared nominal reference | T |
| f1 | Robust measured dispersion | T |
| f2 | Causal temporal slope | T/s |
| f3 | Power ratio between configured physical bands | dimensionless |
| f4 | Observed device-temperature deviation from calibration | K |
| f5 | Leave-one-sensor-out spatial residual when identifiable | T |
| f6 | Coherence with an estimated common field or declared reference | dimensionless |
| f7 | Valid-sample fraction | dimensionless |

This table is a design candidate, not an implemented or scientifically frozen
extractor. Band definitions, robust estimator, spatial model, coherence method,
missing-context representation, and quality thresholds require validation and
a dedicated profile version before any model is trained.

The implemented `quantum_preview_signal` network preset is an explicit 5 Hz
validation excitation used to produce eligible windows with the existing
legacy-compatible harmonic profile. It does not implement, validate, or activate
this candidate network profile.

For one node or insufficient geometry, f5 and possibly f6 are unavailable. AQSE
must define an explicit mask and degraded path; it must not set them to zero and
claim that the residual or correlation is absent. The feature schema, mask
policy, scaler, and trained model are versioned together.

Legacy finite-vector and continuous-network vectors cannot share a checkpoint
merely because both have length eight.

## Anti-leakage boundary

Feature code can access only:

- observations acquired by the window end;
- observed device telemetry;
- nominal or measured pose explicitly available to the real pipeline;
- calibration and reference state fitted from permitted prior data;
- network observations aligned causally.

It cannot access:

- exact B_true or unobserved environment components;
- injected event or fault cause;
- true dipole position, moment, or trajectory;
- hidden bias, random-walk state, or real device parameters;
- session seed or random substream state;
- future observations;
- test-set statistics.

An automated anti-leakage test must show that identical ObservationFrames give
identical features when TruthFrames and injected cause labels are changed or
removed.

## Feature-record provenance

Every current window record carries a required HMAC token over the complete
feature profile and the record fields other than the token itself. The quantum
preview verifies this token server-side before accepting a record; changing the
features, quality, identifiers, timing, or profile invalidates it.

The signing key is intentionally ephemeral and generated at backend process
start. Records copied through a browser must therefore be recomputed after a
backend restart. This mechanism protects in-process feature-record integrity; it
does not authenticate a physical sensor, prove raw-data provenance, or provide a
durable archival signature. Those require separately managed identities, keys,
and immutable acquisition lineage.

## Scaling and model compatibility

Raw feature values and units remain in the feature record and preview response;
durable archival is not implemented. Feature extraction does not apply a
per-window min–max transformation or quantum angle scaling. Such scaling could
erase the bias and drift that the system is meant to detect.

The supplied AngleScaler is fitted only on an explicitly selected reference or
training dataset, then frozen. Query, validation, and test samples reuse that
snapshot. The snapshot records profile, mean, scale, reference dataset, and
version. Windows from a different feature profile are rejected rather than
coerced into the existing model.

Overlapping windows from one episode stay in the same train, validation, or test
partition. Splitting them randomly would leak nearly identical data across the
evaluation boundary.
