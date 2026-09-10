# Scientific Scope, Claims, and Limitations

## Purpose

AQSE is a configurable local research demonstrator. It supports reproducible
experiments about hybrid classical and quantum processing of classically read
sensor data. It is not a certified instrument, a validated digital twin, or
evidence of quantum advantage.

The end-to-end v1 draft implements one bounded simulated network task and a
continuous local execution path. “Implemented” means the software component
exists; it does not mean that predictive accuracy, uncertainty, throughput or
scientific benefit has been established. Those claims are valid only when the
corresponding result is recorded in the end-to-end validation document.

This document defines the claims AQSE may and may not make. Implementation and
validation records must remain separate: a planned test is not a passed test,
and a simulated result is not an experimental measurement.

## Research question

The intended experimental question is:

> Does a representation learned through the supplied VQC and trainable quantum
> kernel improve diagnosis of the simulated sensor–environment system relative
> to declared classical methods receiving comparable information?

This question requires measured comparisons. A valid experiment controls at
least feature information, train/test split, bandwidth, latency, false-alarm
rate, compute budget, and random seeds. A result in which a classical baseline
performs better remains a valid result and must be reported.

## Sensor meaning

- The simulator outputs classical sensor observations.
- The vector network is not a network of entangled sensors.
- The quantum processor does not directly receive a sensor wavefunction.
- A statevector backend represents the ideal software state of the supplied
  circuit, not a physical magnetometer state.
- QNG updates VQC parameters. It does not tune the real or simulated sensor
  hardware unless a future, separately designed control loop is introduced.

The project must not claim improved intrinsic magnetic sensitivity, lower
hardware noise, or smaller physical detection limits because a downstream
kernel was computed. Any improvement in task error or detectable signal must be
measured under matched bandwidth, latency, and false-alarm conditions.

## Physical-model limits

### Local field

The default NED field is synthetic and has no asserted geographic provenance.
Without a configured World Magnetic Model provider, AQSE must report that WMM is
not configured. It must not attach location-specific accuracy to a constant
field.

### Point dipole

The point-dipole expression is quasistatic and valid only outside a declared
near-source exclusion region relative to the physical source size. Its 1/r³
singularity is part of the approximation and may not be hidden by a numerical
clamp. Near-field finite-size behavior requires a different source model.

Magnetic-dipole inversion can be nonlinear, ill-conditioned, and globally
ambiguous. Source position, moment, background, node bias, and pose uncertainty
can trade off. A local full-rank Jacobian does not prove global uniqueness.

### Linear gradient

The symmetric traceless gradient tensor is a local source-free magnetostatic
approximation. It is not globally valid across arbitrary currents, magnetic
materials, or large spatial regions. Spatial sampling rules based on a maximum
cycles-per-metre frequency require an explicitly band-limited field; a nearby
dipole is not rigorously band-limited.

### Smooth anomaly

The Gaussian spatial anomaly is phenomenological. It supports controlled
position and speed experiments but does not claim to model a named pipe, rail,
building, buried object, or geographic magnetic map.

### Periodic validation source

The configurable world-frame sinusoid is a controlled excitation. In particular,
the `quantum_preview_signal` preset adds a 5 Hz, 8 nT vector-amplitude source to
exercise the legacy harmonic-feature and fixed-theta preview path. It is not a
claim that a physical anomaly was detected, not a learned source model, and not
AQSE scientific logic.

## Sensor-model limits

The synthetic tri-axial profile models selected linear calibration effects,
causal bandwidth, bias, thermal drift, stochastic errors, clipping, and
availability. Coefficients are controlled inputs, not experimentally fitted
specifications. Quantization and nonzero transport latency are reserved future
stages, not implemented Milestone 1C effects. The current arrival timestamp is
recorded with zero local transport delay.

Named technology profiles such as NV or optically pumped magnetometers require
their own evidence and contracts. Lock, contrast, optical response, heated-cell
temperature, bandwidth, and operating range appear only where physically
meaningful. AQSE must not generalize one instrument's limits to all devices of a
technology.

Noise magnitudes are meaningful only together with units, sampling rate,
bandwidth, one-sided or two-sided convention, and filter position. A per-sample
standard deviation is not interchangeable with T/√Hz. An approximate flicker
process is labelled 1/f only over a measured and declared band.

## Network and observability limits

One to eight nodes is a prototype constraint. Preset node counts are not
universal minimums for localization or tracking.

- A mono-axial or scalar node does not provide three vector equations.
- A static magnetostatic frame has no direct dependence on source velocity;
  velocity requires sequential observations and a motion model.
- Coplanar, symmetric, or poorly spaced geometry can be rank deficient or badly
  conditioned.
- Unknown source orientation, sensor pose, bias, and background add nuisance
  parameters.
- A remote reference may itself see a local source or fault.
- Common response suggests a common cause but does not prove one.
- One-node experiments cannot create valid network correlation or spatial
  residual features.

Coverage is conditional on a declared reference source, orientation, noise,
bandwidth, observation time, threshold, and boundary treatment. Neither d ≤ 2R
nor a bare spatial Nyquist expression guarantees complete magnetic coverage.

When the information is insufficient, AQSE reports not identifiable or low
confidence instead of manufacturing a location, velocity, event class, or
uncertainty interval.

The end-to-end network classifier uses deliberately conditional labels:

- `NORMAL`;
- `ENVIRONMENT_COMPATIBLE`;
- `DEVICE_COMPATIBLE`;
- `MIXED_OR_AMBIGUOUS`.

They name patterns within the frozen simulator domain, not universal causal
classes. A shared response can be consistent with an environmental change or
with a shared instrumental offset. A nearby physical source can create a large
leave-one-out peer residual. The classifier must retain ambiguous outcomes
rather than resolve observational equivalence using hidden simulator truth.

The routing boundary follows observable context, never the hidden scenario:

- one node uses the local `NORMAL` versus `CHANGE_DETECTED` task and cannot
  infer environment versus device;
- two nodes use the local task with an explicit attribution warning;
- three or more nodes may use the network task only when reference, pose,
  alignment, sample completeness and peer context are compatible;
- insufficient peer context causes a declared local fallback or abstention,
  never synthetic network features.

## Causality and leakage limits

Simulator truth is permitted for labels and evaluation only. Predictive input
contains observations and declared real-world-available calibration or pose
data. It excludes hidden faults, exact injected causes, source truth,
unobserved bias, generator seeds, and future samples.

Online feature extraction and synchronization are causal. Centred filtering or
interpolation using future observations belongs only to an explicitly offline
analysis and is not compared as if it had online latency.

Overlapping windows from one episode must not be randomly divided across train
and test. Dataset partitions vary session seed, trajectory, geometry,
calibration, severity, and event episodes. Sensor ID, scenario name, or an
injected-cause field must not become an unintended shortcut feature.

## State8 feature limits

The non-harmonic `aqse.local-state8.v1` and
`aqse.network-state8.v1` profiles use the observed calibrated world-frame
North component, an observed 8-second session reference, 4-second causal
windows and a 1-second hop at 100 Hz. Their eight coordinates describe window
statistics; they are not sufficient statistics for arbitrary magnetic-source
inversion.

In particular:

- the network peer-median residual is a robust common-field residual under a
  local-comparability assumption, not a reconstructed dipole residual or a
  globally optimal field estimate;
- signed peer correlation retains physically meaningful negative spatial
  responses and requires temporal alignment and comparable axes;
- the Hann-periodogram band ratio depends on the fixed bands, detrending,
  sampling rate and 1e-12 nT² numerical floor; the floor is not a detection
  limit;
- the frozen observed reference is a declared causal calibration, not a
  population scaler and not continuous re-zeroing;
- missing, clipped, stuck, clock-invalid, pose-invalid or mathematically
  undefined inputs can make individual coordinates unavailable and must cause
  abstention when the quantum contract is incomplete;
- changing the reference creates a new version and invalidates pending
  downstream results.

The older harmonic feature profile retains phase and cycle/SNR eligibility and
is a different compatibility identity. Equal vector length never makes these
profiles interchangeable.

## Quantum-model limits

The protected TQK8 implementation defines eight qubits, eight inputs, sixteen
trainable parameters, and a fidelity kernel. The ideal self-kernel is a Gram
matrix over a selected batch. It is not:

- a local fixed-size AFSE representation;
- a sensor wavefunction;
- a probability of a physical event;
- a calibrated classifier output;
- a proof of hardware execution.

Qiskit Statevector and the independent NumPy engine are exact simulators. A
shot-based, noisy-simulation, or QPU mode must have a distinct name, backend
record, budget, and validation. No unavailable backend or credential is
fabricated.

The supplied alignment/QNG implementation is connected only through a bounded,
explicit training wrapper. It is not an inference layer and it never starts
because a measurement arrives. The protected loss is binary: the network
representation trial therefore uses only the declared
`ENVIRONMENT_COMPATIBLE` (-1) versus `DEVICE_COMPATIBLE` (+1) subtask. Numeric
four-class codes are not passed into a binary yyᵀ objective.

The end-to-end study compares only deterministic `theta0` and one candidate
after at most ten accepted protected-QNG updates. This small comparison does
not establish that QNG is globally optimal or that a trained quantum kernel is
superior. The legacy manual kernel preview remains isolated from the live
bundle theta and never invokes QNG.

## AFSE and downstream-model limits

The authorised v1 AFSE is `aqse.afse.nystrom-ridge32.v1`, a standard classical
regularised Nyström map constructed from the protected fidelity kernel. It is
not a new quantum algorithm, a sensor wavefunction, a learned physical state,
or proof that a quantum representation is useful.

Its ordered landmarks come from distinct TRAIN lineages, balanced across the
downstream TRAIN classes where the fixed budget permits. The reference size is
at most 32 and is immutable for a bundle. With W the landmark Gram matrix, the
map stores the symmetric ridge inverse-square-root with λ=1e-6 and evaluates
`z(x)=k(x,L)B`. Adding, deleting or reordering query rows cannot refit or change
that space. A live Gram row against the current batch is not an AFSE substitute.

The optional reconstruction residual and TRAIN-residual p99 OOD threshold are
numerical/engineering diagnostics. They are not calibrated uncertainty or a
certificate that a physical cause is novel.

The downstream model is the fixed compact MLP
`aqse.classical.mlp-32x16-tanh-lbfgs.v1`: TRAIN-standardised AFSE input, hidden
layers 32 and 16, tanh activations and LBFGS under the frozen budget. Its stored
numeric arrays are evaluated by a verified NumPy runtime rather than loading an
executable pickle. A separate model with the same architecture receives the
same raw State8 features as a classical reference.

Outputs are **model scores, not probability-calibrated confidence**. Top score
below 0.70 or top-two margin below 0.15 yields `UNCERTAIN`; those thresholds are
fixed engineering choices, not guarantees of coverage. A decreasing training
loss or a non-uncertain score does not demonstrate generalization, causal
identification, improved localization, physical sensitivity or quantum
advantage.

Any change to profile, TRAIN-fitted scaler, theta, protected TQK identity,
landmarks, B matrix, AFSE dimension or MLP parameters creates an incompatible
space. Old and new vectors/models must never be mixed.

## Continuous-operation limits

Configured sampling and refresh rates are engineering starting points, not
guaranteed real-time performance. The end-to-end draft has a separate
observation-only analysis worker with a 1,600-frame bounded buffer, one
newest-complete-window queue slot, skipped-window accounting, result age,
processing duration and p50/p95 latency fields. These safeguards make overload
visible; they do not create a hard real-time guarantee.

When protected exact-state work or training occupies the shared heavy slot,
measurement continues and analysis reports `PAUSED_FOR_TRAINING`/busy. It may
skip obsolete windows instead of building an unbounded backlog. An old result
must retain its acquisition interval and age; the UI must not label it current
after session epoch, configuration, reference or bundle application changes.

Long-session stability, memory, stream recovery and eight-node throughput are
claims only after the required 600-second real-wall-clock soak is recorded.
Accelerated simulator time and short test fixtures do not substitute for that
measurement. A single local result does not promise hard real-time operation on
other hosts.

## Comparison policy

A credible experiment compares, on the same frozen episodes where applicable:

- a physically motivated classical estimator or baseline;
- a classical classifier or kernel on comparable eight-feature input;
- an untrained quantum kernel;
- a trained TQK without QNG when supported;
- a trained TQK with QNG;
- richer classical raw-data or feature baselines when assessing the value of
  the complete eight-feature compression.

Metrics depend on task and include macro-F1 or multilabel metrics, false alarms
per time, detection delay, field or position error, uncertainty coverage,
throughput, latency, circuit and shot counts, and training duration. Confidence
intervals are calculated across episodes and seeds rather than inferred from a
single run.

The completion criterion is a correct and reproducible comparison, not a
preselected winner.

For the bounded `aqse-network-demo-v1` study, the implemented comparison is
narrower than this general research policy: two theta candidates per local or
network bundle, an AFSE MLP and a same-State8 raw-feature MLP. Candidate
selection uses VALIDATION balanced accuracy, then macro-F1, then theta0 on a
tie. TEST is evaluated once after the selection freeze and cannot select a
winner or trigger retuning. Results may be weak, tied or worse than the
classical reference; all are valid outcomes when reported with class support,
coverage, abstentions and whole-episode intervals.

## Scientific references supplied with the network specification

1. N. Wahlström and F. Gustafsson, “Magnetometer Modeling and Validation for
   Tracking Metallic Targets,” IEEE Transactions on Signal Processing 62(3),
   545–556 (2014), DOI 10.1109/TSP.2013.2274639.
2. I. A. Sulai et al., “Characterizing atomic magnetic gradiometers for fetal
   magnetocardiography,” Review of Scientific Instruments 90, 085003 (2019),
   DOI 10.1063/1.5091007.
3. A. Marasli et al., “Dipole localization using an integrated radio-frequency
   atomic magnetometer,” Physical Review Applied 25, 034047 (2026), DOI
   10.1103/n5bz-416x.
4. T. Hubregtsen et al., “Training Quantum Embedding Kernels on Near-Term
   Quantum Computers,” Physical Review A 106, 042431 (2022), DOI
   10.1103/PhysRevA.106.042431.
5. J. Stokes, J. Izaac, N. Killoran, and G. Carleo, “Quantum Natural Gradient,”
   Quantum 4, 269 (2020), DOI 10.22331/q-2020-05-25-269.
6. D. Donnelly and B. Rust, “The Fast Fourier Transform for Experimentalists,
   Part I: Concepts,” Computing in Science & Engineering (2005), NIST
   publication ID 150008.
7. V. Gerginov et al., “Pulsed operation of a miniature scalar optically pumped
   magnetometer” (2017), NIST document 2882.

These references motivate principles and models. AQSE defaults, profile names,
feature choices, API contracts, and workflow remain project specifications and
are not claims extracted from those publications.
