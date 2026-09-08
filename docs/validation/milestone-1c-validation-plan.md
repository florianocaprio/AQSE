# Milestone 1C Scientific Validation Plan

## Status

This is a prospective validation specification. It contains required checks and
acceptance evidence; it is not a test-execution record. No test is claimed to
have run or passed by the creation of this document.

Actual runs must record date, commit, branch, command, host architecture,
container image, random seed, duration, result, warnings, and generated artifact
identifiers. Failed or skipped checks remain visible.

## Protected baseline

Before and after Milestone 1C work, verify that:

- scalar Milestone 1B endpoints and feature semantics remain compatible;
- all original TQK8 scientific tests remain unchanged and execute;
- README_TQK8.md, the original notebook, and validation/tqk8 artifacts are not
  rewritten;
- no physical QPU, IBM credential, or network service is required;
- new quantum paths delegate to the supplied implementation.

A source diff against the baseline commit is required for protected scientific
files. Passing new tests does not authorize an algorithmic change to them.

## Validation layers

Run validation in increasing scope:

1. deterministic unit tests for mathematics and contracts;
2. API schema and error-semantics tests;
3. cross-module finite-vector tests;
4. bounded quantum-preview tests;
5. frontend production build and browser QA;
6. Docker health and endpoint verification;
7. continuous-network simulator/session/API smoke tests;
8. separately declared performance and soak tests.

Statistical tests use fixed seed sets, declared confidence criteria, and enough
replications to test the stated distribution. They must not assert one exact
random sample statistic.

## Coordinate-frame tests

| Check | Required assertion |
| --- | --- |
| Identity | R_world_to_sensor maps NED [N,E,D] to sensor [X,Y,Z] unchanged |
| Known 90° mappings | Independently specified vectors map to expected signed axes |
| Quaternion norm | Every generated q_world_to_sensor has norm one within declared tolerance |
| Matrix orthogonality | max absolute element of RᵀR−I is below tolerance |
| Handedness | det(R) is +1 within tolerance |
| Round trip | Rᵀ(Rv) recovers v |
| Quaternion sign | q and −q yield the same R; canonical serialization is deterministic |
| Pose shape | N orientations serialize as N×4 in wxyz order |

Test expected vectors must not be produced by the same conversion function under
test. Tolerances are recorded with the implementation and data type.

## Environment-field tests

### Constant and gradient fields

- The approved [20,000, 0, 45,000] nT reference converts exactly to the declared
  SI representation within floating-point tolerance.
- Identical static positions see identical constant world truth.
- The five-parameter gradient representation is symmetric and traceless.
- A configured displacement produces G(r−r0) with correct T/m units and sign.
- A deliberately invalid nonsymmetric or nonzero-trace source-free gradient is
  rejected.

### Dipole

- Compare axis and equatorial cases against an independently calculated formula.
- Rotate moment and displacement to verify vector direction and sign.
- At fixed direction, doubling distance changes field magnitude by a factor of
  eight within tolerance.
- At or inside the declared model-validity distance, return a validation state
  rather than a clamped finite field.
- Moving-source position is evaluated at simulated sample time, not UI refresh
  time.
- SI input moment in A·m² and output field in T survive API round trips.

### Gaussian spatial anomaly

- The field reaches configured amplitude and direction at its centre.
- Equal-distance points have equal envelope values.
- A faster straight crossing produces a narrower temporal event for the same
  spatial scale.
- Different nodes receive the same world function evaluated at their own
  positions; no node-local random pulse is substituted.

### Periodic validation source and field-provider status

- Independently evaluate a configured source as
  a_world sin(2π f t + phi) and compare all three truth components.
- Reject zero, negative, or at/above-Nyquist frequency in both finite-vector and
  network configurations.
- Confirm that the same world-frame periodic value is evaluated for every node
  at a common simulated time before each node's pose and transfer response.
- Confirm that the default network contains no periodic source.
- Confirm that `quantum_preview_signal` contains a 5 Hz source whose vector
  amplitude has 8 nT magnitude and is aligned with the default field.
- Exercise network observations through the legacy-compatible feature extractor
  and bounded preview using this preset; require at least two valid windows for
  a non-degenerate preview batch. Record this only as infrastructure-path
  validation, never as AQSE detection or scientific-performance evidence.
- Confirm the provider catalog reports constant and synthetic providers as
  available/configured and WMM as unavailable/not configured, with no bundled
  coefficients or location-accuracy claim.

## Sensor-transfer tests

| Effect | Required assertion |
| --- | --- |
| Ideal sensor | Identity pose and matrices reproduce environment field in sensor axes |
| Pure rotation | Field magnitude is preserved |
| Hard iron | Adds the approved body-fixed output-equivalent vector after the matrices |
| Gain | Diagonal G scales only its declared axes |
| Soft iron | SPD validation passes valid matrices and rejects non-SPD or condition number above 100 |
| Cross-axis | C_axis produces the configured coupling in the approved multiplication order |
| Calibration sweep | Ideal constant-field cloud is spherical; hard iron translates; linear deformation produces an ellipsoid |
| Temperature | Additive k_T(T_device−T_reference) has correct axis, sign, and unit |
| Shapes | All synchronized vector arrays have N×3; quaternions have N×4; masks have N×3 |
| Mode | Tri-axial, mono-axial, and scalar outputs expose only actually measured values |

Include a non-commuting matrix case so a regression in the approved order
C_axis · A_soft · G · R cannot pass accidentally.

## Bandwidth and sampling tests

For the approved one-pole response:

- null or disabled bandwidth is an exact bypass;
- invalid cutoff at zero, negative, or at/above Nyquist is rejected;
- a constant input has no artificial startup transient because h[0]=v[0];
- a step follows the expected causal time constant;
- sinusoidal attenuation and phase lag agree with the implemented discrete
  transfer function;
- the filter state is identical for one monolithic batch and the same samples
  split across multiple batches;
- changing UI refresh does not change samples or filter output;
- no future sample influences current output.

Sampling-rate, profile-bandwidth, event-frequency, and antialiasing checks must
produce comprehensible validation errors or quality warnings.

## Persistent-noise tests

### Reproducibility and independence

- Same session seed, configuration, command log, and simulated times reproduce
  every observation.
- Pause and resume do not reset random walk, OU, thermal, or filter state.
- Chunked and monolithic generation produce identical samples.
- Generating or enabling an S3 fault does not alter environment or S1 streams.
- A new explicitly generated seed changes stochastic realizations and is stored.

### Distribution and time scaling

- White-noise sample mean and variance match configured per-axis RMS within a
  predeclared statistical acceptance interval across fixed seeds.
- White noise is added after the bandwidth filter in the approved model.
- Random-walk increment variance scales with q·dt.
- Ensemble random-walk variance scales with elapsed simulated time.
- Deterministic drift scales with dt rather than sqrt(dt).
- OU stationary variance and lag correlation agree with sigma and
  exp(−dt/tau) within statistical tolerance.
- An approximate flicker process is checked only across its declared frequency
  band and is not labelled 1/f outside it.

If amplitude density in T/√Hz is introduced, test the explicit conversion using
the declared equivalent-noise bandwidth and one-sided or two-sided convention.

## Saturation and availability tests

- Values below the range remain unchanged.
- Values exactly at a range boundary are not flagged clipped.
- Exceeding values clip per axis and set the correct N×3 mask.
- Per-axis counts, total count, and percentage over 3N agree.
- Generic ±4 G and ±8 G convert to ±400,000 and ±800,000 nT.
- Dropout returns null plus flags, never a valid zero, NaN, or Infinity.
- Stuck output repeats the last observable value with a `stuck` flag and is not
  marked as a new valid sample. Explicit stale-age metadata remains future work.
- Lock-loss checks are excluded unless a future technology profile defines lock
  telemetry and recovery semantics.
- A simulated sensor failure remains a data event and does not cause HTTP 500.

Quantization has no Milestone 1C configuration or implementation and is excluded
from current sign-off. If later introduced, its order, step/bit model, units, and
rounding behavior require a separately versioned test section.

## Events and causal separation tests

Exercise the implemented cases below, keeping future cases explicitly marked:

1. common world-field variation;
2. a moving dipole visible mainly near selected nodes;
3. a local external magnetic disturbance;
4. drift on S3 only;
5. increased noise on one node;
6. a shared instrumental disturbance on S1 and S2;
7. clipping, signal loss, frozen data, and clock error;
8. simultaneous physical event and device fault;
9. future extension: recovery with an explicitly modelled transient.

For each implemented case assert whether world truth or the targeted sensor
transfer changes. Cause labels are composable. A common instrumental fault does
not alter world truth, and a local physical field is not automatically labelled
a device fault. The current arrival timestamp has zero transport delay; nonzero
transport and communication-fault behavior are future tests, not current
acceptance claims.

Random dropout and stuck occurrence over dt follows 1−exp(−lambda dt); scheduled
events use explicit start and duration instead. Changing UI frame rate must not
change either sequence.

## Feature and quality tests

### Legacy contract

- Preserve exact names, order, units, and spectral_peak meaning.
- X, Y, Z, and |B| selection use only the selected measured channel.
- |B| is calculated from measured components rather than world truth.
- Finite-vector windows use 1 s duration and 50% overlap.
- Complete-window count, hop, centre timestamp, and discarded remainder match
  the profile.
- Temperature is the mean observed device temperature inside the window.

### Approved finite-vector quality policy

- A series or requested window with fewer than 16 finite samples is rejected
  before feature output.
- Fewer than 2 estimated cycles invalidates harmonic interpretation.
- Peak prominence below 6 dB is flagged.
- Estimated SNR below 0 dB is flagged.
- Any clipped sample makes the initial profile ineligible for preview.
- The sample bound and eligibility settings are schema-fixed at 16 samples,
  2 cycles, 6 dB prominence, 0 dB SNR, and zero clipping; attempts to tune the
  quality policy are rejected.
- Constant and almost-constant complete finite windows return finite
  serialization plus explicit per-feature validity.
- A numerical fallback is never reported as a physically meaningful zero.

Missing/null, frozen, transient-dominated, and broadband-dominated window
classification are pending quality-policy extensions. Current tests must verify
that malformed, non-finite, short, or non-uniform input is rejected, without
claiming those future reason codes are emitted.

### Feature provenance

- Every extracted window has a nonempty required `feature-window-v1` HMAC token.
- The token covers the complete profile and all record fields except itself.
- Modifying features, quality, identifiers, indices, timestamps, or profile
  causes preview validation to reject the record before circuit construction.
- An unmodified record produced by the same backend process verifies.
- A backend restart invalidates prior ephemeral tokens and requires extraction
  to be repeated.
- Test reports describe this as in-process record integrity, not sensor
  authentication, raw-acquisition attestation, or durable archival provenance.

### Continuous-network profile boundary

This subsection validates a candidate contract only. The 4 s / 1 s-hop
relational extractor is not implemented and is excluded from current executable
feature sign-off.

- A 4 s / 1 s-hop profile has a distinct identifier from the finite-vector
  profile.
- A one-node batch cannot manufacture a leave-one-sensor-out residual or network
  coherence value.
- Missing relational context follows its declared mask and degraded path.
- Feature rows with the same length but different profile IDs are incompatible.

### Anti-leakage

- Changing or hiding TruthFrame while keeping observations fixed leaves features
  identical.
- No feature input contains source truth, hidden fault cause, seed, true bias,
  or future samples.
- Online preprocessing is causal.
- Scaler and imputation fit only on permitted reference or training episodes.
- Overlapping windows from one episode remain within one dataset partition.

## Quantum-preview tests

Preserve and run all original TQK8 tests, then add integration tests proving:

- preview delegates to the existing AngleScaler and TQK8 engine;
- raw X and encoded X both have shape M×8;
- exact feature order is preserved;
- the declared reference dataset determines mean and scale;
- self-reference is marked exploratory;
- query data do not refit an existing scaler snapshot;
- theta has exactly 16 finite values and is bitwise or numerically unchanged by
  preview;
- changing a draft control does not execute a circuit;
- the response echoes the executed theta snapshot;
- the workbench default uses 16 windows and a combined reference/query total
  above 128 is rejected;
- incompatible and quality-invalid windows are rejected with identifiers;
- K has shape M×M and is symmetric within tolerance;
- each diagonal entry is approximately one;
- all entries are in [0,1] within numerical tolerance;
- the smallest eigenvalue is not below the declared negative numerical
  tolerance;
- Qiskit and independent NumPy engines agree on a small deterministic batch;
- changing theta changes a suitable non-degenerate kernel case;
- no QNG function is invoked;
- no network connection, IBM credential, or physical QPU is used;
- reported kernel statistics and duration are calculated from the returned run;
- the kernel is labelled as a Gram matrix, not AFSE.

The test record must also state that sensor-integrated QNG training is not
connected, AFSE mathematics is pending, the final neural model is not
implemented, and no physical QPU path exists. These are capability boundaries,
not skipped preview computations.

## API tests

- Preserve scalar routes and response semantics.
- Reject non-finite, malformed, oversized, wrong-frame, wrong-unit, and
  incompatible-profile payloads.
- Preserve separate ObservationFrame and TruthFrame schemas. Any future
  PredictionFrame must remain a separate downstream contract.
- Enforce session and configuration versions at effective-time boundaries.
- Scheduling an event reports its effective frame and new configuration version;
  every later observation/truth frame carries that version, while earlier frames
  retain their historical version.
- Synchronous replay accepts at most 5,000 generated frames, rejects larger
  sessions clearly, and preserves deterministic observations/truth within the
  supported bound.
- Return sensor faults as observations with quality, not transport failures.
- Distinguish backend readiness, simulation lifecycle, worker health, and job
  failure.
- Do not return stack traces, environment secrets, or hidden truth through
  prediction endpoints.
- Validate cancellation and idempotency when jobs or sessions are implemented.

## Frontend QA

The production build is necessary but not sufficient. Browser QA verifies:

- all UI copy is English;
- Overview reflects real shared state;
- vector frame diagram and X/Y/Z colors are consistent;
- numeric and keyboard alternatives exist for geometry controls;
- draft and executed values are visually distinct;
- stale simulation, features, scaler, and kernel results are marked;
- truth, observations, and predictions remain distinct and blind mode changes
  visibility only;
- backend unavailable, simulation stopped, application error, and sensor fault
  have different states;
- Run Quantum Preview is the only preview trigger;
- editing theta does not invoke the backend;
- real heatmap values and lineage are inspectable;
- AFSE states mathematical implementation pending and shows no fabricated
  embedding;
- unavailable estimators display not available instead of invented numbers;
- mobile, keyboard, and wide-screen layouts remain readable.

## Network acceptance matrix

For the implemented continuous simulator/session/API layer, run deterministic
smoke scenarios for 1, 2, 4, and 8 nodes. Validate:

- one authoritative backend world per session;
- sample and stream ordering;
- correct reference-node accounting;
- bounded frame buffers and explicit overwrite/gap reporting;
- reconnect and gap recovery;
- environment sharing and independent device streams;
- geometry/observability warnings;
- no fabricated network correlation in the one-node control;
- simulation progress under concurrent control and read requests;
- simulation-lag reporting.

Queue depth, skipped analysis windows, feature/prediction data age, and analysis
compute-duration reporting belong to the pending continuous-analysis worker and
must not be marked as current network acceptance criteria.

The saved `eight_node_event_demo` preset must reproduce its declared timeline:

1. a moving dipole represented only in world truth;
2. a common world-field offset;
3. increasing S3 drift without changing world truth;
4. a shared instrumental offset on S1/S2;
5. an S4 dropout interval with return when the interval ends;
6. simultaneous world-field offset and S4 dropout.

The saved `single_sensor_ambiguity_demo` applies equal selected-axis offsets in
two separate phases: a world-field cause from 1–3 s and a device-bias cause from
4–6 s. It demonstrates that similar readings from one sensor do not identify the
cause; it does not run a diagnostic model or retrain on test observations.

## Performance and soak validation

Preview, inference, training, and continuous simulation have separate budgets.
Record rather than assume:

- host CPU and memory;
- architecture and container versions;
- node and sample rate;
- UI refresh and analysis cadence;
- feature throughput;
- preview M, backend, state count, and duration;
- training batch, steps, metric mode, circuit budget, and duration;
- queue depth, skipped windows, and data age;
- peak and steady-state memory.

A quick smoke test and a separate eight-node run of at least 20 minutes are
required for continuous-network sign-off. The long run must use a dedicated
command and record actual elapsed simulated and wall time. It must never be
listed as passed if interrupted or not executed.

## Validation record template

For every executed command, append a result record containing:

    command
    start and finish time
    commit and dirty-worktree status
    Docker image IDs and architecture
    host summary
    selected seed and scenario
    result: passed | failed | skipped
    exact counts and measured timings
    warnings and known limitations
    artifact paths or identifiers

Final sign-off reports every failure and skipped check. It does not summarize a
partially run suite as all tests passed.
