# Vector Magnetometer and 1–8-Node Network Model

## Scope and compatibility

Milestone 1C adds a vector magnetometer as a separate sensor model. It does not
replace or change the scalar Milestone 1B simulator, endpoint, feature order, or
nanotesla-facing contract.

The continuous 1–8-node simulator, in-memory session API, bounded frame buffer,
and SSE observation stream are implemented in Milestone 1C over the same
physical and sensor boundaries. Relational network features and downstream
inference are still separate pending increments. This document therefore
distinguishes:

- **validated legacy baseline** — finite scalar Milestone 1B;
- **finite-vector Milestone 1C model** — tri-axial simulation, selected-channel
  features, and bounded kernel preview;
- **continuous-network Milestone 1C implementation** — persistent backend time,
  one to eight nodes, declared events and faults, bounded replay, and streaming;
- **candidate continuous-network analysis** — the unimplemented 4 s / 1 s-hop
  relational feature profile and later inference.

No model described here represents a calibrated commercial magnetometer or an
entangled quantum-sensor network.

## Quantities and units

New physical and network computations use SI internally.

| Quantity | Internal unit | Typical UI unit |
| --- | --- | --- |
| Magnetic flux density | tesla, T | µT, nT, or pT |
| Position and baseline | metre, m | m, cm, or mm with explicit conversion |
| Time | second, s | s or ms |
| Sampling rate and bandwidth | hertz, Hz | Hz |
| Magnetic dipole moment | ampere square metre, A·m² | A·m² |
| Field gradient | tesla per metre, T/m | nT/m or µT/m |
| Temperature | kelvin, K | K or °C with an explicit offset conversion |
| Thermal coefficient | tesla per kelvin, T/K | nT/K |
| Bias random-walk intensity | square tesla per second, T²/s | nT²/s |
| Noise amplitude density, when used | tesla per square-root hertz, T/√Hz | fT/√Hz or pT/√Hz |

Unit boundaries are explicit: the network API and backend remain in SI, while
the GUI converts magnetic-field values to the displayed nT, µT, or pT units.
In particular:

    1 µT = 10⁻⁶ T
    1 nT = 10⁻⁹ T
    1 pT = 10⁻¹² T
    1 G  = 10⁻⁴ T = 100,000 nT

The approved synthetic finite-vector reference field is

    B0_world = [20,000, 0, 45,000] nT
             = [20, 0, 45] µT
             = [20e-6, 0, 45e-6] T.

It is a generic local field for the demonstrator, not a value inferred from a
geographic location and not a World Magnetic Model result.

## Coordinate frames

### World frame

The local world frame is right-handed North-East-Down, abbreviated NED:

- +X_world points North;
- +Y_world points East;
- +Z_world points Down;
- position is r_world = [north, east, down] in metres relative to a declared
  local origin.

NED is right-handed because North × East = Down.

### Sensor frame

The sensor or body frame is right-handed:

- +X_sensor is the sensor forward axis;
- +Y_sensor is the sensor right axis;
- +Z_sensor is the sensor down axis in the aligned reference pose.

At identity orientation the world and sensor axes are aligned.

### Rotation and quaternion contract

The direction-cosine matrix R_world_to_sensor maps components expressed in NED
to components expressed on the sensor axes:

    B_sensor = R_world_to_sensor · B_world.

Quaternions are serialized scalar-first:

    q_world_to_sensor = [w, x, y, z].

They represent the same world-to-sensor transform, use right-hand-positive
rotations, and must have unit norm. For deterministic serialization, equivalent
quaternions q and −q are canonicalized to a non-negative scalar component when
that does not introduce a discontinuity in an active trajectory.

The matrix invariants are:

    RᵀR = I
    det(R) = +1.

Implementation tests must state concrete vector mappings for identity and
known 90-degree rotations. Merely comparing two matrices created by the same
helper is not an independent convention test.

Euler angles may be offered as UI inputs, but simulation state and interpolation
use quaternions or rotation matrices. High-dynamic motion must not depend on an
Euler representation that can encounter gimbal lock.

## Environment field

The environment owns world-frame magnetic truth:

    B_environment_world(r,t)
      = B_earth_world(r,t)
      + B_common_world(t)
      + G_world(t) · (r − r0)
      + Σ B_dipole,j_world(r,t)
      + Σ B_local,j_world(r,t)
      + Σ B_periodic,j_world(t).

The output named environment_field_world is the total above. When component
inspection is required, disturbance fields are carried as separate truth
components rather than inferred by subtracting undocumented values.

One environmental process is generated once and evaluated for every node. It
must not be sampled independently per sensor. A world-fixed anomaly rotates
into each sensor frame; a body-fixed device offset does not.

### Field providers

The environment boundary exposes:

1. the configured ConstantLocalFieldProvider;
2. the configured SyntheticSpatialFieldProvider for gradients, point dipoles,
   Gaussian anomalies, and periodic sources;
3. a reserved WorldMagneticModelProvider boundary that is unavailable and not
   configured in Milestone 1C.

The provider catalog reports WMM as unavailable and not configured. No WMM
coefficients, epoch, or geodetic-location accuracy are bundled. A constant field
must not be labelled as WMM output.

### Explicit periodic validation source

Each enabled periodic source contributes a world-frame NED vector

    B_periodic,j_world(t)
      = a_j_world sin(2π f_j t + phi_j),

where a_j_world is a three-component amplitude in tesla, f_j is positive and
strictly below the acquisition Nyquist frequency, and phi_j is in radians. The
source is evaluated once as environmental truth and shared spatially by all
nodes; each node then observes it through its own pose and sensor transfer.

The `quantum_preview_signal` preset uses four nodes and adds an explicit 5 Hz
source with vector-amplitude magnitude 8 nT, aligned with the default local
field. It exists only to exercise the legacy-compatible harmonic feature and
fixed-theta preview path. It is infrastructure validation input, not an anomaly
model, detected event, AQSE scientific algorithm, or scientific-performance
claim. The default network preset does not enable this source.

## Source-free linear gradient

For a declared local magnetostatic region containing no source or current, the
linear gradient tensor obeys both divergence-free and curl-free constraints:

    G = Gᵀ
    trace(G) = 0.

A convenient five-parameter representation is:

    G = [[ gxx,  gxy,          gxz       ],
         [ gxy,  gyy,          gyz       ],
         [ gxz,  gyz,  −gxx − gyy       ]].

This representation prevents the UI or API from creating three unrelated axis
gradients that contradict the stated source-free model. If a region contains a
declared current or magnetic source, a different model and assumptions must be
used rather than silently relaxing the invariant.

## Point magnetic dipole

For sensor position r and source position r_source, define

    rho = r − r_source
    rho_hat = rho / ||rho||.

The quasistatic point-dipole model in SI units is

    B_dipole(r,t)
      = µ0 / (4π ||rho||³)
        · [3 rho_hat (m(t) · rho_hat) − m(t)],

where m is in A·m² and the result is in tesla.

### Domain and singularity policy

The point-dipole approximation is valid only when distance is sufficiently
larger than the characteristic source dimensions and retardation is negligible.
Every source therefore declares a model-validity distance. A sample at or below
that distance is invalid for this model and is reported as such.

AQSE must not hide the singularity by clamping ||rho|| to an arbitrary epsilon.
If traversal through the source volume is required, a finite-size source model
must be selected explicitly. Validation checks the expected 1/r³ decay along a
fixed direction without claiming that the dipole is globally identifiable.

## Smooth phenomenological anomaly

The approved finite-vector local-anomaly model is a Gaussian world-frame field:

    B_anomaly_world(r)
      = A · d_hat · exp(−||r − r0||² / (2 L²)),

where A is the peak field in tesla, d_hat is a unit world-frame direction, and
L is the spatial scale in metres. The model is smooth and useful for controlled
crossing tests, but it is phenomenological: it is not a geographic map, WMM
component, or Maxwell-complete model of a specific object.

For a straight trajectory, temporal width varies with L and speed. Peak response
also depends on closest approach. The simulator must calculate this dependence
from position instead of injecting a time pulse independently at each node.

## Sensor transfer model

For a tri-axial sensor, use column vectors and the approved ordered model:

    u_i(t) = R_world_to_sensor,i(t) · B_environment_world(r_i(t),t)

    v_i(t) = C_axis,i · G_gain,i · A_soft,i · u_i(t)

    p_i(t) = v_i(t)
             + b_hard,i
             + b_temperature,i(t)
             + b_deterministic_drift,i(t)
             + b_random_walk,i(t)
             + b_correlated,i(t)
             + b_instrument_event,i(t)
             + eta_white,i(t)

    h_i(t) = H_bandwidth,i {p_i(t)}

    measured_i(t) = availability_i(
        saturate_i(project_mode_i(h_i(t))))

The right-most matrix acts first. Definitions are:

- A_soft is the soft-iron deformation;
- G_gain is diagonal and contains positive per-axis gain factors;
- C_axis models controlled cross-axis coupling or non-orthogonality;
- b_hard is a body-fixed, output-equivalent additive offset in sensor axes;
- b_temperature = k_T (T_device − T_reference);
- b_deterministic_drift is the configured body-frame drift rate multiplied by
  simulated time;
- b_random_walk is a persistent device bias process;
- b_correlated is the persistent device-side correlated process;
- b_instrument_event represents declared node or shared-instrument offsets and
  drift applied only to their targeted nodes.

For the approved finite-vector profile, A_soft is symmetric positive definite
and its condition number must not exceed 100. Gain and cross-axis matrices must
remain finite and nonsingular. These are demonstrator safety constraints, not
universal calibration limits.

Hard-iron is added after the calibration matrices and is therefore defined in
output-equivalent sensor-frame field units. A future model that treats it as a
physical field before gain must use a different versioned transfer contract.

### Measurement modes

Every node declares one mode:

- tri-axial vector: returns the three measured sensor-frame components;
- mono-axial: returns the projection on one declared sensitive unit axis;
- scalar total field: returns a scalar magnitude according to its declared
  sensor profile.

A scalar reading is never expanded into three invented components. Three
reported channels are not assumed to provide three independent constraints for
every geometry.

## Finite bandwidth

Bandwidth is a causal hardware-response stage and remains separate from
data-cleaning preprocessing. It is disabled by default in the approved finite
vector profile.

When enabled, each axis uses a first-order causal response with

    tau = 1 / (2π f_c)
    a = exp(−dt / tau)
    h[0] = v[0]
    h[k] = a h[k−1] + (1−a) v[k].

The parameter f_c is the continuous-time-equivalent one-pole cutoff. It must be
positive and below Nyquist. Near Nyquist the discrete response does not have an
exact digital −3 dB point at the entered frequency; profiles requiring that
property need a separately versioned, prewarped digital filter.

White readout noise is added to the deterministic sensor response and biases
before this bandwidth stage. Consequently, enabling the hardware response
colors and reduces the statistics of the configured per-sample white-noise
input. Measurement-mode projection and saturation follow the filter, with
saturation always applied last to the pre-clipping projected value.

## Saturation and availability

Vector saturation accepts optional explicit per-axis limits. For backward
compatibility, `saturation_limit_T` remains the scalar fallback applied to all
three axes when `saturation_limits_T` is absent. Generic examples include:

- ±4 G = ±400,000 nT;
- ±8 G = ±800,000 nT.

They are demonstration ranges, not specifications of every magnetometer. Each
saturation-mask component is evaluated on its corresponding pre-clipped value
using a strict exceedance test. Exactly reaching the boundary is not counted as
clipped. The response exposes the per-sample mask; windowed feature quality
reports the saturation fraction for the selected channel.

Availability then decides whether the projected, clipped observation is emitted.
Absence is serialized as null plus quality flags, never as a valid zero or JSON
NaN. Quantization is a reserved future transfer stage: Milestone 1C has no step,
bit-depth configuration, or quantization operation.

Every reading carries acquisition and arrival timestamps, but the current local
transport latency is exactly zero and no network-delay distribution is
simulated. A nonzero transport model requires a separate versioned contract and
must not be inferred merely from the presence of `arrival_time`.

## Persistent stochastic processes

Randomness is generated by the authoritative backend process, not during UI
rendering.

### White readout noise

For the approved finite-vector contract,

    eta_white,i[k] ~ Normal(0, diag(sigma_x², sigma_y², sigma_z²)),

where each sigma is an RMS standard deviation in nT per sample at the GUI
boundary and in T per sample in both the network API and backend. Axes are
independent in this version. The configuration field is named
white_noise_std_T_per_sample in the API and backend. This
parameter is not a spectral amplitude density. A future input expressed in
T/√Hz requires a separately named field plus an explicit equivalent-noise-bandwidth
and one-sided or two-sided conversion contract.

### Bias random walk

For each axis,

    b[k+1] = b[k] + sqrt(q dt) xi[k],
    xi[k] ~ Normal(0,1),

with q expressed internally in T²/s and displayed as nT²/s where appropriate.
The process starts at zero unless an explicit initial bias is configured. Its
variance grows proportionally to elapsed simulated time; it is not regenerated
as independent white noise per batch.

### Correlated process

A stationary Ornstein–Uhlenbeck component may be represented by

    a = exp(−dt/tau)
    u[k+1] = a u[k] + sigma_stationary sqrt(1−a²) xi[k].

The state is preserved across output batches, UI refreshes, pause, and resume.
Its time constant and stationary standard deviation are explicit.

### Flicker approximation

A flicker approximation may use a documented sum of correlated processes with
separated time constants. It can be labelled approximately 1/f only over a
declared frequency band and only after its PSD has been checked there. An
arbitrary slowly changing random signal is not called flicker noise.

### Independent random streams

One session seed derives stable named substreams indexed by physical source,
node, and stochastic component. A recommended derivation uses a SeedSequence
formed from the session seed and a stable component identifier. Adding a fault
to S3 must not change the environmental realization or S1 white-noise samples.

The same seed, initial configuration, command log, and simulated-time schedule
must reproduce samples. Generate New Realization creates, displays, and records
a different seed; it never randomizes silently.

## Temperature

Ambient and device temperature are distinct states. Device temperature follows
a causal first-order thermal response to the configured ambient target. The
legacy `ambient_temperature_K` value is a constant target when
`temperature_driver` is absent. An optional node-level driver supports:

- `constant`, which retains that target;
- `ramp`, a bounded linear change for `ramp_duration_s` followed by a hold;
- `sinusoidal`, with explicit amplitude, frequency, and phase.

Ramp and sinusoidal targets are validated to remain within the demonstrator
temperature bounds, and sinusoidal frequency must remain below the session
Nyquist frequency. The additive thermal bias is

    b_temperature(t) = k_T · (T_device(t) − T_reference),

where k_T is a three-axis coefficient. Thermal gain variation is a separate
future option and must not be implied by this additive model.

A heated-cell setpoint, lock state, optical contrast, or similar telemetry is
exposed only by a sensor-technology profile for which it has declared meaning.

## Pose and trajectory scenarios

### Static reference

Position and orientation are fixed. This is the reference for noise, drift,
temperature, and deterministic transform tests.

### Calibration pose sweep

The approved finite-vector calibration dataset uses deterministic seeded
Haar-uniform rotations on SO(3). It is a pose cloud, not a claim of continuous
physical motion. Under a constant ideal field, measured directions form a
sphere; hard-iron translates it; the approved linear matrix deforms it into an
ellipsoid.

### High-dynamic motion

This is a separate continuous scenario with bounded angular velocity and a
quaternion trajectory. It does not reuse independently sampled calibration
poses as adjacent time samples.

### Translation and anomaly crossing

The implemented position models are static and constant-velocity linear motion;
the latter also drives the local-anomaly crossing and combined-stress presets.
Spatial fields are evaluated at each actual position. Circular, user-defined,
and stochastic trajectories are future capabilities and are not selectable in
Milestone 1C.

## Network presets and observability

AQSE supports one to eight total nodes. Reference nodes count toward that limit
and are not immune to local sources or device errors.

| Preset | Typical nodes | Purpose |
| --- | ---: | --- |
| Single-sensor control | 1 | Local signal, quality, and ambiguity baseline |
| Basic gradiometry | 2, extensible to 8 | Calibrated difference and directional-gradient approximation |
| 2D localization | 3–4 | Declared-height inverse problem with known and unknown parameters listed |
| 3D tracking and moment | 5–8 | Non-coplanar geometry and sequential state estimation |
| Area monitoring | 1–8 | Conditional SNR or detection map and uncertainty |
| Quantum preview harmonic signal | 4 | Explicit 5 Hz validation source for the legacy feature/preview path |
| Eight-node causal event demo | 8 | Reproducible moving-source and composable truth-labelled fault timeline |
| Single-sensor ambiguity control | 1 | Separate equal-amplitude world-field and device-bias phases with similar selected-axis responses |

These counts are experiment presets, not universal mathematical minima.
Instantaneous magnetostatics do not directly measure source velocity. Local
identifiability is assessed with a noise-weighted, nondimensionalized Jacobian,
its numerical rank, and singular values. Velocity requires a temporal window
and motion model. Sufficient local rank does not guarantee a unique global
solution.

Coverage plots must identify the reference source, orientation assumption,
bandwidth, observation time, noise, and detection criterion. AQSE does not use
d ≤ 2R as a general coverage guarantee and does not infer a required node count
from a spatial Nyquist rule unless the stated band-limited assumptions hold.

## Events and faults

Events are composable and may overlap. The truth taxonomy distinguishes:

- world-common field variation;
- mobile dipole affecting only some nodes;
- external local magnetic disturbance;
- single-node drift or noise increase;
- shared instrumental disturbance across a declared group;
- saturation;
- signal loss;
- frozen output;
- clock offset or synchronization error;
- technology-specific lock loss and recovery;
- simultaneous physical event and device fault.

External magnetic events alter B_environment_world. Implemented node bias,
drift, noise-burst, shared-instrument-offset, dropout, and stuck events alter
only targeted device observations and leave world truth unchanged. Clock
parameters exist on node configuration; nonzero transport/communication fault
events and technology-specific lock/recovery events remain future extensions.

Event rates use a continuous-time convention:

    p_event_over_dt = 1 − exp(−lambda dt).

Amplitude, duration, onset, recovery, affected nodes, frame, and effective time
are explicit. Labels are multilabel or otherwise composable; local/common and
physical/instrumental are separate axes. Correlation is evidence, not proof of
cause.

Current per-node observed quality uses composable `clipped`, `signal_absent`,
`stuck`, and `clock_error` flags. Out-of-range distinctions beyond clipping,
technology-specific lock/recovery, and communication-delay flags require future
contracts. If the UI holds a last value, it must display its age and exclude it
from new-sample analysis.

Every scheduled event is assigned an effective frame and configuration version.
Frames before that boundary retain historical versioning; later observation and
truth frames carry the effective version. Synchronous same-realization replay is
bounded to 5,000 generated frames. Larger histories must be reset and replayed
incrementally rather than synchronously rebuilt in one request.

## Typed vector acquisition contract

A finite vector acquisition keeps synchronized arrays:

| Field | Shape | Frame or meaning |
| --- | --- | --- |
| time | N | simulated seconds |
| position | N×3 | world NED, m |
| orientation | N×4 | q_world_to_sensor, wxyz |
| temperature | N | observed device K |
| earth_field_world | N×3 | Earth/reference contribution, T |
| environment_field_world | N×3 | total world field, T |
| ideal_field_sensor | N×3 | after rotation, before sensor effects, T |
| measured_field | N×3 | sensor-frame observable, T |
| saturation_mask | N×3 | boolean pre-clipping exceedance |

Truth arrays and measured arrays are exposed through different contracts or
channels. Feature extraction accepts only the measured view.

The continuous network adds session, frame, node, sequence, acquisition-time,
arrival-time, measurement-mode, configuration-version, calibration-version, and
quality provenance. ObservationFrame and TruthFrame are separate versioned
schemas. PredictionFrame remains a downstream architectural contract rather
than a current network-simulator output.

## Scientific limitations

- The synthetic field and sensor models are controlled research models, not a
  validated digital twin.
- A point dipole is not valid inside or near an extended source.
- A linear symmetric traceless gradient is local and source-free by assumption.
- Magnetometer geometry alone does not guarantee global source identifiability.
- Vector components depend on pose, calibration, and frame convention.
- A remote reference can be contaminated.
- Noise parameters are meaningful only with their sample rate, filter order,
  bandwidth convention, and units.
- The explicit periodic source and `quantum_preview_signal` preset are controlled
  validation excitations, not evidence of detection or model performance.
- Quantization and nonzero transport latency are not implemented.
- WMM is unavailable and not configured; the default field has no geographic
  accuracy claim.
- The network is classical; it does not model entangled sensors.
- No simulated result demonstrates improved intrinsic sensor sensitivity.
