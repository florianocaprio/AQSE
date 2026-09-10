# AQSE Feature Profiles, Windowing, and Quality

## Purpose

AQSE always presents eight ordered numeric coordinates to the protected TQK8
boundary. Eight columns alone do not imply semantic compatibility: a vector is
usable only with the exact versioned feature profile, feature order, units,
window policy, reference policy, scaler, encoding policy and fitted bundle that
produced it.

The end-to-end demonstrator supports three deliberately separate feature
families:

- the preserved legacy harmonic scalar/vector-compatible profile;
- `aqse.local-state8.v1` for causal single-node change detection;
- `aqse.network-state8.v1` for peer-conditioned pattern classification.

The State8 profiles are implemented processing contracts. They are not new
quantum algorithms, and they do not change the author's VQC, TQK, loss or QNG
implementation.

## Preserved legacy harmonic profile

The validated Milestone 1B order remains immutable:

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

`spectral_peak` is a PSD magnitude, not the peak frequency. Frequency remains
f2. The scalar/vector-compatible extractor selects one declared measured
channel (`X`, `Y`, `Z`, or measured magnitude); it does not average axes,
concatenate 24 coordinates, or substitute simulator truth. Its harmonic quality
and `phase-direct.v1` encoding contracts remain confined to that legacy family.

The State8 path must never reuse the harmonic-cycle/SNR eligibility test or the
phase-specific encoding merely because both paths contain eight values.

## Implemented State8 acquisition contract

Both State8 profiles consume calibrated, observed tri-axial magnetometer
readings. The available declared pose is used to recover the world-frame North
component; hidden sensor-error parameters and truth frames are not inputs. The
field is converted once from tesla to nT.

The frozen acquisition policy is:

- sample rate: 100 Hz;
- explicit observed-reference interval: first 8 s / 800 samples;
- causal window: 4 s / 400 samples;
- hop: 1 s / 100 samples;
- no future samples, padding, gap interpolation or per-window scaling;
- complete, uniformly sampled vector observations with stable declared
  calibration and pose.

For every node, reference acquisition stores the measured mean North component
and mean observed device temperature. The reference is immutable for that
session. Re-reference creates a new identity and invalidates dependent results;
the runtime does not continuously re-zero drift.

If reference acquisition is incomplete or invalid, the system reports a
warming-up/reference-unavailable state. It does not fabricate a reference or a
diagnosis.

## Shared State8 coordinates

Let `s_i(t)` be the focal node's observed North-component anomaly relative to
its frozen measured reference. Both profiles use these fixed coordinates:

| Index | Name | Definition | Unit |
| ---: | --- | --- | --- |
| f0 | `mean_north_anomaly` | arithmetic mean of `s_i` | nT |
| f1 | `north_anomaly_mad` | `1.4826 * median(abs(s_i - median(s_i)))` | nT |
| f2 | `north_anomaly_slope` | ordinary least-squares slope against acquisition time | nT/s |
| f3 | `lower_to_upper_band_power_db` | measured lower/upper band-power ratio | dB |
| f4 | `mean_temperature_delta` | observed mean temperature minus reference mean | K |
| f7 | `received_sample_fraction` | received sample count / 400 | dimensionless |

For f3, AQSE linearly detrends the observed channel, applies a Hann window, and
forms a one-sided periodogram. It integrates PSD bins using the frequency-bin
width over `[0.25, 2)` Hz and `[2, 20]` Hz, then computes
`10 log10(P_lower / P_upper)`. Each power uses the fixed `1e-12 nT²` numerical
floor. The floor is numerical regularisation, not a claimed sensor threshold.

Neither State8 profile has a phase coordinate. All eight real-valued
coordinates use their dedicated TRAIN-fitted State8 encoding policy; the
legacy phase wrap is not applied.

## `aqse.local-state8.v1`

The local profile completes the shared coordinates with:

| Index | Name | Definition | Unit |
| ---: | --- | --- | --- |
| f5 | `successive_difference_rms` | RMS of successive differences of `s_i` | nT |
| f6 | `lag1_pearson_correlation` | signed lag-one Pearson correlation | dimensionless |

This profile is valid with one node and drives the binary task `NORMAL` versus
`CHANGE_DETECTED`. It can report an observed change, but it cannot identify an
environmental or device cause from a single ambiguous channel.

With two nodes, AQSE still uses compatible local bundles. It may describe
shared or discrepant behavior, but it explicitly preserves the ambiguity of
which side is responsible.

## `aqse.network-state8.v1`

For a focal node `i`, the network profile constructs a pointwise leave-one-out
peer reference from the median of aligned, usable peer anomalies. Its remaining
coordinates are:

| Index | Name | Definition | Unit |
| ---: | --- | --- | --- |
| f5 | `peer_median_residual_rms` | RMS of focal anomaly minus peer median | nT |
| f6 | `signed_mean_peer_correlation` | arithmetic mean of signed focal/peer correlations | dimensionless |

The minimum compatible network is three nodes: one focal node and at least two
valid peers. Signed correlations are retained because negative spatial
responses can be physically meaningful. Axes are never correlated before the
declared pose transformation places them in the same world-frame quantity.

The residual is a robust observed common-field residual under a local
comparability assumption. It is not a reconstructed dipole residual, a source
localisation result, or proof of a device fault; a genuine nearby physical
source can also make it large.

## Missing peers and degraded routing

Missing, stuck, clock-invalid, clipped, non-vector, calibration-incompatible or
otherwise unusable peer readings are excluded before the peer median and
correlation are computed. They are never replaced by zero. A network window is
eligible only when at least two valid peers remain.

If the declared network has three or more nodes but the usable peer context for
a focal node falls below that minimum, the network feature record exposes null
coordinates and quality flags such as `peer_context_invalid` and
`insufficient_valid_peer_context`. The continuous runtime then routes that node
through the separately compatible local bundle and labels the result
`degraded_local`; it does not insert fake network values into a network model.
If the local window is also invalid, inference abstains.

The runtime modes therefore mean:

- N=1: `local`, change detection only;
- N=2: `local_two_node_ambiguous`, no manufactured culprit attribution;
- N>=3 with valid peers: `network`, peer-conditioned classification;
- N>=3 without sufficient valid peers: `degraded_local` or abstention.

Routing follows observable context and quality, never the hidden scenario or
the name of an injected event.

## State8 quality and provenance

Every record binds the session, node, profile fingerprint, frozen reference,
window start/end, source frame IDs, valid peer IDs, nullable feature values and
an eight-position validity mask. `valid_for_quantum` is true only when every
coordinate is present.

Initial State8 eligibility requires a complete uniform window, finite computed
features, no clipping, a valid frozen reference, stable declared calibration,
vector readout and usable pose. Undefined correlations remain null; zero is not
used as a fallback measurement. The received fraction is reported even when a
window is ineligible. Observable quality failures are separate from simulator
truth/audit flags.

The model bundle binds profile fingerprint, TRAIN-fitted scaler and encoding,
theta/TQK identity, AFSE space and downstream classifier. A local vector cannot
be sent to a network bundle, and a new reference or incompatible profile cannot
silently reuse stale results.

## Demonstration-study and replay reporting

`aqse-network-demo-v1` uses one focal-node primary example per independent
episode, so additional peer nodes and overlapping windows are not counted as
independent samples. Labels remain in a separate supervisory channel and are
not present in inference DTOs.

For evaluation, each episode also retains 17 causal replay windows. These
separately labelled traces support descriptive reporting of:

- sample and eligible counts by profile, node count and class;
- class support/recall, confusion matrices, balanced accuracy and macro-F1;
- abstentions, uncertainty, OOD flags and coverage;
- normal-episode/window false positives;
- first post-onset operational detection and p50/p95 delay, with censored
  changed episodes recorded explicitly.

Intervals resample independent episodes, not correlated windows. Replay metrics
are descriptive within the frozen simulator study; they are not hardware
detection guarantees.

## Classical comparators and scientific limits

Each compatible bundle retains three distinct outputs:

1. the quantum-kernel -> AFSE -> compact MLP path;
2. the same compact MLP fitted directly on the same raw State8 coordinates;
3. a TRAIN-NORMAL p99 observable threshold rule over absolute mean anomaly,
   MAD, absolute slope and, for the network profile only, peer residual.

The observable rule reports `NO_OBSERVED_CHANGE`, ambiguous common change,
ambiguous spatial disagreement or mixed observed change. It is explicitly a
non-causal engineering heuristic, not a neural prediction or calibrated model
score.

VALIDATION and the single frozen TEST evaluation record whether the
quantum->AFSE path helped, tied or hurt relative to the raw-feature MLP. The
software does not assume that the quantum path wins, and this document does not
invent or restate results that have not been produced by the canonical run.

All model scores are research outputs and are not probability-calibrated.
`ENVIRONMENT_COMPATIBLE`, `DEVICE_COMPATIBLE` and
`MIXED_OR_AMBIGUOUS` describe patterns inside the simulated domain; they are not
universal causal certificates. Real QPU operation, physical source inversion,
hardware calibration claims and field-deployment validation remain out of
scope.
