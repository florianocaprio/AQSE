# AQSE TQK8 Quantum Preview Contract

## Scope

The implemented Milestone 1C quantum preview is a bounded, explicit
sensor-to-kernel computation. It is not QNG training, not a classifier, not
AFSE, and not a physical QPU job.

The preview must delegate numerical work to Floriano's existing TQK8
implementation through the application adapter. This document does not modify
or reinterpret the supplied circuit, fidelity kernel, derivatives, loss,
Fubini–Study metric, QNG update, or original validation tests.

## Existing TQK8 contract

The validated circuit accepts eight encoded feature angles, uses eight logical
qubits and sixteen trainable parameters, and contains seven CZ gates.

For qubit i, the ordered sequence is:

    RY(x_i)
      ↓
    RZ(theta_i)
      ↓
    CZ entanglement
      ↓
    RY(theta_8+i)
      ↓
    RZ(x_i)

The CZ edges are:

    (0,1), (2,3), (4,5), (6,7),
    (1,2), (3,4), (5,6).

Each input feature is uploaded twice, first through RY and later through RZ.
The two trainable groups are alpha = theta[0:8] and beta = theta[8:16]. A
diagram that shows only one data-upload gate is scientifically incorrect.

For encoded rows x and z, the exact-state fidelity kernel is

    K_theta(x,z) = |<psi_theta(z) | psi_theta(x)>|².

The self-kernel over M rows is a real M×M Gram matrix.

## Preview pipeline

    selected quality-valid, server-signed feature windows
                    ↓
            raw X, shape (M,8)
                    ↓
       existing fitted AngleScaler snapshot
                    ↓
         encoded angles, shape (M,8)
                    ↓
       existing VQC with fixed theta snapshot
                    ↓
              existing TQK
                    ↓
            real kernel matrix K

The workbench default is 16 windows. The hard maximum is 128 windows in total
across reference and query sets; the API rejects larger previews rather than
truncating silently. At least two reference windows are always required.

All rows must:

- contain exactly eight finite values;
- be marked valid for preview under the same quality-policy version;
- share one exact FeatureProfile;
- have names and units matching that profile;
- identify their source window;
- carry a valid server-issued provenance token for the complete profile and
  record.

Invalid, forged, or incompatible records cause an explicit request-validation
error that identifies the affected record where applicable. They are never
silently dropped or converted to zeros.

The current feature record does not carry a calibration-version field. Durable
multi-session evaluation must add and verify that lineage before claiming
calibration compatibility.

## Existing AngleScaler

Milestone 1C reuses the supplied AngleScaler. It must not introduce a second
formula with the same name.

For reference matrix X_ref, the existing implementation computes each column:

    mean_j  = mean(X_ref[:,j])
    std_j   = population standard deviation of X_ref[:,j]
    scale_j = std_j when std_j > 1e-12, otherwise 1.

For any compatible query matrix X:

    z = (X − mean) / scale
    encoded = (π/2) tanh(z/2).

The output angles are smoothly bounded by ±π/2. Raw feature values remain in the
feature record and preview response; encoded angles do not replace them. Durable
archival remains a separate pending capability.

### Reference dataset policy

The scaler is fitted once on an explicitly selected reference dataset. Its
snapshot stores:

    algorithm identifier
    deterministic scaler version
    reference_dataset_id
    feature_profile_id
    mean[8]
    scale[8]

Query rows never fit their own independent scaler implicitly. Future held-out
evaluation data must not influence fitting.

For the first exploratory preview, the approved self-reference mode may use the
selected preview windows as X_ref. The UI and result mark this mode as
SELF-REFERENCE — EXPLORATORY. It is not a held-out evaluation and its scaler
must not be reused under a different profile without explicit selection.

## Draft and executed theta

The frontend maintains sixteen finite draft values. Editing them does not call
the backend. Run Quantum Preview creates an immutable executed snapshot:

    draft_theta[16] --explicit run--> executed_theta[16]

The request passes a copy. The computation does not mutate theta and invokes no
optimizer. The backend response echoes the executed values. Theta version and
the comparison between draft and executed state are workbench-lineage fields,
not fields currently issued by the preview backend response. If the draft
changes afterward, the workbench must mark the existing heatmap stale.

The backend accepts the finite values supported by the supplied TQK8 contract;
it does not silently clamp or reorder them. A UI display range is an editing aid
and not a different scientific parameterization.

## Backend mode

The approved interactive default is the existing Qiskit Statevector engine.
The independent NumPy state simulator remains an explicit selectable reference
and cross-check backend. Both are ideal exact-state simulations.

The preview does not:

- use Aer shots unless a separately named future mode implements them;
- simulate hardware noise while labelling the result ideal;
- access IBM Runtime or credentials;
- contact a physical QPU;
- expose statevectors or fabricated quantum amplitudes in the UI.

If a selected backend is unavailable, the result reports failure or an explicit
user-selected fallback. It must not relabel NumPy output as Qiskit output.

## Implemented backend request contract

A preview request contains:

    mode: self_reference | reference_query
    backend: qiskit | numpy
    complete feature_profile
    reference_dataset_id
    reference_windows with raw features, quality, IDs, and provenance
    optional causal query_windows with the same contract
    theta[16]

The current stateless request sends reference rows directly and either uses them
as the exploratory query batch or supplies a separate causal query batch. It
does not accept a theta-version field; the workbench records that version next
to the immutable request snapshot. Payload size, reference size, M, numeric
finiteness, identifiers, profile compatibility, reference diversity, and
feature provenance are validated before circuit construction.

## Implemented backend result contract

A successful preview contains:

- deterministic preview identifier;
- ordered window identifiers;
- raw feature rows and encoded angles or an inspectable selected-row mapping;
- real M×M reference-kernel values and, for reference-query mode, the query ×
  reference cross-kernel;
- global minimum and maximum;
- mean over i ≠ j;
- maximum diagonal deviation max_i |K_ii − 1|;
- symmetry deviation;
- measured server execution duration;
- exact backend mode;
- feature-profile snapshot;
- scaler snapshot and reference identifier;
- executed theta snapshot.

The response does not currently issue an execution timestamp or theta-version
identifier. Those belong to the workbench execution snapshot and must not be
misrepresented as backend fields.

Mathematically, an ideal fidelity Gram matrix is symmetric, positive
semidefinite, has unit diagonal, and has entries in [0,1]. Floating-point tests
use a declared numerical tolerance. Scientific values are not silently clipped
to conceal a violation; presentation-only color-domain clipping, if used, is
clearly separated from returned values.

## Kernel is not AFSE

K describes pairwise relations inside the selected batch. Its row length changes
with M and its meaning depends on the other selected samples. It is therefore
not automatically the fixed-size local representation z(x) required by AQSE.

The heatmap is labelled Fidelity Kernel or Gram Matrix. It must not be labelled
Local Embedding, AFSE, latent vector, or learned sensor state.

## Preview is not training

The preview performs no alignment-loss evaluation, derivative computation,
Fubini–Study metric evaluation, or QNG step. Its theta is fixed from request to
response.

The separate training loop remains:

    labelled training X
       → K_theta
       → centred alignment loss
       → gradient and Fubini–Study metric
       → damped QNG update
       → candidate theta.

The currently supplied scientific demo uses a binary centred-alignment target
and an SVC. It is not silently advertised as a multilabel network trainer or the
final AQSE neural model. Any new task or loss semantics require explicit
implementation and validation.

## Provenance boundary

Before any scaling or circuit construction, the backend verifies the required
HMAC token on every feature record against its complete feature profile. A
modified feature, quality flag, identifier, time boundary, or profile is
rejected. The HMAC key is ephemeral to the backend process, so records must be
re-extracted after restart.

This is an integrity check for feature records produced by the current process.
It is not physical-sensor authentication, raw-acquisition attestation, or a
durable archival signature. The preview remains exploratory when reference and
query are the same selected windows, even when their tokens are valid.

## Required safeguards

- Reuse the original AngleScaler, VQC, and TQK code paths.
- Preserve feature and theta order.
- Fit only on the declared reference data.
- Reject non-finite or quality-invalid inputs.
- Verify every feature-record provenance token before scaling.
- Enforce M ≤ 128 before expensive work.
- Keep theta unchanged during preview.
- Record actual duration and backend.
- Permit cancellation at the orchestration boundary when implemented.
- Do not import or invoke QNG from the preview path.
- Prevent network access and physical-QPU submission.
- Invalidate cached results when feature, scaler, theta, or backend version
  changes.
