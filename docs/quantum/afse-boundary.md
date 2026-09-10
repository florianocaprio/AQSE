# Local Functional Embedding / AFSE Boundary

## Status

**Implemented and connected for the AQSE v1 local research demonstrator.**

The authorised method is `aqse.afse.nystrom-ridge32.v1`: a standard classical
regularised Nyström map built from the author's frozen quantum kernel. It is not
a newly invented quantum algorithm, a sensor wavefunction, or evidence of
quantum advantage. The protected VQC, TQK, alignment loss and QNG sources are
not redesigned by this component.

The earlier architecture-only boundary has therefore become a fitted,
versioned implementation. No performance value is asserted here; measured
VALIDATION/TEST evidence belongs to the canonical evaluation artifacts and
validation record.

## Position in the connected path

```text
observed State8 window
        -> frozen State8 scaler/encoder
        -> protected VQC(theta) and TQK fidelities to frozen landmarks
        -> fixed Nyström vector z(x)
        -> TRAIN-fitted standardisation
        -> compact MLP scores and uncertainty/context gates
```

AFSE fitting is an explicit TRAIN-only operation after the encoder and theta
candidate are fixed. Live inference never refits AFSE, changes landmarks or
grows a Gram matrix from query history.

## Why a live Gram row is not AFSE

For a live batch of `Q` samples, a row
`[K(x,x_1), ..., K(x,x_Q)]` changes dimension and meaning with batch membership
and order. It is relational batch geometry, not a stable local representation.

AQSE instead compares every query with one ordered, frozen landmark bank. The
result has one fixed coordinate system for the exact profile/encoder/theta
bundle. Adding, removing or reordering unrelated query rows cannot change an
individual `z(x)`.

## Frozen Nyström definition

At most 32 distinct TRAIN observations are selected as landmarks, balanced
across downstream task classes and using one distinct TRAIN lineage per
landmark. Selection is deterministic with seed `2001003`; VALIDATION and TEST
are not landmark sources. If fewer than 32 eligible independent TRAIN lineages
are available, AQSE uses the smaller recorded size without duplication or dummy
padding. That size remains fixed for the bundle.

For ordered encoded landmarks `L`, AQSE computes:

```text
W     = K_theta(L, L)
W_sym = (W + W.T) / 2
W_sym = U diag(s) U.T
lambda = 1e-6
B     = U diag((max(s, 0) + lambda)^(-1/2)) U.T
z(x)  = k_theta(x, L) @ B
```

`reference_size == output_dimension`, normally 32. The implementation checks
kernel/Gram invariants, symmetry and finiteness. Eigenvalues below
`-1e-10 * max(1, largest_eigenvalue)` are rejected; clipping only tiny negative
eigenvalues is recorded as floating-point stabilisation.

The optional diagnostic is:

```text
r(x) = K_theta(x, x) - ||z(x)||^2
```

Only negatives within `1e-10` are clipped to zero; a materially negative value
is an error. `r(x)` is a regularised kernel-reference reconstruction residual,
not a calibrated probability or proof that a physical cause is novel. The
heuristic OOD flag compares it with the empirical TRAIN residual p99 fixed in
the artifact. TEST does not define this threshold.

Landmark quantum states and `B` are cached for the exact encoder/theta identity.
Inference prepares query states and compares them with that fixed cache rather
than rebuilding all historical pairwise kernels.

## Fitted artifact and query contract

The non-executable JSON/numeric artifact retains:

- method/schema/content identities and effective source hashes;
- TRAIN dataset ID and digest;
- exact State8 feature profile, scaler and real-valued encoding-policy IDs;
- theta ID, digest, 16 values and fitted backend semantics;
- ordered landmark sample, lineage and class identities plus encoded values;
- requested/actual reference size and output dimension;
- ridge, eigenvalue tolerance, eigenspectrum and clipped-negative count;
- frozen `B` matrix, TRAIN residual p99 and heuristic OOD semantics.

A query supplies encoded features, unique sample IDs and an exact compatibility
context containing feature profile, scaler, encoding policy and theta identity.
Profile-compatible live acquisition provenance remains distinct from the
TRAIN dataset provenance. Mismatched or corrupted artifacts are rejected before
execution.

The output includes sample IDs, fixed-size real vectors, reconstruction
residuals and heuristic OOD flags. No simulator truth, event name, hidden defect
state or supervisory label is accepted by this boundary.

## Downstream model and matched baselines

The AFSE vector feeds `aqse.classical.mlp-32x16-tanh-lbfgs.v1`:

```text
z(x) -> TRAIN-fitted standardisation -> 32 tanh -> 16 tanh -> task output
```

The frozen implementation uses scikit-learn `MLPClassifier` with `lbfgs`,
`alpha=1e-3`, `max_iter=500`, `max_fun=15000`, `random_state=2001005` and no
internal early-stopping split. It persists bounded numeric JSON parameters,
not pickle/joblib, and verifies its portable NumPy evaluator against the fitted
estimator to at most `1e-9` maximum absolute score error.

Each bundle also stores:

- the same MLP architecture fitted directly on the same raw State8 features,
  with a separate TRAIN-fitted standardiser;
- a TRAIN-NORMAL p99 observable dispersion/spatial-residual rule.

The raw MLP is the matched classical baseline. The observable rule is a
separate non-causal engineering heuristic, not a model score or neural
prediction. For network profiles it can flag a peer-residual disagreement; it
still cannot prove whether a device or a spatially local physical field caused
that disagreement.

Bundle evaluation retains paired independent-episode comparisons and explicitly
records whether the quantum->AFSE path helped, tied or hurt relative to the raw
MLP. A stronger classical result is retained rather than hidden. No outcome is
claimed in this boundary document before the canonical evaluation produces it.

## Replay metrics and operational interpretation

The primary reporting unit is one independent episode/focal node. In addition,
17 causal windows per episode are retained as separately labelled replay traces.
Both AFSE and raw-baseline paths are evaluated on the same traces to report
normal false positives, first post-onset operational detection, detection-delay
p50/p95 and censored changed episodes, including node-count strata.

These replay values describe the bounded simulator study. Overlapping windows
are not treated as independent samples, and the figures are not field-detection
guarantees.

During live inference, quality, peer-context, uncertainty and heuristic-OOD
gates remain separate:

- invalid features cause abstention before AFSE;
- insufficient network peers route to a separately compatible local bundle or
  abstain;
- top score below 0.70 or top-two margin below 0.15 displays `UNCERTAIN`;
- scores are labelled `model score`, not probability-calibrated confidence;
- AFSE OOD is a TRAIN-residual heuristic, not a causal diagnosis.

## Versioning and atomic invalidation

Changing any of the following creates an incompatible representation space:

- State8 profile/fingerprint, declared calibration or reference identity;
- TRAIN-fitted scaler or encoding policy;
- theta, VQC/TQK semantics or backend compatibility;
- landmark membership/order, `B`, ridge or output dimension;
- AFSE-output standardiser, MLP parameters or class order.

The cohesive research bundle binds these components and their validation
evidence. Applying a new compatible bundle atomically invalidates landmark-state
caches and pending predictions. Old and new vectors or classifiers are never
mixed. Training completion does not silently promote a new bundle.

## Scientific limits

AFSE is a fixed classical representation of quantum-kernel similarities. Its
existence does not establish quantum superiority, calibrated uncertainty,
hardware sensitivity, physical source localisation or universally identifiable
causes. The network classes are conditional simulated-domain interpretations,
and one/two-node cases retain explicit attribution limits.

The local demonstrator may be functionally complete even when the measured AFSE
path ties or underperforms its raw-feature baseline. Predictive evidence and
negative findings must be reported from frozen artifacts without reopening
historical TEST data or tuning after TEST.
