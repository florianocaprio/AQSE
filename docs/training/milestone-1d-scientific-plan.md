# AQSE Milestone 1D — Scientific Training Plan

## Document control

| Field | Value |
| --- | --- |
| Status | **1D.0 design proposal — awaiting Floriano's scientific approval** |
| Date | 2026-09-09 |
| Working branch | `codex/milestone-1d-tqk-training` |
| Working baseline | `1cb07cdccabf9c655a63f2b23aab37383ae90b69` |
| Milestone 1C tag | `milestone-1c` → `95c5483c0192ba7605713c428e02b2527ab1f919` |
| Production behavior changed by 1D.0 | No |

The master prompt named `95c5483` as both the tag and `main` baseline. Before
this work, the approved infrastructure-health patch had intentionally advanced
`main` and `origin/main` to `1cb07cd`; the `milestone-1c` tag remained at
`95c5483`. This branch therefore starts from the current, clean `origin/main`
and retains that patch. No history or tag was rewritten.

This document specifies decisions for later Milestone 1D increments. It does
not authorize or claim a production training service, a new active encoder,
AFSE mathematics, a neural model, or continuous inference.

## 1. Existing boundary and terminology

The implemented path currently ends at a bounded, fixed-theta fidelity-kernel
preview. The following distinctions remain mandatory:

- the feature vector has eight values, but it is not a network-correlation
  representation;
- an `M x M` Gram matrix is relational batch geometry, not AFSE and not a
  fixed-dimensional sample embedding;
- QNG updates the VQC during a controlled training job; it is not an inference
  layer;
- simulator truth is a label/evaluation channel and never an encoder input;
- the process-local feature HMAC protects records only during one backend
  lifetime and is not durable observation provenance;
- the present quantum preview keeps its existing AngleScaler behavior and must
  remain compatible with its historical results.

The protected `tqk8.py` contract remains the authority for the implemented
circuit, kernel, loss, state derivatives, empirical Fubini–Study metric and QNG
step. Later application code may adapt inputs and orchestrate calls, but must
not reconstruct or silently reinterpret that mathematics.

## 2. Proposed first scientific task

### 2.1 Narrow claim

The proposed first task is binary discrimination between **nominal** and
**elevated simulated device white-noise conditions** under controlled harmonic
excitation. It is an integration and calibration experiment for the existing
feature-to-kernel training path. It is not evidence of general fault diagnosis,
physical quantum advantage, improved magnetometer sensitivity, or the ability
to distinguish arbitrary environmental events from instrument failures.

Proposed labels, pending approval:

- `-1`: nominal device-noise regime;
- `+1`: elevated device-noise regime.

The label builder derives the class from the declared intervention ledger, not
from SNR, variance, another predictor feature, sensor ID, seed, scenario name,
absolute event time, or a post-hoc threshold. Feature extraction receives only
measured samples and observable metadata.

The recommended first mechanism is an **episode-level noise regime**: one
declared white-noise setting applies for the complete episode, and the label is
constant for that episode. This avoids ambiguous windows that straddle a
`NODE_NOISE_BURST` transition. Burst-event classification remains an
alternative requiring a separately approved transition/exclusion policy; it is
not silently mixed into the first dataset.

### 2.2 Proposed controlled domain

The first experiment should use the implemented synthetic vector magnetometer,
a fixed declared calibration and pose, and a selected linear measured channel.
A periodic world-field excitation supplies enough cycles for the current
harmonic estimator. Each independent episode samples the following nuisance
variables independently of class:

- excitation amplitude, frequency and phase;
- ambient temperature and permitted thermal response;
- static world field and sensor bias within a predeclared bounded range;
- episode seed and, where included, drift and correlated-noise realizations.

The two class-conditional white-noise ranges must be chosen by a **training-only
pilot**. The pilot sweeps a declared grid, measures feature-window eligibility
and class overlap, then freezes one domain before validation/test generation.
It must not inspect the sealed test partition. A trivially separable SNR regime
is useful for integration smoke testing but must be reported as such.

The main study should use independent episodes rather than treating overlapping
windows as independent replicates. At least five complete seeded repetitions
are proposed for the final bounded comparison, subject to the measured runtime
budget. Anything smaller is labelled a smoke test.

### 2.3 Eligibility and abstention

The current harmonic quality policy remains unchanged. Dataset construction
must retain an audit record for every proposed window, including rejected
windows. It must report, by class and episode:

- proposed, accepted and rejected window counts;
- each quality-rejection reason;
- class coverage after eligibility filtering;
- abstention/eligibility rate and whether either class disappeared.

Rejected windows are neither silently dropped nor relabelled nominal. Training
is rejected when a partition has an empty class, constant labels, non-finite
features, or insufficient independent groups.

### 2.4 Mandatory negative control

A one-node common-field offset and an equal sensor-frame instrumental bias can
be observationally equivalent for a selected linear channel under a compatible
pose. Their truth labels may differ, but identical observable samples must
produce identical features, encoded inputs, kernel rows and predictions.

This counterexample is a required negative control. AQSE must not require or
claim above-chance separation when the observation contract contains no
identifying information. A later physical-event versus device-fault task needs
an approved network-aware feature contract and sufficient spatial/calibration
evidence; it is not part of the first task.

## 3. Feature and encoding contract

### 3.1 Raw feature semantics

The order remains exactly:

```text
[amplitude, phase, frequency, variance, drift, snr,
 spectral_peak, temperature]
```

Important limitations:

- `amplitude` is the fitted harmonic amplitude, not mean/DC field;
- `phase` is relative to each window start;
- `spectral_peak` is PSD magnitude, not the frequency coordinate of the peak;
- there is no DC mean, spatial residual, coherence or network correlation;
- adding a constant offset to a linear channel can therefore be information
  that the extractor intentionally discards.

A successful model cannot establish recovery of information absent from these
inputs.

### 3.2 Legacy encoding

The current preview fits the supplied `AngleScaler` on its declared reference
matrix and applies its mean/std/tanh map to all eight columns. This remains the
legacy policy and is not changed by Milestone 1D.0. Fitting on query,
validation, or test data is prohibited.

Because ordinary mean and standard deviation do not encode circular distance,
the legacy phase column can map nearby physical phases on opposite sides of the
`-pi/pi` cut to distant encoded coordinates. The isolated 1D.0 probe records
the actual behavior; it does not alter it.

### 3.3 Candidate direct-phase policy

The following is a proposal for later approval, not an active implementation:

1. preserve the raw eight-feature profile and order;
2. fit the unchanged AngleScaler on the training partition only;
3. transform all columns with that object;
4. keep its outputs for the seven non-phase columns;
5. replace encoded column `1` with observed phase wrapped by
   `((phase + pi) mod 2*pi) - pi`, with `+pi` represented as `-pi`;
6. retain `window_start` as the explicit phase origin;
7. store the scaler's phase mean/scale for provenance but mark them unused by
   this encoding policy.

Proposed identifier: `aqse.tqk8.encoding.phase-direct.v1`.

The legacy policy and this candidate are incompatible encoder versions. A
checkpoint from one must be rejected by the other. Wrapping resolves circular
coordinates only; it does not synchronize different window clocks or provide a
global phase reference.

Activation requires the real-circuit periodicity and seam-continuity tests,
Qiskit/NumPy agreement, explicit approval, and a separate later implementation
increment.

## 4. Dataset, label and split artifacts

### 4.1 Three separate immutable artifacts

Later 1D.1 should create three logically and physically separate artifacts:

1. **Observation/feature artifact** — immutable measured samples, observable
   metadata, window boundaries, feature records and quality; no simulator
   truth or labels in predictor columns.
2. **Label artifact** — episode/window references, `-1/+1` label, declared
   intervention source and label-policy version; accessible only to dataset
   assembly, loss and evaluation code.
3. **Experiment manifest** — lineage, hashes, versions, units, generation and
   split policy, seeds, exclusions, initial test-seal state and software
   identity.

A fourth operational artifact is separate from those immutable snapshots: an
**append-only test-access ledger**. Each entry links the prior entry digest,
the immutable dataset manifest digest, actor/process identity, reason and UTC
time. Opening the sealed test set adds a ledger entry; it never rewrites the
dataset manifest.

Generated datasets and checkpoints stay outside Git. Use bounded JSON for
metadata and non-executable numeric arrays loaded with `allow_pickle=False`.
Every payload receives a cryptographic content digest. A digest proves content
identity, not physical sensor authenticity.

### 4.2 Required manifest fields

At minimum:

```text
dataset_id, episode_id, generative_lineage_id, partition
full_git_sha, simulator_schema_version, observation_schema_version
feature_profile_id, canonical_feature_profile_fingerprint
extractor_version, quality_policy_version
encoding_policy_id, scaler_id, reference_dataset_id
physical_units_and_conversions, calibration_version, pose_provenance
raw_observation_digest, feature_artifact_digest, label_policy_version
window_boundaries_and_raw_sample_intervals
generation_seed, split_seed, optimizer_seed
declared_interventions, exclusions, eligibility_summary
created_at_utc, initial_test_seal_state, test_access_ledger_id
```

The existing ephemeral HMAC remains enforced for live preview. Durable replay
must revalidate immutable observations and deterministically recompute or
verify features without disabling that live check.

### 4.3 Independence and splitting

Create `generative_lineage_id` before generating paired variants or splitting.
The same group contains:

- every window from one simulator episode;
- all synchronized sensors in that episode;
- all overlapping windows sharing raw samples;
- deterministic replay or re-export of the same realization;
- paired variants sharing underlying random streams or latent realization.

Detect shared raw intervals using at least observation/acquisition digest,
sensor identity and start/end sample indices rather than transient UUID alone.
Assign whole groups to train/validation/test before any fitted preprocessing,
reference-bank selection, subsampling or optimization. The builder must detect
duplicate observation digests and intersecting raw-sample intervals across
partitions. A new session UUID is not evidence of independence.

The split manifest freezes group membership and seed. Training fits the scaler,
selects the bounded quantum reference bank and updates theta. Validation alone
selects configurations/checkpoints. Test stays sealed until that selection is
frozen; every test opening is append-only audit metadata.

## 5. Mathematical and orchestration contract

### 5.1 Supplied mathematics

For an approved encoded vector `a(x)`, retain the supplied fidelity kernel:

```text
K_theta(x,z) = |<psi_theta(a(z)) | psi_theta(a(x))>|^2
```

Retain the existing centered-alignment loss, `loss_grad_metric` and empirical
mean Fubini–Study metric convention exactly as implemented. Do not introduce a
factor of four, rename it measured-field Fisher information, or change the loss
when a study is unsuccessful. The optimizer must reuse `fit_qng` with its
damping, clipping and Armijo line search.

### 5.2 Proposed wrapper for 1D.3

The application must not invoke the scientific CLI. A dedicated future
training adapter must delegate through the public `StateEngine` and supplied
functions; it must not reach through `TQK8Adapter._engine` or modify either
adapter's protected dependency. The proposed wrapper calls
`fit_qng(..., steps=1, verbose=False)` repeatedly on the same frozen full
training set and current theta, recording only actually accepted updates. It
may be adopted only after a test shows equivalence to one
`fit_qng(..., steps=N)` call for fixed inputs, initial theta and optimizer
parameters. An empty one-step history is a real stop, not a fabricated epoch.

The wrapper runs off the HTTP event loop with one admitted job. Proposed states:
`CREATED`, `RUNNING`, `COMPLETED`, `CANCELLED`, `FAILED`. If a one-step call
returns no history, the only justified generic reason is
`NO_ACCEPTED_UPDATE`; the current protected API does not distinguish a small
gradient from exhausted line search. A more specific label requires separately
measured evidence. Cancellation is
cooperative between optimizer steps and never mutates the active model. A
checkpoint must be explicitly selected and explicitly applied.

Preview, diagnostics and future training must share a coordinated heavy-quantum
admission boundary or use an isolated worker process. Their current independent
guards are not sufficient once training exists. The simulator and lightweight
health endpoints must remain responsive.

Proposed starting bounds, pending benchmark and approval:

| Resource | Default | Hard maximum |
| --- | ---: | ---: |
| Frozen training windows | 32 | 128 |
| Accepted-update attempts | 10 | 50 |
| Concurrent training jobs | 1 | 1 |

Validation and test banks are independently bounded and frozen. The 128-row
engineering cap is not a claim of statistical sufficiency.

Log state preparations, gradient/metric evaluations, kernel and backtracking
evaluations, accepted delta, wall time and measured peak memory. Exact-state
preparations are not physical shots. Timeout values remain unresolved until the
1D.0 budget probe and later representative benchmark are reviewed.

## 6. Checkpoint and compatibility proposal

A checkpoint should consist of immutable JSON metadata plus numeric arrays in a
non-executable container. Required fields include:

- checkpoint/schema ID and content digests;
- full Git SHA and protected-source hashes;
- feature, quality and encoding policy versions;
- scaler snapshot and reference-dataset ID;
- initial, final and separately selected theta snapshots;
- exact circuit/backend semantics and TQK version;
- dataset/split/group manifests and immutable training-window IDs;
- optimizer parameters, seed, real accepted-step history and stop reason;
- validation selection rule and measured metrics;
- creation time, software/runtime versions and compatibility tuple.

Loading rejects a mismatch in the canonical fingerprint of the complete feature
profile—including channel, window geometry, sampling, order, units and quality
policy—or in phase origin, encoding,
scaler, circuit source, theta shape, backend semantics, dataset lineage or
checkpoint schema. Applying a compatible theta creates a new active encoder
version and invalidates dependent kernels/representations; it does not alter
stored observations. There is no automatic promotion, warm start or browser
`localStorage` persistence. Artifact writes must be atomic: write and verify a
temporary bounded payload, then rename it into the configured artifact root
outside Git.

## 7. Comparative validation plan

All methods use identical group partitions, observable information and frozen
evaluation episodes:

1. training-selected SNR threshold (strong task-specific baseline);
2. RBF SVC with phase represented as `sin(phase), cos(phase)` or an equivalent
   circular distance;
3. fixed-theta TQK plus explicitly labelled research-only precomputed-kernel
   SVC;
4. ordinary-gradient training of the same supplied loss/gradient plus the same
   evaluator;
5. supplied QNG training plus the same evaluator.

The nine classical coordinates produced by replacing one phase coordinate with
its sine/cosine pair add no new observed information. Matched initialization
seeds and training banks are required for GD/QNG. Hyperparameter candidates and
selection budgets are frozen before validation; the test set never chooses
them.

Report training and validation alignment separately from held-out balanced
accuracy, macro-F1, confusion matrix, per-class coverage, eligibility/abstention
and actual compute cost. AUC is reported only for valid scores with both
classes. Decision scores are not probabilities. Intervals resample independent
lineage groups, not overlapping windows, and are omitted with an explicit
reason when too few groups exist.

Quantum superiority is not an acceptance condition. Negative, null and
inconclusive results are retained.

## 8. Acceptance matrix for later increments

| Requirement | Earliest gate | Required evidence |
| --- | --- | --- |
| Protected scientific assets unchanged | Every gate | SHA-256 manifest and original tests |
| Deterministic dataset generation | 1D.1 | Same manifest/seed → same digests |
| Replay and overlap group isolation | 1D.1 | No lineage or raw-sample intersection across partitions |
| Observation/label separation | 1D.1 | Schema tests and truth-free encoder input |
| Train-only fitting and reference selection | 1D.2 | Query/test perturbation leaves fitted artifacts unchanged |
| Phase periodicity and seam continuity | 1D.2 | Qiskit and NumPy numerical tests with declared tolerance |
| Version mismatch rejection | 1D.2 | Feature/encoding/scaler/checkpoint negative tests |
| Invalid objective rejection | 1D.3 | Constant label, empty class, non-finite and collapsed-kernel tests |
| Kernel numerical invariants | 1D.3 | Symmetry, diagonal, range, PSD tolerance and cross-kernel orientation |
| Original-wrapper equivalence | 1D.3 | Repeated one-step vs `steps=N` theta/history comparison |
| Honest lifecycle and bounded work | 1D.3 | Cancellation, busy, stop-reason and concurrency tests |
| Frozen inference | 1D.4 | Invariance to query ordering and unrelated batch members |
| Truth-invariant prediction | 1D.4 | Observation-equivalent negative control |
| Independent evaluation | 1D.4 | Sealed test ledger and grouped metrics |
| Label sanity | 1D.4 | Permutation at independent label-assignment unit |
| Checkpoint durability | 1D.5 | Restart/reload with digests and compatibility validation |
| UI causality and invalidation | 1D.5 | Explicit actions; stale responses cannot overwrite current state |
| Simulator responsiveness | 1D.6 | Bounded latency while one training job runs |

Numerical tolerances must be fixed before held-out evaluation. Tests validate
contracts and predictable mathematics; they must not require every dataset to
converge or QNG to beat a baseline.

## 9. Proposed implementation map — not yet authorized

Names below are planning targets and may be refined during review without
changing existing module boundaries:

```text
backend/app/training/
  models.py              # immutable DTOs and compatibility tuple
  datasets.py            # observation artifact assembly and grouped split
  labels.py              # separate intervention-to-label policy
  encoding.py            # approved policy adapter; legacy remains unchanged
  runner.py              # bounded wrapper around supplied fit_qng
  checkpoints.py         # safe persistence, hashes and compatibility
  evaluation.py          # frozen comparisons and test-seal ledger
  service.py             # one-job orchestration outside event loop

backend/app/api/training.py
  # endpoints only after their increment is explicitly approved

backend/tests/training/
frontend/src/worksheets/QngWorksheet.tsx
frontend/src/state/...
  # later explicit job/request revision, checkpoint controls and stale-response
  # rejection; no AFSE or neural implementation
```

No candidate continuous-network feature profile is included in this map. Its
bands, residual estimator, coherence, calibration, missing-context and one-node
semantics require a separate scientific decision.

## 10. Decisions requiring Floriano's approval

Before 1D.1 or later work, approve or revise:

1. the nominal/elevated device-noise task and the meaning of `-1/+1`;
2. selected measured channel, episode duration, harmonic domain and training-only
   pilot rule for noise ranges;
3. minimum independent group counts and train/validation/test proportions;
4. durable artifact location, retention, atomic-write policy and test-seal
   authority;
5. `generative_lineage_id` composition and paired-variant policy;
6. legacy encoding only versus the proposed direct-phase policy;
7. phase-origin requirement and tolerance for periodicity/seam tests;
8. theta initialization distribution and matched-seed policy;
9. training/reference limits, update budget, timeout and cancellation semantics;
10. comparator hyperparameter budgets, checkpoint selection metric and interval
    method;
11. whether the research-only precomputed-kernel SVC is approved solely as an
    evaluator;
12. the stage at which the sealed test partition may be opened.

AFSE mathematics, output dimension and fitting policy remain separate future
decisions. No neural architecture is proposed here.

## 11. Reference basis

The repository contracts and Floriano's protected source take precedence over
external literature. Methodological references are:

- James Stokes et al., [Quantum Natural Gradient](https://quantum-journal.org/papers/q-2020-05-25-269/), *Quantum* 4, 269 (2020).
- Thomas Hubregtsen et al., [Training Quantum Embedding Kernels on Near-Term Quantum Computers](https://arxiv.org/abs/2105.02276), *Physical Review A* 106, 042431 (2022).
- scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) and [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).

These sources motivate QNG, trainable-kernel alignment and leakage-safe grouped
evaluation. They do not prescribe AQSE's candidate task, phase adapter, stage
gates or engineering limits.

Related project records:

- [AQSE development roadmap](../roadmap/aqse-development-roadmap.md)
- [Milestone 1D.0 design and compatibility audit](../validation/milestone-1d-design-audit.md)
