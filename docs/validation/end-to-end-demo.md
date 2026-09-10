# AQSE end-to-end research demonstrator validation

## Record policy

This is the single detailed evidence location for the end-to-end v1 delivery.
It deliberately separates:

- **frozen decisions**, recorded before canonical generation/evaluation;
- **implemented behavior**, established by source and component contracts;
- **measured results**, added only after the named command really completes.

`PENDING` and `NOT RUN` are not passes. Source presence is not a performance
result. The scientific label for every artifact and UI output is
`research / not validated for field deployment`.

## Pre-execution decision freeze

The following design was fixed before canonical generation, fitting,
VALIDATION selection or the one-time new-study TEST evaluation.

### Study and partitioning

- Study: `aqse-network-demo-v1`.
- Population: 160 independent episodes: four declared scenarios × node counts
  1–8 × five replicates.
- Split assigned before simulation inside every scenario/node-count cell:
  three TRAIN, one VALIDATION, one TEST; totals 96/32/32.
- Primary reporting unit: one predeclared focal-node example per independent
  episode, never every correlated peer/window.
- Seeds: generation 2001001, split 2001002, bank/landmarks 2001003, theta
  2001004, neural 2001005 and paired bootstrap 2001006.
- Canonical generation domain separator:
  `aqse-network-demo-v1/canonical-generation/v3`.
- Acquisition: 100 Hz, 28 s per episode, observed reference `[0,8)`, event
  start 14 s, and supervised focal window `[18,22)`.
- The moving environmental dipole and focal thermal ramp are physically
  inactive before 14 s, activate on the common `[14,24)` event window, and
  use elapsed time relative to that onset. Their onset is sealed in the
  episode-plan validator rather than represented by a label-only marker.
- Scenarios: `NORMAL`, `ENVIRONMENT_COMPATIBLE`,
  `DEVICE_COMPATIBLE`, `MIXED_OR_AMBIGUOUS`.
- NORMAL receives a same-schedule zero-amplitude sham event. Sensor IDs/order,
  direction, nuisance temperature and admissible severities are generated
  independently of class where appropriate.

### Frozen simulation domain

These values are source-contract decisions, not measured limits:

| Quantity | Frozen range/value |
| --- | --- |
| base white noise | 0.15–0.60 nT |
| base OU noise | 0.05–0.25 nT |
| base temperature offset | -1.5–1.5 K |
| common North perturbation | 6–18 nT |
| focal extra perturbation | 2–6 nT |
| moving source initial x | -4.2 to -3.8 m |
| moving source y | -0.35 to 0.35 m |
| moving source height | 1.0–1.4 m |
| moving source x velocity | 0.19–0.21 m/s |
| focal noise multiplier | 4–8 |
| focal temperature ramp | 0.04–0.08 K/s |
| focal thermal bias | 2–5 nT/K |

Changing these values or the protocol digest creates another study. They must
not be tuned after TEST.

### Feature and routing contract

- Profiles: `aqse.local-state8.v1` and
  `aqse.network-state8.v1`, incompatible with the historical harmonic
  profile.
- Frozen measurement/reference/window policy: observed calibrated world-frame
  North component, 8 s reference, 4 s causal window, 1 s hop at 100 Hz.
- Common coordinates: mean anomaly, 1.4826×MAD, OLS slope, lower/upper-band
  Hann-periodogram power ratio, mean temperature delta and received fraction.
- Local coordinates f5/f6: successive-difference RMS and lag-one Pearson
  correlation.
- Network coordinates f5/f6: leave-one-out peer-median residual RMS and
  arithmetic mean of signed aligned peer correlations.
- No imputation. Nullable values, per-feature validity and explicit quality
  abstention are retained.
- N=1 local only; N=2 local with attribution ambiguity; N≥3 network only with
  compatible peers, otherwise declared local degradation/abstention.
- Encoder: `aqse.state8.angle-scaler-all8.v1`, protected AngleScaler fitted on
  TRAIN and applied normally to all eight non-phase coordinates.

### Quantum, AFSE and classifier contract

- Exactly two theta candidates per task: deterministic nonzero theta0 and the
  result of at most ten accepted calls to the unchanged protected
  `fit_qng`.
- QNG bank: at most 32 independent TRAIN lineages, balanced across its binary
  task, with at least four eligible examples/class or a declared ineligible
  optimization.
- QNG budget: learning rate 0.2, damping 1e-3, maximum step norm 0.4.
- Local QNG label: NORMAL (-1) versus CHANGE_DETECTED (+1).
- Network QNG label: ENVIRONMENT_COMPATIBLE (-1) versus DEVICE_COMPATIBLE
  (+1); NORMAL and MIXED/AMBIGUOUS do not enter the protected binary loss.
- AFSE: `aqse.afse.nystrom-ridge32.v1`, up to 32 distinct balanced TRAIN
  landmarks, ridge λ=1e-6, negative-eigenvalue relative tolerance 1e-10 and
  TRAIN-residual p99 heuristic OOD gate.
- MLP: `aqse.classical.mlp-32x16-tanh-lbfgs.v1`, TRAIN-only standardisation,
  32/16 tanh layers, LBFGS, alpha 1e-3, `max_iter=500`,
  `max_fun=15000`, seed 2001005, no early stopping.
- Persistence: bounded JSON/numeric arrays, no pickle/joblib; portable NumPy
  inference must agree with fitted sklearn scores to at most 1e-9.
- Operational uncertainty: top score <0.70 or top-two margin <0.15.
- Baseline: the same compact MLP on the same raw State8 input with TRAIN-only
  standardisation.
- Candidate selection: maximum VALIDATION balanced accuracy, then macro-F1,
  then theta0 on a tie.
- TEST: one evaluation only after the selection freeze; it cannot change the
  winner, method or threshold.

Protocol source digest before canonical generation:
`9f8545865a1d176220803f11fa66505cbe0753d2e23f11137eb34db0400dab60`.
The canonical artifact must record and match this digest; otherwise this record
must be updated as a new pre-execution decision before any measurement.

## Test sealing and historical evidence

The new study stores observations and labels in physically separate
partition-specific files. Its append-only ledger has the fixed access plan:

1. sequence 0: `sealed`, created at publication;
2. sequence 1: `opened` for frozen TEST observations under the selection
   freeze;
3. sequence 2: `opened` for matching TEST labels and the one final
   evaluation.

Any retry after sequence 2 is rejected. A missing final evaluation after an
advanced ledger is a hard recovery error, not permission to reopen TEST.

The historical Milestone 1D study is not the new study and must not be read
semantically, regenerated or resealed. Its expected ledger SHA-256 is:

`e0d4898141d8be07f4a4f1af7582791eae61994ced52bb7b3dba30a7ddda4b70`.

## Pre-freeze probe disclosure and invalidation

During implementation, a transient pre-freeze smoke probe simulated 20 N=3
episodes to reproduce a window-selection defect and inspect finite feature
output. It wrote no dataset, model, manifest, access ledger or other artifact;
it computed no metric and informed no hyperparameter, selection or performance
tuning.

Some provisional split assignments could have corresponded to future TEST, so
those identities are explicitly non-canonical. Before canonical generation,
the v2 domain separator above was added to every seed, parameter, lineage and
episode derivation. Automated fixtures use the disjoint namespace
`aqse-network-demo-v1/test-fixture-generation/v1` and seed 91000001.
Regression tests require canonical and fixture identities to be disjoint.

## Implemented behavior

This section records code paths present in the branch. It does not claim that
the final delivery suite has passed.

### State8 and study

- Typed immutable references, profiles, records, nullable values and quality
  masks.
- Batch extraction plus a latest-window path that does not need to recompute
  every historical window.
- World-frame observed-value boundary and hidden-truth exclusion.
- Deterministic 160-episode generator with focal-node and lineage identities.
- Separate observation/label payloads, allowlisted bounded readers, atomic
  idempotent writes and partition-specific loaders.
- Opaque historical-ledger verification and chained new-study TEST ledger.

### Fitting, bundle and persistence

- Protected QNG wrapper equivalence and small Qiskit/NumPy preflight before
  candidate fitting.
- TRAIN-only State8 encoder; candidate AFSE and raw-feature downstream models.
- Nyström eigenvalue checks, immutable landmark order, cached landmark states,
  fixed query dimension and query-order/batch invariance.
- Safe numeric MLP artifact and verified NumPy evaluator.
- Complete bundle compatibility across task/profile/scaler/theta/TQK/AFSE/MLP.
- Atomic bundle/freeze/evaluation writes and atomic active-pair pointer.
- Read-only registry summary; opening the registry performs no training or TEST
  access.

### Continuous worker and API

- One observation-only worker per session, bounded worker registry, 1,600-frame
  deque, one newest-window slot and 512 retained latency samples.
- Session reference acquisition, latest-window routing, quality abstention,
  bundle-cache invalidation, worker epoch, p50/p95 latency, result age and
  skipped-window accounting.
- Shared heavy-slot pause during training; no QNG in inference and no automatic
  retraining.
- Persistent idempotent training intent, single admitted heavy job, real
  accepted-step updates and cooperative cancellation.
- REST endpoints:
  - `GET /api/demo/registry`;
  - `POST /api/demo/analysis/{session_id}/start`;
  - `GET /api/demo/analysis/{session_id}`;
  - `POST /api/demo/analysis/{session_id}/stop`;
  - `POST /api/demo/training/jobs`;
  - `GET /api/demo/training/jobs/{job_id}`;
  - `POST /api/demo/training/jobs/{job_id}/cancel`;
  - `POST /api/demo/bundles/apply`.

### GUI

- Overview: connected path, registry/application, reference/analysis status,
  latest result, quality and age.
- Sensors: 1–8 nodes, lifecycle, draft/executed distinction, real event API and
  bounded event log.
- Features: live State8 values/reference/context/validity plus isolated legacy
  harmonic workflow.
- Quantum Engine: protected circuit and frozen bundle identity, separate
  editable manual preview.
- QNG Training: actual dependency diagram, start/status/cancel/progress/apply.
- AFSE: actual method/reference/dimension/residual and real z(x).
- Neural Model: actual architecture/scores/uncertainty/conditional output and
  raw baseline.
- Experiments: registry, saved bundles, final read-only metrics, explicit apply
  and exports.
- Request IDs, cancellation/AbortSignal use and session/worker epochs protect
  against stale async responses.

## Canonical study and model results

No value may be inserted into this section until it is read from verified
canonical artifacts produced by `make prepare-demo`.

| Field | Actual result |
| --- | --- |
| Study artifact ID | **PENDING CANONICAL RUN** |
| Study content digest | **PENDING CANONICAL RUN** |
| Selection freeze ID | **PENDING CANONICAL RUN** |
| Final evaluation ID | **PENDING CANONICAL RUN** |
| New TEST ledger final SHA/count | **PENDING CANONICAL RUN** |
| Selected local bundle/theta/accepted QNG steps | **PENDING CANONICAL RUN** |
| Selected network bundle/theta/accepted QNG steps | **PENDING CANONICAL RUN** |
| Local AFSE M/dimension/eigenvalue notes | **PENDING CANONICAL RUN** |
| Network AFSE M/dimension/eigenvalue notes | **PENDING CANONICAL RUN** |
| MLP convergence warnings | **PENDING CANONICAL RUN** |
| Local VALIDATION and TEST metrics/support/confusion | **PENDING CANONICAL RUN** |
| Network VALIDATION and TEST metrics/support/confusion | **PENDING CANONICAL RUN** |
| Raw baseline comparisons | **PENDING CANONICAL RUN** |
| Quantum→AFSE helped/tied/hurt | **PENDING CANONICAL RUN** |
| Bootstrap degeneracy/uncertainty notes | **PENDING CANONICAL RUN** |

TEST results must remain unchanged even if unfavorable. No additional study,
seed, model threshold or hyperparameter search is authorised by a negative
finding.

## Final software validation

Record the exact final command, exit status, count and material warning.

| Check | Exact command | Result |
| --- | --- | --- |
| Container build | `make build` | **PENDING FINAL RUN** |
| Backend pytest | via `make test` | **PENDING FINAL RUN** |
| Protected original quantum tests | via backend suite | **PENDING FINAL RUN** |
| Ruff | via `make test` | **PENDING FINAL RUN** |
| Frontend TypeScript | via `make test` | **PENDING FINAL RUN** |
| Frontend ESLint | via `make test` | **PENDING FINAL RUN** |
| Frontend Vitest | via `make test` | **PENDING FINAL RUN** |
| Vite production build | via `make test` | **PENDING FINAL RUN** |
| Diff whitespace | `git diff --check` | **PENDING FINAL RUN** |

## API and browser acceptance

`make acceptance` must write a JSON report outside Git under the configured
artifact root. The final run must cover:

| Scenario | Result/evidence |
| --- | --- |
| backend/frontend/lightweight quantum health | **PENDING** |
| prepared registry and active compatible pair | **PENDING** |
| N=1 local-only acquisition/inference | **PENDING** |
| N=2 attribution-ambiguous acquisition/inference | **PENDING** |
| N=4 connected network acquisition/inference | **PENDING** |
| N=8 connected network acquisition/inference | **PENDING** |
| common environmental event | **PENDING** |
| focal S3 drift/noise | **PENDING** |
| moving/local spatial source | **PENDING** |
| shared offset and mixed/ambiguous condition | **PENDING** |
| dropout, clipping and stuck abstention/flags | **PENDING** |
| training start/cancel/duplicate intent and health responsiveness | **PENDING** |
| explicit atomic bundle application | **PENDING** |
| mismatched bundle rejection | **PENDING** |
| reset/replay reproducibility and stale-response protection | **PENDING** |
| restart and saved-bundle reload without auto-training | **PENDING** |

Browser acceptance must use the real app at both viewports and preserve
meaningful screenshots outside Git:

| Viewport | Result | Screenshot/report |
| --- | --- | --- |
| 1440×900 | **PENDING** | **PENDING** |
| 390×844 | **PENDING** | **PENDING** |

## Eight-node real-wall-clock soak

Required command: `make soak` with the default 600-second duration. Accelerated
simulator time does not reduce the required wall time.

| Measure | Actual result |
| --- | --- |
| Environment / host | **PENDING** |
| Wall duration | **PENDING** |
| Node count / sampling / hop | **PENDING** |
| Backend/frontend health throughout | **PENDING** |
| Analysis windows completed/skipped | **PENDING** |
| Maximum queue depth/buffer behavior | **PENDING** |
| Analysis processing p50/p95 | **PENDING** |
| Result-age observations | **PENDING** |
| RSS start/end/peak/trend | **PENDING** |
| Report path | **PENDING** |

The final interpretation must be “measured on this local run”, not a hard
real-time or general stability guarantee.

## Protected identities and repository integrity

Expected hashes to remeasure at the end:

| Asset | Expected SHA-256 | Actual |
| --- | --- | --- |
| `tqk8.py` | `cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689` | **PENDING** |
| `sampler_qng.py` | `7489c5c2b3eb0783499c333dc0826994b146d7cf5631469a0520bac9570b2ea6` | **PENDING** |
| original TQK8 test | `c87c56c0eb3f1ea20d2dd124427aea96f868ae4674b5b390e2688e9da8bb0c09` | **PENDING** |
| original notebook | `9d7e0337af7c039894fc6c53567de2883063b32c89e368753bd01e1d9e9e82be` | **PENDING** |
| historical TEST ledger | `e0d4898141d8be07f4a4f1af7582791eae61994ced52bb7b3dba30a7ddda4b70` | **PENDING** |

Final repository checks:

| Check | Actual |
| --- | --- |
| branch | `codex/milestone-1d-tqk-training` |
| final local SHA | **PENDING FINAL COMMIT** |
| remote SHA/divergence | **PENDING FINAL PUSH** |
| `main` unchanged | **PENDING FINAL RECHECK** |
| `milestone-1c` tag unchanged | **PENDING FINAL RECHECK** |
| secrets/runtime/build/ZIP files excluded | **PENDING FINAL RECHECK** |
| final `git status` clean | **PENDING FINAL RECHECK** |

## Scientific conclusion

Two separate conclusions must be written after the measured sections exist:

1. **Functional research demonstrator:** pending final connected-path,
   browser, restart and soak evidence.
2. **Predictive performance / quantum benefit:** pending the frozen canonical
   VALIDATION/TEST artifacts; no direction is assumed in advance.

Until then, the correct delivery statement is: **working end-to-end draft under
final validation**, not final scientific acceptance.
