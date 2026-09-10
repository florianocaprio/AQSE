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

The values below were read from the verified immutable artifacts produced by
`make prepare-demo`. Re-running that command reused the same study, freeze and
final evaluation and did not reopen TEST.

| Field | Actual result |
| --- | --- |
| Study artifact ID | `aqse-network-study-5f33f5c4d856361f` |
| Study content digest | `5f33f5c4d856361fd94f3c2bc93867f8dee84b1f088450457f3bbee4bdc00c12` |
| Selection freeze ID | `aqse-demo-freeze-39c61760d233f694` |
| Final evaluation ID | `aqse-demo-final-ad5fb1055eb68ead` |
| New TEST ledger final SHA/count | `c4431d837d9a486625d0c7bf3cd7117952c0c333dc08e80ee66921e8e5d45f6d`; exactly 3 events (`sealed`, observations open, labels/final open) |
| Selected local bundle/theta/accepted QNG steps | `aqse-demo-bundle-06a2399a1c4c853b`; `protected_qng`; 10 accepted updates, `MAX_UPDATES_REACHED`; alignment loss 0.855553 → 0.728577 |
| Selected network bundle/theta/accepted QNG steps | `aqse-demo-bundle-cd888ecfb28b2145`; deterministic `theta0`; 0 updates; selected by the predeclared theta0 tie rule |
| Local AFSE M/dimension/eigenvalue notes | 32/32; λ=1e-6; minimum Gram eigenvalue 0.0002263613; 0 negative eigenvalues clipped; TRAIN residual p99 0.5582050 |
| Network AFSE M/dimension/eigenvalue notes | 32/32; λ=1e-6; minimum Gram eigenvalue 0.0003971019; 0 negative eigenvalues clipped; TRAIN residual p99 0.7522633 |
| MLP convergence warnings | None; local/network iterations 60/47; terminal loss 0.0011908/0.0013336; fitted-sklearn versus persisted-NumPy maximum score error 0 |
| Local VALIDATION | n=32, support CHANGE/NORMAL 24/8; BA 0.8750 [0.7270, 0.9808], macro-F1 0.8454 [0.6744, 0.9646], coverage 0.9688; operational confusion columns CHANGE/NORMAL/ABSTAIN, matrix `[[20,3,1],[1,7,0]]` |
| Local TEST | n=32, support CHANGE/NORMAL 24/8; BA 0.5833 [0.4399, 0.7708], macro-F1 0.5897 [0.4074, 0.7949], coverage 1.0; recall CHANGE 0.9167, NORMAL 0.2500; confusion `[[22,2,0],[6,2,0]]` |
| Network VALIDATION | n=24, support 6/class; BA 0.9167 [0.8125, 1.0], macro-F1 0.9143 [0.7474, 1.0], coverage 0.9167; 2 heuristic-OOD abstentions |
| Network TEST | n=24, support 6/class; BA 0.7083 [0.5238, 0.8667], macro-F1 0.7009 [0.5051, 0.8509], coverage 0.9167; 2 uncertain; recall DEVICE/ENV/MIXED/NORMAL 0.8333/0.6667/1.0/0.3333; confusion columns DEVICE/ENV/MIXED/NORMAL/ABSTAIN, matrix `[[5,0,0,0,1],[0,4,0,2,0],[0,0,6,0,0],[0,4,0,1,1]]` |
| Raw baseline comparisons | Local TEST raw BA/F1/coverage 0.6875/0.6952/0.9063; network TEST raw 0.7500/0.7565/1.0 |
| Quantum→AFSE helped/tied/hurt | HELPED on VALIDATION for both tasks; HURT on TEST for both. Local TEST ΔBA -0.1042 [-0.2815, 0.0400], ΔF1 -0.1055 [-0.2912, 0.0589]. Network TEST ΔBA -0.0417 [-0.2750, 0.1609], ΔF1 -0.0556 [-0.2875, 0.1476] |
| Bootstrap/operational notes | 1,000 independent-episode replicates, no degenerate interval. Delta intervals include zero. NORMAL false-positive episode rate was 1.0 for both tasks; false-positive window rates were 0.4191 local and 0.3725 network |

The exact selected local theta vector is
`[-1.1322515836500304, 0.6049354826834435, -0.17738183188246315,
-0.7108717370199786, 0.12715886642685476, 0.6100628185718163,
-0.6325535184119011, -0.09283882026917967, -1.1669023233007738,
0.4866923756318499, -0.33534032294628247, 0.19642307714943072,
0.22669815086146883, 0.25001886893741027, 0.23555479904484355,
-0.7185928531479768]`.

The exact selected network theta0 vector is
`[0.03091583171290202, 0.6435929483820464, -0.30386974524518817,
-0.2064912691813241, 0.1248217931497525, 0.6653650667371245,
-0.22168729538374543, -0.0928388202692696, -0.4282288977232318,
0.4436856931191977, -0.34695130722395184, 0.34124211469903654,
0.23278896361304602, 0.21787369401425893, 0.2785735202085471,
-0.7185928531479769]`.

TEST results must remain unchanged even if unfavorable. No additional study,
seed, model threshold or hyperparameter search is authorised by a negative
finding.

## Final software validation

Record the exact final command, exit status, count and material warning.

| Check | Exact command | Result |
| --- | --- | --- |
| Container build | `make build` (also run by `make demo` and `make test`) | PASS; native `linux/arm64` backend and frontend images |
| Backend pytest | via `make test` | PASS; 369/369 in 75.49 s, one external Starlette/AnyIO deprecation warning |
| Protected original quantum tests | via backend suite | PASS; 8/8 |
| Ruff | via `make test` | PASS; all checks passed |
| Frontend TypeScript | via `make test` | PASS |
| Frontend ESLint | via `make test` | PASS, zero warnings allowed |
| Frontend Vitest | via `make test` | PASS; 47/47 across 14 files |
| Vite production build | via `make test` | PASS; 654 modules; non-blocking 794.58 kB chunk warning |
| Diff whitespace | `git diff --check` | PASS in final pre-commit validation |

## API and browser acceptance

`make acceptance` must write a JSON report outside Git under the configured
artifact root. The final run must cover:

| Scenario | Result/evidence |
| --- | --- |
| backend/frontend/lightweight quantum health | PASS; backend 8.608 ms, quantum 1.059 ms with NumPy reference not executed, frontend HTTP 200 in 2.931 ms |
| prepared registry and active compatible pair | PASS; canonical study/freeze/final plus stable application `aqse-demo-application-9bfb1b5789fbfefd` |
| N=1 local-only acquisition/inference | PASS; 1 result, `CHANGE_DETECTED`, p50/p95 10.53 ms, no skipped window |
| N=2 attribution-ambiguous acquisition/inference | PASS; 2 results (`CHANGE_DETECTED`, `NORMAL`), p50/p95 45.47 ms, no fabricated peer attribution |
| N=4 connected network acquisition/inference | PASS; 4 results (`NORMAL` + 3 `ENVIRONMENT_COMPATIBLE`), 40.43 ms, no skipped window |
| N=8 connected network acquisition/inference | PASS; 8 results (6 `NORMAL`, 1 `ENVIRONMENT_COMPATIBLE`, 1 `ABSTAIN`), 116.45 ms, no skipped window |
| common environmental event | PASS; one scheduled control, maximum absolute feature delta 49.128; changed features on S1–S4 |
| focal S3 drift/noise | PASS; two controls, maximum feature delta 19.235; focal/peer outputs remain observation-derived |
| moving/local spatial source | PASS; maximum feature delta 9.202 and spatially distinct S1–S4 response |
| shared offset and mixed/ambiguous condition | PASS; shared offset delta 43.304 and mixed case delta 40.226; displayed rule remained `MIXED_OBSERVABLE_CHANGE`, not physical-cause truth |
| dropout, clipping and stuck abstention/flags | PASS; S1 dropout, S2 stuck and S3 clipping each produced `ABSTAIN` with 3 valid peers; quality abstention count 3 |
| training start/cancel/duplicate intent and health responsiveness | PASS; initial 201, idempotent retry 200 with same job; 26 sensor frames advanced; health 0.679 ms; terminal state `cancelled`; no auto-apply |
| explicit atomic bundle application | PASS; identical retry retained the same application ID and compatible freeze |
| mismatched bundle rejection | PASS; HTTP 409 with actionable immutable-freeze mismatch message |
| reset/replay reproducibility and stale-response protection | PASS; 3,200 frames replayed with identical features/predictions; worker epoch 1→2; stale generation rejected |
| restart and saved-bundle reload without auto-training | PASS; external `make down && make demo` restored the identical application/freeze/local/network IDs; GUI registry showed both bundles active |

Automated acceptance report:
`/Users/florianocaprio/Projects/AQSE-artifacts/validation/demo-acceptance-20260910T003452486787Z.json`
(SHA-256 `63354dda37739c43f75a5c064d73e20afd997d30e5fa5f915efd318e423e02cd`).
It records `PASSED`, 182 HTTP calls in 12.114 s, an observation-only
prediction boundary and no simulator-truth endpoint call.

Browser acceptance must use the real app at both viewports and preserve
meaningful screenshots outside Git:

| Viewport | Result | Screenshot/report |
| --- | --- | --- |
| 1440×900 | PASS; session created/started, SSE connected, live State8 and network MLP output observed; all eight worksheets opened; console had no error/warning | `/Users/florianocaprio/.codex/visualizations/2026/09/08/01a08119-b668-78d0-9f58-ee7b276db53b/aqse-e2e/overview-live-desktop-1440x900.jpg` (SHA-256 `12fe4983e4f4cc8d24f33cd113cf0e622a7fde461300bf433d2172113fd73254`) |
| 390×844 | PASS after correcting a discovered Sensors overflow; every worksheet measured `scrollWidth=clientWidth=390`, navigation/key controls usable, console clean | `/Users/florianocaprio/.codex/visualizations/2026/09/08/01a08119-b668-78d0-9f58-ee7b276db53b/aqse-e2e/overview-live-mobile-390x844.jpg` and `sensors-live-mobile-390x844-fixed.jpg` (SHA-256 `e2a0d9e3931f1cfa5d8a0db792ca3dd9011b9c2e97f22d7f95ca6df9de18e5c3`, `54c3278904caad37a756765fb0b2ede867495a80cc051c3b8d32aa65b0d95d62`) |

## Eight-node real-wall-clock soak

Required command: `make soak` with the default 600-second duration. Accelerated
simulator time does not reduce the required wall time.

| Measure | Actual result |
| --- | --- |
| Environment / host | Darwin 25.6.0 arm64; Docker 29.5.3 `linux/aarch64`; both images native `linux/arm64` |
| Wall duration | requested 600.0 s; measured 600.2057 s |
| Node count / sampling / hop | 8 / 100 Hz / 1 s analysis hop, real time scale 1.0 |
| Backend/frontend health | Backend 120/120 polls, 0 failures, p95 6.376 ms, max 24.978 ms; both Compose services healthy before and after. Frontend was not polled continuously by the soak script |
| Analysis windows completed/skipped | 588 / 0; 0 quality abstentions in this run |
| Maximum queue depth/buffer behavior | newest-window queue max 0; sensor buffer bounded at 12,000/12,000; 48,001 old frames overwritten after 60,001 generated |
| Analysis processing p50/p95 | 60.988 / 83.059 ms |
| Result-age observations | final sampled result age 927.618 ms |
| RSS start/end/peak/trend | 793,260,032 / 1,121,890,304 / 1,121,890,304 bytes; +328,630,272 bytes; ten-minute OLS slope 1,664,194,151.6 bytes/hour. This warm-up/cache-inclusive observation is a limitation, not a leak diagnosis or forecast |
| Report path | `/Users/florianocaprio/Projects/AQSE-artifacts/validation/demo-soak-20260910T005948702591Z.json`; SHA-256 `5271d0c2b3197d0f4233383f58ee2da14de97a562ba0185b350f5a65602b9d1a` |

The final interpretation must be “measured on this local run”, not a hard
real-time or general stability guarantee.

## Protected identities and repository integrity

Expected hashes to remeasure at the end:

| Asset | Expected SHA-256 | Actual |
| --- | --- | --- |
| `tqk8.py` | `cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689` | exact match |
| `sampler_qng.py` | `7489c5c2b3eb0783499c333dc0826994b146d7cf5631469a0520bac9570b2ea6` | exact match |
| original TQK8 test | `c87c56c0eb3f1ea20d2dd124427aea96f868ae4674b5b390e2688e9da8bb0c09` | exact match |
| original notebook | `9d7e0337af7c039894fc6c53567de2883063b32c89e368753bd01e1d9e9e82be` | exact match |
| historical TEST ledger | `e0d4898141d8be07f4a4f1af7582791eae61994ced52bb7b3dba30a7ddda4b70` | exact opaque match; no semantic read |

Final repository checks:

| Check | Actual |
| --- | --- |
| branch | `codex/milestone-1d-tqk-training` |
| validated implementation SHA before this evidence commit | `a680752c0d45032afcdd669dad9a38b5da6fa723` |
| final local/remote SHA | Reported after the evidence commit and push; a commit cannot contain its own SHA |
| `main` unchanged | PASS; local and origin remain `1cb07cdccabf9c655a63f2b23aab37383ae90b69` |
| `milestone-1c` tag unchanged | PASS; peeled tag remains `95c5483c0192ba7605713c428e02b2527ab1f919` |
| secrets/runtime/build/ZIP files excluded | PASS in the final tracked/staged-file audit; artifacts and screenshots are outside Git |
| final `git status` clean | Verified and reported after the final push |

## Known warnings and corrected incidents

- The first preparation invocation used a script path that could not import
  `app`; it failed before creating a study or ledger event. The Make target now
  runs the module form, and both the canonical run and idempotent rerun passed.
- The first acceptance attempt received HTTP 403 from Vite for the internal
  Compose hostname. Vite now allowlists only the `frontend` service alias;
  acceptance then passed.
- Browser QA exposed a 490 px Sensors document width at a 390 px viewport.
  The descriptive-analysis grid now stacks below 560 px; all eight worksheet
  widths remeasured at exactly 390 px.
- Pytest emits one Starlette use of a deprecated AnyIO alias. Vite emits one
  non-blocking 794.58 kB chunk-size warning.
- A crash after fail-closed TEST-ledger advancement but before final artifact
  persistence would require manual recovery and must never trigger an
  automatic TEST retry.
- The ten-minute RSS observation grew by 328.6 MB. It remains a documented
  signal for longer profiling, not proof of a leak and not grounds for a
  post-TEST model change.

## Scientific conclusion

1. **Functional research demonstrator:** complete for the bounded local v1
   scope. The real observation-only path, explicit training/cancellation,
   atomic bundle application, replay, restart, desktop/mobile GUI and 600 s
   eight-node run were exercised. This is not field-deployment validation or a
   hard real-time guarantee.
2. **Predictive performance / quantum benefit:** the frozen quantum→AFSE model
   helped both tasks on VALIDATION but hurt both on the single held-out TEST
   comparison. NORMAL recall and replay false positives are material limits.
   Delta intervals include zero, the study is small and simulated, and no
   quantum advantage is demonstrated. No tuning, selection change, refit or
   additional TEST access followed these results.

The branch is ready for Floriano's final manual acceptance and merge review;
no pull request or merge is part of this delivery.
