# AQSE Milestone 1D.0 — Design and Compatibility Audit

## Record identity

| Field | Value |
| --- | --- |
| Audit date | 2026-09-09 |
| Scope | Scientific design, compatibility audit, baseline validation and isolated probes only |
| Branch | `codex/milestone-1d-tqk-training` |
| Branch start / current `main` | `1cb07cdccabf9c655a63f2b23aab37383ae90b69` |
| `origin/main` at branch creation | `1cb07cdccabf9c655a63f2b23aab37383ae90b69` |
| Milestone 1C tag | `95c5483c0192ba7605713c428e02b2527ab1f919` |
| Scientific decision | Pending Floriano's explicit approval |

Validation environment: macOS `arm64`; native Docker containers reported Linux
`aarch64`, Python `3.12.14`, Node.js `v22.23.2` and pnpm `10.6.5`. No
`linux/amd64` override was used.

The master specification named `95c5483` as the expected `main` baseline. The
repository audit found that both local and remote `main` had intentionally
advanced to `1cb07cd` through the already approved health/diagnostics
reliability patch, while the milestone tag correctly remained on `95c5483`.
The dedicated 1D branch was therefore created from the clean current
`origin/main`. No reset, rebase, merge, tag move or modification of `main` was
performed.

## 1. Audit scope and method

The audit read the current architecture, feature, simulator, quantum-preview,
workbench, Docker and validation contracts and inspected their implementations.
It then ran fixed-seed probes in the isolated `backend/research_audit` package.
Those probes import existing code; they do not expose an API, change production
configuration, persist a dataset, activate an encoder or train/promote a model.

The concise report can be reproduced in the development container with:

```sh
docker compose run --rm backend python -m research_audit
```

Tests are in `backend/tests/research_audit/`. Thresholds cover deterministic
mathematical properties or generous local resource bounds; no test requires a
scientific performance win.

Sources inspected included:

- `docs/architecture/canonical-aqse-pipeline.md` and
  `scientific-scope-and-limitations.md`;
- `docs/features/feature-profiles-and-quality.md`;
- `docs/quantum/quantum-preview-contract.md` and `afse-boundary.md`;
- feature models, extractor, windowing and provenance implementations;
- network models, simulator, session and observation/truth contracts;
- quantum adapter, health/diagnostics, preview and protected user pipeline;
- original quantum tests, notebook and validation artifacts;
- workbook controller, state/invalidation, QNG/AFSE/Neural worksheets and
  frontend request types;
- `Makefile`, dependency manifests, Dockerfiles, Compose configuration and the
  Milestone 1C validation record.

## 2. Implementation inventory

| Area | Observed implementation | 1D interpretation |
| --- | --- | --- |
| Sensor simulation | Scalar legacy simulator plus bounded 1–8-node vector simulator/session, deterministic streams, SI-unit internals and separate observation/truth DTOs | Implemented baseline; physics/order must remain unchanged |
| Current features | Eight-value harmonic profile: amplitude, phase, frequency, variance, drift, SNR, PSD peak magnitude and temperature | Implemented; no DC mean, spatial residual, coherence or network correlation |
| Feature quality | Fixed schema thresholds, complete causal windows, clipping/quality ledger | Implemented; not silently tuned for training |
| Live feature integrity | Process-local HMAC over the full profile and feature record | Implemented for one backend process; not durable provenance |
| Quantum preview | Bounded self-reference or reference/query fixed-theta fidelity kernel, maximum 128 total windows | Implemented exploratory path; not training or held-out evaluation |
| Input scaling | Supplied AngleScaler fit on the declared preview reference rows, applied to all eight columns | Implemented legacy policy; phase circularity is not represented explicitly |
| TQK8 scientific engine | 8 qubits, 8 inputs, 16 theta, 7 CZ gates, RY upload plus RZ re-upload, exact Qiskit/NumPy engines | Implemented and protected |
| Loss and QNG | Centered-alignment loss, exact derivatives, empirical mean Fubini–Study metric, damping/clipping/Armijo QNG | Supplied scientific capability exists; no sensor-training application path |
| Application adapter | State, Gram and cross-Gram calls plus metadata | Preview-ready; it does not expose the differential interface required by `fit_qng` |
| Workbench QNG page | Architecture/status surface only | Correctly not connected |
| AFSE and neural pages | Explicit pending boundaries | No mathematics/model implemented or authorized |
| Experiment history | Bounded browser-session ledger with immutable request snapshots | Useful provenance display, not a durable scientific archive |

### Confirmed capability boundaries

- The preview never calls `alignment_loss`, `loss_grad_metric` or `fit_qng`.
- No production API route imports or exposes `fit_qng`; its only application
  execution in 1D.0 is the isolated audit package.
- The frontend currently supplies its transient network `session_id` as
  `reference_dataset_id`; this is not a durable dataset identity.
- `FeatureProfile.profile_id` is constant for the harmonic family even though
  channel, window geometry and sampling fields vary. The complete profile is
  signed and returned, but compatibility must not rely on `profile_id` alone.
- `WindowFeatureRecord` does not contain the session/configuration schema,
  calibration version, raw-observation digest, raw interval identity or
  generative lineage needed for grouped training.
- Multiple nodes can generate separate windows, but the present feature vector
  itself contains no inter-node relationship.
- The candidate 4-second/1-second-hop continuous-network relational profile is
  documentation only and was not implemented by this work.
- Preview and explicit quantum diagnostics currently have separate
  single-operation guards. A future training service must use coordinated
  admission or an isolated worker so heavy quantum tasks cannot overlap and
  starve the simulator/API.
- One frontend import-test fixture gives `spectral_peak` the unit `Hz` although
  the backend contract requires `nT²/Hz`, and the TypeScript feature arrays are
  permissive strings. The backend plus a canonical full-profile fingerprint
  must remain the compatibility authority; this audit does not alter the
  unrelated fixture.
- The historical TQK8 report records the environment in which Qiskit was then
  unavailable. It remains valid historical evidence and must not be rewritten
  now that the current pinned container includes Qiskit.

These are design gaps for later increments, not defects repaired during 1D.0.

## 3. Protected scientific assets

The following SHA-256 values were recorded before the 1D.0 additions and are
the required values after validation:

| Protected file | SHA-256 |
| --- | --- |
| `backend/app/quantum/user_pipeline/tqk8.py` | `cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689` |
| `backend/app/quantum/user_pipeline/sampler_qng.py` | `7489c5c2b3eb0783499c333dc0826994b146d7cf5631469a0520bac9570b2ea6` |
| `backend/app/quantum/user_pipeline/__init__.py` | `5219f5e58e4d7e9012d15f5ecec79e9a126c03ba7c2a6a0c7d5e261ef718759b` |
| `backend/tests/quantum/conftest.py` | `23734af624fba6ce158d402dc288e9225ffe84ab469cf020eee4a722ba4992be` |
| `backend/tests/quantum/test_tqk8.py` | `c87c56c0eb3f1ea20d2dd124427aea96f868ae4674b5b390e2688e9da8bb0c09` |
| `docs/notebooks/TQK8_walkthrough.ipynb` | `9d7e0337af7c039894fc6c53567de2883063b32c89e368753bd01e1d9e9e82be` |
| `docs/quantum/README_TQK8.md` | `5bdaab2d542fe3ee31a4fda69a4612b03fec5909bd0f5482f919524f9cb2d890` |
| `docs/validation/tqk8/history.json` | `d79a59cea578734e91356ad84f0e7db4233df5321f5fcf3184bc412b83e56ac8` |
| `docs/validation/tqk8/metrics.json` | `805321cc6ee7cdd549204f715e82357bcd5ab3039e1fc6870bca0aa3442b2009` |
| `docs/validation/tqk8/tests.txt` | `69f91cfbf7b91a7e9882a8d5e40cb625e0b33c503a354adcefecb33941c6740c` |
| `docs/validation/tqk8/trained_tqk8.npz` | `055c626a1262d3f6642605c9ea1e843708fa836a9df740a4209d47197e9ec29e` |
| `docs/validation/tqk8/validation_report.json` | `0d58da015c8410efb9e128a748139eda2a9b7cab19410fc5a43e7d951682cf5c` |

The protected implementation and original tests were not edited. A final hash
and Git-diff check is part of the completion gate.

## 4. Diagnostic evidence

### 4.1 Legacy AngleScaler at the phase wrap

**Hypothesis.** Two physically nearby phases on opposite sides of the canonical
cut can be mapped far apart because the legacy scaler treats phase as an
ordinary linear variable.

**Inputs.** Seed `104729`; eight training rows; phase training values placed
symmetrically near `-pi` and `+pi`; query phases `-pi + 0.001` and
`+pi - 0.001`. The other seven columns are fixed-seed finite random values.

**Observed result.** Circular separation was
`0.0020000000000000248 rad`. The legacy encoded phases were
`-0.7261870141709146` and `+0.7261870141709146 rad`, a linear separation of
`1.4523740283418292 rad`.

**Interpretation.** The legacy policy is deterministic and remains valid for
its preview contract, but it does not preserve circular neighbourhoods at this
cut. This supports evaluating, not yet activating, the direct wrapped-phase
policy proposed in the scientific plan.

### 4.2 Actual VQC direct-phase periodicity and seam continuity

**Hypothesis.** When the phase feature is supplied directly as circuit input,
adding `2*pi` preserves the physical state density and fidelity, and states on
the two sides of the phase cut converge as their circular separation tends to
zero.

**Inputs.** Seed `104729`; finite fixed-seed feature and anchor vectors; random
non-zero theta with L2 norm `2.1063429792622217`; direct phase `0.731` versus
`0.731 + 2*pi`; seam inputs `-pi + 1e-4` and `+pi - 1e-4`. Both exact engines
were executed.

| Engine | Max density error, `+2*pi` | Anchor-kernel error, `+2*pi` | Max seam density delta | Seam anchor-kernel delta |
| --- | ---: | ---: | ---: | ---: |
| NumPy | `8.8861e-17` | `2.2204e-16` | `2.0378884e-05` | `3.2228855e-05` |
| Qiskit | `7.7579e-17` | `5.5511e-17` | `2.0378884e-05` | `3.2228855e-05` |

**Interpretation.** Within declared numerical tolerance, the actual circuit is
periodic in a directly supplied phase and the two exact implementations agree.
The finite seam delta is consistent with comparing inputs separated by
`2e-4 rad`; coordinates jump but density/kernel remain continuous. This does
not solve differences in window-start time or clock synchronization.

### 4.3 Constant-field-offset information loss

**Hypothesis.** The current harmonic profile for a linear measured channel
does not preserve a constant field offset because its regression contains an
unreported intercept and the remaining statistics are offset-invariant.

**Inputs.** Seed `240517`; x channel; 400 samples at 100 Hz; four seconds of
`25 nT + 8 nT * sin(2*pi*5 Hz*t + 0.37) + 0.15 nT/s*t` plus shared Gaussian
noise with `0.05 nT` standard deviation. The paired observation adds exactly
`375 nT` to every x sample.

| Feature | Baseline | With `+375 nT` |
| --- | ---: | ---: |
| amplitude | `8.003125457651361` | `8.003125457651365` |
| phase | `0.36879888037873537` | `0.36879888037873737` |
| frequency | `5.000055913366462` | `5.000055913366462` |
| variance | `31.9818967476638` | `31.981896747663804` |
| drift | `0.1520682920105414` | `0.1520682920105283` |
| snr | `41.19108852799183` | `41.19108852799243` |
| spectral_peak | `85.25145017285477` | `85.2514501728548` |
| temperature | `293.1500000000001` | `293.1500000000001` |

Maximum absolute feature difference: `5.968558980384842e-13`.

**Interpretation.** The difference is floating-point noise. The representation
cannot support claims about recovering DC offset. Adding that information
would require a separately approved feature-profile version and retraining.

### 4.4 Observation-equivalent physical and instrumental causes

**Hypothesis.** With one identity-oriented, ideal-response node, a uniform
world-field event and an equal node-bias event can yield identical observable
vectors despite different truth causes.

**Inputs.** Seed `240523`; one vector node; 400 samples at 100 Hz; identical
harmonic background; either a four-second `world_field_offset` or a four-second
`node_bias` of `(7e-9, -2e-9, 3e-9) T`.

**Observed result.** Maximum observation difference was exactly `0 T`; maximum
eight-feature difference was exactly `0`. The physical truth channel recorded
the event field `(7e-9, -2e-9, 3e-9) T` and cause `world_field_offset`; the
instrumental case recorded zero environmental event field and cause
`node_bias`.

**Interpretation.** The two truths are not identifiable from these one-node
observations. Identical predictive inputs must remain identical throughout the
encoder, kernel and evaluator. This is a required negative control, not a
classification target that AQSE should be forced to solve.

### 4.5 One toy QNG-iteration resource budget

**Hypothesis.** A minimal, explicitly bounded exact-state step can run locally,
while state-evaluation count and memory formulas expose why representative
training sizes must be benchmarked before service limits are frozen.

**Inputs.** Seed `240529`; four finite encoded rows; labels
`[-1,+1,-1,+1]`; non-zero theta; unchanged `fit_qng`; NumPy exact-state engine;
one requested step, learning rate `0.2`, damping `1e-3`, max step `0.4`.

**Observed result.** One update was accepted. The engine counted `136` exact
state evaluations against a worst-case one-step cap of `180` with twelve line
search trials. Theoretical persistent core arrays occupied `280704 bytes`;
Python `tracemalloc` peak was `359576 bytes`; elapsed wall time in the Docker
run was `0.13279883399809478 s`.

For `N` training rows, the probe's lower-level core-array estimate is:

```text
states + derivatives + kernel + metric
= 69,632*N + 8*N^2 + 2,048 bytes
```

This estimates approximately `2,238,464 bytes` at `N=32` and
`9,046,016 bytes` at `N=128`. A full derivative pass uses `33*N` state
evaluations; every backtracking kernel adds another `N`, up to a theoretical
`45*N` in the current twelve-trial line search.

**Interpretation and limit.** Only `N=4` and the NumPy engine were timed. The
formula excludes Python service state, Qiskit circuit/binding objects,
temporary matrix products, allocator/RSS behavior, datasets and concurrent
simulation. `tracemalloc` is not a process-RSS measurement. The 32/128 row
limits and any timeout therefore remain engineering proposals; representative
Qiskit and NumPy benchmarks are required before 1D.3.

## 5. Validation record

### 5.1 Existing baseline separated from audit probes

The application baseline was run with the audit directory explicitly excluded:

```sh
docker compose run --rm \
  -v ./docker-compose.yml:/workspace/docker-compose.yml:ro \
  backend python -m pytest --ignore=tests/research_audit
```

Result: **145 passed**, one upstream Starlette/AnyIO deprecation warning.

An earlier manual baseline-only attempt omitted the read-only Compose-file
mount required by `test_docker_healthcheck.py`; it collected 145 tests and
reported `144 passed, 1 failed` because neither expected Compose path existed
inside that one-off container. Repeating with the repository's declared mount
passed all 145. The failed invocation is retained here as environmental audit
history rather than misreported as a product defect.

### 5.2 Combined gate

`make test` rebuilt the images and executed the application plus new probes:

- backend: **150 passed** = 145 baseline + 5 isolated audit tests;
- original protected quantum regression: **8 passed** in its separate run;
- Ruff: pass;
- frontend typecheck: pass;
- ESLint: pass with zero warnings;
- frontend: **29 passed in 9 files**;
- production build: pass, 651 modules transformed.

The isolated audit-only gate also passed independently: Ruff, Ruff format and
**5/5 tests** (`5 passed in 0.62 s`). Both long-running AQSE services were
separately observed in Docker `healthy` state.

The production build emitted the existing non-blocking Vite chunk-size warning:
the single JavaScript bundle was `719.40 kB` before gzip and `210.04 kB` after
gzip. Pytest emitted the same single upstream deprecation warning.

A final clean-tree validation, protected-hash comparison, `git diff --check`
and remote-SHA verification are required immediately before publication.

## 6. Requirement-to-evidence and future-test matrix

| Requirement | 1D.0 evidence | Later required test/gate |
| --- | --- | --- |
| Scientific source unchanged | Hash manifest and original 8 quantum tests pass | Recheck at every increment |
| Legacy phase behavior known | Fixed-seed seam probe | Preserve legacy preview regression |
| Candidate phase is physically periodic | Qiskit/NumPy direct-input probe | Approve policy; train-only wrapper and compatibility tests |
| DC information limit explicit | Paired `+375 nT` observation probe | Reject claims/tasks requiring absent mean field |
| Identifiability limit explicit | World-offset/node-bias negative control | Identical-input inference invariant across truth labels |
| Initial compute bound measured | N=4 one-step count/time/memory probe | Representative N=32/128, both engines, service responsiveness |
| Dataset groups are independent | Contract only | Episode/lineage/replay/overlap split tests in 1D.1 |
| Fitted transforms use training only | Contract only | Perturb held-out rows; fitted artifacts unchanged |
| Wrapper preserves QNG | Design only | Repeated `steps=1` vs `steps=N` equivalence |
| Test set remains sealed | Design only | Immutable split and append-only opening ledger |
| Checkpoints are compatible and safe | Design only | Digest, no-pickle reload, mismatch rejection and restart tests |
| UI remains honest/causal | Existing placeholder only | Explicit run/cancel/apply, stale-response and invalidation tests |
| AFSE/neural boundaries preserved | Existing pages/docs inspected | No output until separately approved mathematics/model exists |

## 7. Audit conclusions and approval gate

1. The proposed nominal/elevated device-noise task is compatible with measured
   harmonic features, but it is expected to have a strong simple SNR baseline.
2. The current feature representation demonstrably discards constant offsets
   and contains no network evidence; claims must stay within that information.
3. Legacy AngleScaler phase semantics are not circular at the wrap. The real
   VQC itself supports a direct periodic phase candidate, but its activation
   requires a new versioned encoding policy and explicit approval.
4. Distinct causes can be observation-equivalent. Scientific validation must
   preserve this negative control rather than optimize against hidden truth.
5. Existing live provenance and identifiers are insufficient for durable,
   independent datasets. 1D.1 must add immutable observations, separate labels,
   generative lineage, grouped splits, a canonical full-profile fingerprint,
   immutable dataset/run manifests and a separate append-only sealed-test
   access ledger.
6. The toy QNG step is locally feasible, but it is not a representative
   N=32/128 service benchmark and does not justify a final timeout.
7. A future wrapper can reuse the protected optimizer, but adapter capability,
   coordinated quantum admission, cancellation and checkpoint compatibility
   must be designed without editing the scientific source.
8. The recommended first dataset uses one fixed noise regime for each complete
   episode. A `NODE_NOISE_BURST` alternative needs an explicit policy for
   transition and overlapping windows before it can be approved.

Milestone 1D.0 should stop after publishing these documents and audit probes.
Proceeding to 1D.1–1D.6 requires Floriano's explicit scientific approval of the
open decisions in the training plan.

Related 1D.0 documents:

- [AQSE development roadmap](../roadmap/aqse-development-roadmap.md)
- [Milestone 1D scientific training plan](../training/milestone-1d-scientific-plan.md)
