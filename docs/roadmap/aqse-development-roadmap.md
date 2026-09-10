# AQSE Development Roadmap

## Document status

This is the canonical implementation roadmap for the AQSE local research
demonstrator. It separates:

1. **decisions** authorised by the completion mandate;
2. **implemented code** present in the branch;
3. **measured evidence** recorded only after a command or experiment runs.

The end-to-end completion mandate supersedes the earlier instruction to stop
between 1D.5, 1E, 1F and 1G for the explicitly listed v1 scope. It does not
authorise a merge, scientific-source redesign, historical TEST reopening, or
claims unsupported by measured evidence.

Repository context:

- working branch: `codex/milestone-1d-tqk-training`;
- completion-mandate entry HEAD:
  `67f1de10fcd09a32bafab48d600c05d8f82a0021`;
- immutable `main` baseline:
  `1cb07cdccabf9c655a63f2b23aab37383ae90b69`;
- immutable peeled `milestone-1c` tag:
  `95c5483c0192ba7605713c428e02b2527ab1f919`;
- last pushed implementation SHA before the final evidence commit:
  `a680752c0d45032afcdd669dad9a38b5da6fa723`; the final self-referential SHA
  is reported after push in the delivery response.

## Current completion snapshot

| Area | Authorised decision | Code state | Measured state |
| --- | --- | --- | --- |
| Historical 1D | Preserve published 1D.1–1D.4b evidence; never reopen its TEST | Existing loaders/results retained | Historical results remain the published record |
| State8 profiles | Add distinct local/network non-harmonic profiles | Implemented in the working branch | Component/integration evidence belongs in the end-to-end validation record |
| New study | 160 independent episodes, fixed 96/32/32 split | Generator, typed archive and sealed ledger implemented | Canonical study `aqse-network-study-5f33f5c4d856361f`; TEST ledger closed at 3 events |
| Quantum training | Compare theta0 with at most 10 accepted protected-QNG updates | Protected wrappers and bounded fitting integrated | Local selected protected QNG after 10 steps; network selected theta0 by tie rule |
| AFSE | Fixed regularised Nyström map, up to 32 TRAIN landmarks | Fitted artifact and immutable query runtime implemented | Both selected maps are 32D/32 landmarks; 0 negative eigenvalues clipped |
| Classical output | Standardised 32→16 tanh MLP, fixed LBFGS budget | sklearn fit plus safe numeric NumPy runtime implemented | NumPy agreement exact; AFSE helped VALIDATION but hurt both TEST comparisons |
| Bundle registry | One compatible local/network pair, explicit application | Atomic persisted registry/application implemented | Idempotent apply, mismatch rejection and identical reload after Docker restart passed |
| Continuous analysis | Observation-only, causal, bounded newest-window scheduling | Worker and status/result API implemented | 600.206 s at 8 nodes: 588/0 windows, queue max 0, health 120/120 |
| GUI | Retain and complete eight worksheets | Connected views and controls implemented in the working branch | Real 1440×900 and 390×844 QA passed; all worksheets and live path exercised |

“Implemented” above describes executable code, not scientific performance.
Only [the validation record](../validation/end-to-end-demo.md) may declare an
experiment or acceptance check passed.

## Dependency-ordered delivery

### 1. Historical evidence preservation

Milestones 1A–1D.4b remain regression baselines:

- author-supplied VQC/TQK, loss, numerical derivatives, Fubini–Study metric and
  QNG sources stay byte-identical;
- historical dataset/result artifacts remain outside Git;
- the historical TEST ledger is verified opaquely against
  `e0d4898141d8be07f4a4f1af7582791eae61994ced52bb7b3dba30a7ddda4b70`;
- no new AFSE, neural or network-task choice is justified with the consumed
  historical TEST.

### 2. Observable State8 prerequisite

Two incompatible feature profiles serve the real live task:

- `aqse.local-state8.v1`;
- `aqse.network-state8.v1`.

Both use 100 Hz measurements, an explicit observed reference over the first
8 s, 4 s causal windows and a 1 s hop. They retain missing observations,
per-feature validity and quality flags. The local profile supports N=1 and
fallback operation. The network profile requires at least three comparable
aligned nodes and uses a leave-one-out peer median. N=2 remains explicitly
ambiguous.

The historical harmonic profile and `aqse.tqk8.encoding.phase-direct.v1`
remain separate compatibility domains. State8 applies the protected
`AngleScaler` to all eight real-valued coordinates under
`aqse.state8.angle-scaler-all8.v1`.

### 3. Bounded network demonstration study

The frozen study is `aqse-network-demo-v1`:

- four scenario strata: `NORMAL`, `ENVIRONMENT_COMPATIBLE`,
  `DEVICE_COMPATIBLE`, `MIXED_OR_AMBIGUOUS`;
- node counts 1 through 8;
- five independent episodes per scenario/node-count cell;
- three TRAIN, one VALIDATION and one TEST per cell, assigned before
  simulation: 96/32/32;
- 28 s at 100 Hz; reference `[0,8)`; one predeclared focal-node example at
  `[18,22)`;
- separate physical observation and label files;
- episode/lineage grouping prevents peer nodes, replays or correlated variants
  from crossing partitions;
- fixed seeds 2001001–2001006 and canonical generation separator
  `aqse-network-demo-v1/canonical-generation/v3`;
- the environmental dipole and focal thermal ramp are scheduled physical
  interventions on `[14,24)`, with pre-onset reference/replay remaining
  unperturbed by those sources.

No hidden intervention, scenario name, random seed, fault truth or future
sample enters predictive input.

### 4. Frozen representation and downstream model

Each local/network task compares only:

- deterministic `theta0`;
- one candidate from at most ten accepted calls to the unchanged protected
  QNG wrapper.

The binary protected alignment loss receives only compatible -1/+1 targets.
The network QNG subtask excludes NORMAL and MIXED/AMBIGUOUS examples; the
four-class downstream model may use every eligible TRAIN class.

For each candidate, the branch fits:

1. a TRAIN-only State8 encoder;
2. `aqse.afse.nystrom-ridge32.v1` with distinct balanced TRAIN landmarks,
   ridge `1e-6`, immutable ordered basis and TRAIN-residual p99 heuristic OOD
   gate;
3. `aqse.classical.mlp-32x16-tanh-lbfgs.v1`, with TRAIN-only
   standardisation, hidden layers 32/16, tanh, LBFGS, alpha `1e-3`,
   `max_iter=500`, `max_fun=15000`, seed 2001005;
4. the same compact MLP on the same raw State8 inputs as a classical baseline.

Model selection is VALIDATION balanced accuracy, then macro-F1, then theta0 on
a tie. The selected theta, encoder, AFSE, output scaler and classifier form one
immutable bundle. TEST cannot change the selection.

### 5. Continuous operation and controlled promotion

The live path consumes `ObservationFrame` only, freezes one session reference,
extracts the newest complete causal window, executes the compatible bundle and
publishes:

- profile, bundle, theta, reference and application identities;
- sensor/peer/context identities and window interval;
- feature values, units, validity and quality flags;
- encoded angles and fixed AFSE coordinates when eligible;
- reconstruction residual and heuristic OOD flag;
- uncalibrated class scores, displayed class and raw-feature baseline;
- processing time, result age, p50/p95 latency, skipped windows and bounded
  queue depth.

The worker never trains. During explicit training it yields the shared heavy
slot, leaves acquisition running and reports a paused/busy state rather than
building a backlog.

Training jobs use durable intent identity, one bounded worker, real accepted
step history and cancellation. Completion creates saved candidates but does
not promote them. The user applies a selection-freeze-compatible local/network
pair explicitly and atomically; caches and pending results are invalidated
together.

### 6. Workbench and delivery

The existing eight worksheets now have the following target state:

| Worksheet | End-to-end responsibility |
| --- | --- |
| Overview | service/session/analysis state, active bundle, live quality and age |
| Sensors | 1–8 node configuration, lifecycle, observed plots and explicit live events |
| Features | live State8 values/reference/context plus separate legacy harmonic workflow |
| Quantum Engine | protected circuit metadata, frozen live theta and isolated manual preview |
| QNG Training | bounded Train/Cancel/status/history and explicit Apply |
| Local Embedding / AFSE | fitted method, dimension, residual/OOD and live z(x) |
| Neural Model | architecture, real scores, uncertainty gates and raw baseline |
| Experiments | read-only study/results/registry, exports and compatible application |

Final acceptance must exercise real backend state at 1440×900 and 390×844,
restart persistence, invalid bundle rejection and a ten-minute eight-node soak.
Screenshots and reports remain outside Git.

## Delivery gates

| Gate | Condition | Current state |
| --- | --- | --- |
| E2E-1 contract | design freeze recorded before canonical study evaluation | Recorded |
| E2E-2 implementation | State8, study, AFSE, MLP, bundles, worker, APIs and GUI connected | Closed; backend 369/369, frontend 47/47, static checks/build pass |
| E2E-3 scientific run | TRAIN fit, VALIDATION freeze, exactly one new TEST evaluation persisted | Closed; immutable canonical IDs and 3-event ledger recorded |
| E2E-4 operational | services healthy, browser routes pass, persistence/replay/application verified | Closed for automated/local evidence; acceptance report passed and restart preserved active pair |
| E2E-5 soak | eight nodes, 600 s real wall time, bounded queue/memory/latency report | Closed for this local observation; RSS growth and non-real-time scope disclosed |
| E2E-6 review | branch pushed clean, local/remote aligned, external manual acceptance | Automated evidence complete; external manual acceptance/merge review remains |

No gate requires the quantum/AFSE model to beat the classical baseline.
Negative, tied or weak predictive results close the bounded experiment when
reported honestly; they do not authorise an undeclared tuning loop.

## Completed automated acceptance evidence

The detailed validation record now contains actual evidence for:

- complete backend/frontend test, lint, typecheck and production build suites;
- protected-source hashes and opaque historical TEST-ledger identity;
- `make build`, canonical/idempotent `make prepare-demo`, `make demo`,
  `make acceptance` and `make soak`;
- frozen classification metrics, supports, confusions, coverage, uncertainty
  and paired raw-baseline comparisons;
- 1/2/4/8-node operation, controlled perturbations, quality abstention,
  training cancellation/idempotency and atomic compatible application;
- deterministic replay/stale-generation rejection and saved-bundle reload
  after a fresh backend/frontend restart;
- meaningful desktop and narrow browser screenshots outside Git;
- a 600 s real-wall-clock eight-node report with RSS, queue, skips, health and
  latency observations;
- final whitespace, tracked-file, ref/hash and local/remote checks.

Only Floriano's manual acceptance and any separately authorised merge review
remain. The negative TEST comparison does not reopen selection or tuning.

## Compatibility and scientific protection

A deployable bundle binds task, profile/fingerprint, units/reference policy,
TRAIN-fitted encoder, exact protected-source identities, theta, TQK semantics,
AFSE landmarks/B/ridge, MLP parameters/class order and measured provenance.
Mismatched parts are rejected; query acquisition identity never masquerades as
the fitting dataset identity.

The following remain outside the completion mandate:

- changing the author's circuit, kernel, alignment loss or QNG mathematics;
- QPU, shots/noise-model research or entangled sensor networks;
- hardware/calibrated sensitivity claims;
- inverse dipole localization or geographic recognition;
- automatic labels, model self-training, continual learning, AFSE refits or
  hidden promotion;
- cloud services, database, queue, API key or credential requirements.

## Git and review workflow

All completion work remains on `codex/milestone-1d-tqk-training`. Validated
increments may be committed and normally pushed. Do not rebase, amend
published history, force-push, move tags, open a pull request, merge into
`main`, or create a release without Floriano's later explicit instruction.

Generated studies, bundles, ledgers, reports, screenshots, `.env`, caches,
logs, ZIP files, `node_modules` and build outputs stay outside Git.

## Related documents

- [Final delivery status](../final-delivery.md)
- [Operator guide](../user-guide.md)
- [Detailed end-to-end validation](../validation/end-to-end-demo.md)
- [Canonical AQSE pipeline](../architecture/canonical-aqse-pipeline.md)
- [Scientific scope and limitations](../architecture/scientific-scope-and-limitations.md)
- [Feature profiles and quality](../features/feature-profiles-and-quality.md)
- [AFSE boundary](../quantum/afse-boundary.md)
- [Historical 1D.4b evaluation](../validation/milestone-1d-4b-held-out-evaluation.md)
