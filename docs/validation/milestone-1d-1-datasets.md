# AQSE Milestone 1D.1 — Immutable Controlled Datasets

## Status and boundary

Milestone 1D.1 is implemented and locally validated on
`codex/milestone-1d-tqk-training`; external review is pending. This increment
creates offline dataset infrastructure only. It does not implement or expose a
production encoder, QNG training, model selection, test evaluation, AFSE, a
neural model, or continuous inference.

The protected author-supplied TQK8 circuit, kernel, loss and QNG sources and
their original tests were not modified.

## Approved decisions

- Binary episode label: `-1` nominal device white noise and `+1` elevated
  device white noise.
- One static vector magnetometer, X-channel features, identity orientation and
  response matrices.
- 100 Hz, 10 seconds and 1000 samples at `t[k] = k / 100`.
- Uniform field `[20000, 0, 45000] nT`; X harmonic amplitude `[6, 10] nT`,
  frequency `[4, 8] Hz` and phase `[-pi, pi)`.
- Constant episode temperature `[292.15, 294.15] K`; no thermal bias.
- Nominal RMS `[0.25, 0.75] nT` and elevated RMS `[1.5, 3.0] nT`, independently
  sampled per episode and applied per axis before the disabled bandwidth filter.
- No dipole, gradient, OU process, random walk, drift, dropout, stuck fault or
  clock error. Saturation is prohibited and verified.
- Existing harmonic feature profile v1: one-second complete causal windows,
  50% overlap and eight ordered observable features.
- Pilot first: 12 independent episodes per class; at least 80% retained windows
  per class and no episode with zero valid windows.
- Development set: 60 independent episodes per class, split by lineage exactly
  36/12/12 per class into train/validation/test. Pilot lineages are excluded.
- The episode/lineage is the primary statistical unit. Windows never determine
  the split.

## Implemented controls

The offline package under `backend/app/training/` provides:

- deterministic pre-generation episode plans with independent nuisance,
  intervention and measurement RNG streams;
- an independent label artifact and an origin policy that never enters the
  eight predictor columns;
- canonical JSON plus C-order, explicitly little-endian numeric identities;
- separate scientific-content digests, file digests and runtime metadata;
- a full `FeatureProfile` fingerprint including feature order and units,
  channel, sampling, window, overlap/hop, quality policy, extractor version and
  phase reference;
- exact stratified lineage splits, replay/paired-lineage inheritance checks,
  duplicate-observation checks and lineage-aware raw-interval overlap checks;
- retention of every proposed window with its accepted/rejected disposition and
  reasons—no replacement sampling;
- safe, non-pickle NumPy arrays and canonical JSON only;
- atomic writes, no overwrite, path-containment checks, schema/version checks,
  per-file SHA-256 verification, object-array rejection, truncation rejection,
  a 256 MiB single-file load guard and a configurable 1 GiB artifact-root cap;
- read-only manifests and scientific payloads;
- a sealed test partition, rejected by default, with a chained append-only
  access ledger. Software sealing tests use an explicitly identified fixture;
- archive reload that reconstructs feature records and issues a new valid
  process-local HMAC instead of persisting the ephemeral live signature.

The manifest declares the expected future encoding
`aqse.tqk8.encoding.phase-direct.v1`, while scaler, theta, reference bank and
model identifiers remain null with `fit_state=not-fitted`.

## Historical v1 pilot and development artifacts

The measurements in this section are the original historical archive-v1
evidence. They are preserved verbatim and are not claims of archive-v2
conformance.

Artifacts were generated through the backend container and stored outside Git
in the host directory `/Users/florianocaprio/Projects/AQSE-artifacts` (mounted
as `/artifacts`). The values below are measurements from this run, not design
targets.

| Measurement | Pilot | Development |
| --- | ---: | ---: |
| Dataset ID | `aqse-pilot-1d0f7febb460b694` | `aqse-development-9804f50a8c029cae` |
| Episodes / lineages | 24 / 24 | 120 / 120 |
| Proposed windows | 456 | 2280 |
| Train accepted / rejected | n/a | 1368 / 0 |
| Validation accepted / rejected | n/a | 456 / 0 |
| Pilot accepted / rejected | 456 / 0 | n/a |
| Test windows | n/a | 456 proposed; quality counts intentionally not disclosed |
| Scientific digest | `1d0f7febb460b694c045a12a87ccc6a4557676ac23fdfd2da0be054f7383f7d3` | `9804f50a8c029caea047afc03a8fbe9a4ba9a5dda78dc76a44648f9ecc670fb8` |
| Feature profile fingerprint | `cc6f91470912cfc25025bb6190673f67cd3f56fc8265a7f4da311ad3c0eaeb17` | same |
| Serialized bytes | 1,478,651 | 7,376,699 |
| Archive phase time | 76.20 ms | 420.91 ms |
| Archive phase peak traced memory | 3,543,223 bytes | 11,700,506 bytes |
| In-memory regeneration time | 1,632.98 ms | 7,853.13 ms |
| Regeneration process maximum RSS | \- | 78,565,376 bytes after both stages |
| Observation tensor dimensions | 24 × 1000 × 3 | 120 × 1000 × 3 |
| Feature dimensions per episode | 19 × 8 | 19 × 8 |
| Test state | not applicable | **sealed** |

Archive phase time and memory exclude simulator/feature-generation time and
Docker startup. The separate regeneration measurement includes simulation and
feature extraction but not archive writes or Docker startup; maximum RSS is the
process high-water mark after both stages. These are operational metadata and
do not enter scientific identity. No development-test partition was loaded or
analyzed after sealing.

## Canonical v2 dataset freeze

The canonical archive-v2 freeze was executed on 2026-09-09 from branch
`codex/milestone-1d-tqk-training`. The approved branch commit at the start was
`db9494d437344100f81029148d70890e5c9762b7`. The container had no Git metadata,
so both archives correctly report `repository_base_sha=unrecorded` and
`repository_dirty=null`; no commit SHA was retroactively injected. The
effective source-file SHA-256 manifest stored in each archive is the
authoritative generation evidence. The recorded archive-writer source hash is
`c528d58a220fcedcc659ebc447783a01fbf6ec41814fb2ca15fc8331dda3f376`.

The first freeze command published pilot
`aqse-pilot-ce60fa50e20793cc` before complete-development candidate validation
detected an assignment-order mismatch and rejected/cleaned the development
staging directory. The mismatch was reproduced with a temporary fixture and
fixed by joining assignments by `episode_id`. The already immutable pilot was
not deleted or rewritten and is explicitly **not** designated canonical. The
command was then rerun successfully; the following two artifacts are the
canonical freeze:

| Measurement | Canonical pilot v2 | Canonical development v2 |
| --- | --- | --- |
| Dataset ID | `aqse-pilot-eafdf33fbb1e5811` | `aqse-development-064acca20fc788c6` |
| Scientific digest | `eafdf33fbb1e58116f5bff87fd209736d58faefcb5e5ff070c4fd6fb2e05937e` | `064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7` |
| Feature-profile fingerprint | `cc6f91470912cfc25025bb6190673f67cd3f56fc8265a7f4da311ad3c0eaeb17` | same |
| Episodes / lineages | 24 / 24 pilot | 72 / 72 train; 24 / 24 validation; 24 / 24 test |
| Proposed windows | 456 pilot | 1368 train; 456 validation; 456 sealed test |
| Accepted / rejected | 456 / 0 pilot | 1368 / 0 train; 456 / 0 validation; test not accessed |
| Test state | not applicable | `sealed` |

### Opaque v1/v2 numeric equivalence

The comparison below used only streaming SHA-256 over corresponding NPY bytes.
It did not call NumPy and did not semantically open the historical or v2 test
partition. All **18/18** development payloads are byte-identical.

| Partition | Numeric payload | Shared v1/v2 SHA-256 | Result |
| --- | --- | --- | --- |
| train | `time_s.npy` | `c35cc99495f029f3276ca29209ccabcf00a7dd6a0fc38b4b6836369739cbe24a` | match |
| train | `measured_field_T.npy` | `7e903cb2b55f81b059fdc2b11bfafdf32d1d3cd235900b0bda4a2263cdd0a0a4` | match |
| train | `temperature_K.npy` | `b4a9ca8c6d2a5a9107a0ab8121f62d8aa1ad33e4bfad0a4139eacaadfb87a8b5` | match |
| train | `saturation_mask.npy` | `8ca5671f4ebc2c3c698b00cae68fc5e86dc8345fa3519b3cc803081700b2380a` | match |
| train | `features.npy` | `71df716a7d9e31f800c7837fc3501ca141ebcf505143c8846e315e4acadeb772` | match |
| train | `valid_mask.npy` | `2daeb65771cc4f883f29c359dc801de127a34b5a8e776e3e6c4ef8c32f0e5e25` | match |
| validation | `time_s.npy` | `fd04586958188bbbf53d6f18f0ca6bcafdc5baaa5a4a3911e901cdb4858206f8` | match |
| validation | `measured_field_T.npy` | `642e19850162ece941f7a00e57591bb4238bcc3d0fd21271c77d742765075acf` | match |
| validation | `temperature_K.npy` | `68f82a5cb39bf74d94a1943cdd89a352897e10a6a00a6031804673d73e7ec5ee` | match |
| validation | `saturation_mask.npy` | `247364d855e0c7b8f49569f240a52220c9de1c96e97f9acee2d57de6e29064ca` | match |
| validation | `features.npy` | `dea0836f107232d3e40342214fad219203485495f96a206f1aa268771d93e2f7` | match |
| validation | `valid_mask.npy` | `7fddd8e2bc07ee9104f6eb9f4e9fd78d0dcab4953f3f695db0e62952db0b9af0` | match |
| test | `time_s.npy` | `fd04586958188bbbf53d6f18f0ca6bcafdc5baaa5a4a3911e901cdb4858206f8` | match |
| test | `measured_field_T.npy` | `845449e2d8d57a802461290e6287fb35b246c6f7cafcdc566bc69f336575110a` | match |
| test | `temperature_K.npy` | `fd2da74ffda7edbee54e422353d66419705ead2b1fd1ee446c64bfa85ebd9502` | match |
| test | `saturation_mask.npy` | `247364d855e0c7b8f49569f240a52220c9de1c96e97f9acee2d57de6e29064ca` | match |
| test | `features.npy` | `4215fed0d3f174eb2dfb204a6cecc3eb860456a80601aaaff40a988ce5b75697` | match |
| test | `valid_mask.npy` | `7fddd8e2bc07ee9104f6eb9f4e9fd78d0dcab4953f3f695db0e62952db0b9af0` | match |

### Permitted v2 semantic validation and test seal

The approved v2 loader semantically validated pilot, train and validation.
Raw-observable feature recalculation, feature arrays, valid masks, window
ledgers, quality accounting, full profile fingerprint and software provenance
all matched. Labels were loaded through their separate channel. The observable
DTO exposed none of `plans`, `labels`, `target`, configured noise or seed
fields, and all six returned NumPy arrays were read-only. Train and validation
had no shared lineage and no shared observation-content digest.

After publication no semantic loader was called for the v2 test. Opaque
verification confirmed its six manifest-listed numeric files, byte sizes and
SHA-256 values. `test_state` is `sealed`; quality counts remain hidden; the
access ledger contains exactly one entry: sequence 0, event `sealed`, with no
open event. No held-out values, labels, quality distribution, feature
statistics, kernels, baselines or model metrics were accessed or calculated.

The final freeze gate measured **201/201 backend tests**, including **8/8
original TQK8 tests**, and **51/51 training tests**; the explicit combined
TQK8/training rerun was **59/59**. Ruff, TypeScript typecheck, ESLint, 32/32
frontend tests and the production Vite build passed. `git diff --check` passed.
On the final images both Docker services were healthy; backend health returned
HTTP 200 in 1.92 ms, lightweight quantum readiness returned HTTP 200 in 3.00
ms, and the frontend returned HTTP 200 in 5.47 ms. The known non-blocking
warnings remain Starlette's `BlockingPortal` deprecation and Vite's existing
chunk-size advisory.

## GUI reliability changes included by explicit request

The previous GUI audit informed four bounded corrections:

1. raw charts render received numeric payloads even when `stuck` or
   `clock_error` flags make them ineligible for analysis; missing values remain
   explicit gaps;
2. the contiguous feature selector retains the first valid frame following a
   buffer gap and still excludes stuck/clock-error frames;
3. feature and quantum requests carry identities so late responses from a
   superseded request cannot replace current results or provenance;
4. the QNG diagram now distinguishes labels, alignment loss, loss gradient and
   empirical Fubini--Study geometry, and states that sensor training is not
   connected.

These changes do not add a training action or connect the dataset to the
quantum engine.

## Validation evidence

The final local suite measured:

- backend: **176 passed**, including **8/8** original TQK8 tests;
- frontend: **32 passed** across 10 test files;
- Ruff lint, TypeScript typecheck and ESLint: passed;
- production Vite build: passed;
- Docker backend and frontend: healthy on the final images;
- backend health: HTTP 200 in 3.54 ms;
- lightweight quantum readiness: HTTP 200 in 4.05 ms;
- one explicitly requested quantum diagnostic: HTTP 200 in 158.80 ms;
- frontend: HTTP 200 in 3.91 ms.

Dedicated tests cover cross-process regeneration, split cardinality,
lineage inheritance, duplicate and overlap rejection, observable-only feature
extraction, label separation, complete window accounting, corrupt/unsafe file
rejection, root limits, sealing/fixture ledger behavior, manifest permissions,
HMAC re-signing and the absence of training/AFSE/neural execution paths.

Known non-blocking warnings are the Starlette `BlockingPortal` deprecation
warning and Vite's existing bundle-size advisory. `ruff format --check .`
also reports 29 historical files outside this change set; all 12 new Python
files are formatter-clean. They were not mass-formatted because that would
rewrite unrelated and protected-scope code.

## External-review corrective addendum — 2026-09-09

This addendum records new evidence without rewriting the measurements above.
The review baseline was commit
`8691fd08afadb4a1c6e061977fdb4a09bee40db5`, based on
`1cb07cdccabf9c655a63f2b23aab37383ae90b69`. Local and remote milestone HEAD
matched and the working tree was clean before reproduction.

Six regressions were first run against the real application code using only
temporary pytest archives. All six failed as the review predicted: renamed
duplicate content crossed partitions, train and denied-test requests decoded
test arrays, an unmanifested NPY was returned, generation plans reached the
loader result, and a window-quality mutation could be hidden by refreshing
ordinary file checksums. These are reproduced failures, distinct from the
external audit and from the earlier 176-test baseline.

The corrective implementation creates archive contract v2 for new fixtures:

- observation content identity is independent from episode/export identity;
  a separate binding covers episode, lineage, raw-source origin and interval;
- the writer validates plan/label/assignment correspondence, content
  duplicates and raw-source overlap through the actual archive path;
- opaque verification streams sizes and SHA-256 without NumPy or reserved
  semantic decoding; authorization and an identity-bound, serialized ledger
  append precede any test semantic load;
- feature values have one canonical archived copy in `features.npy`; window
  identity, indices, times, observable quality, coverage, units and complete
  profile fingerprint are bound to scientific identity and verified by
  versioned raw-observable recalculation before HMAC re-signing;
- typed observation, label and generation loaders form separate channels;
- manifest-only allowlisting, per-read containment/symlink checks, bounded JSON,
  pre-allocation NPY-header validation, dtype/shape/physical-size checks,
  non-finite rejection and complete pre-publication candidate validation are
  enforced;
- software provenance records effective source hashes, runtime versions,
  algorithm/schema IDs, units, calibration/pose and discoverable Git base/dirty
  state. Operational write metadata remains outside numeric content identity.

The post-correction training package currently passes **50 tests**. These cover
R1--R5, including real-writer duplicate and independent-stream cases, semantic
I/O spies, denied and fixture-authorized sealing, concurrent ledger appends,
array/window/quality/index/fingerprint/unit mutations, extra files, symlinks,
oversized headers, wrong shapes, non-finite values, JSON bounds, candidate
cleanup, observable DTO exclusion and software provenance. Final full-suite
counts are recorded only after the release validation below is rerun.

Historical study archives remain unchanged v1 artifacts. V1 has bounded opaque
verification support only and is not described as conforming to v2. The
separate migration/rebuild proposal is documented but was not executed; no
study archive or sealed-study ledger was semantically opened, regenerated,
rewritten, or migrated during this correction.

The GUI received no functional or visual changes in this corrective pass.
STUCK/gap handling and the QNG diagram remain preserved. Preset/session
controls, cancellation and post-unmount request behavior remain review items;
no claim is made that every asynchronous path is resolved.

Final corrective validation on the uncommitted candidate produced:

- `make test`: **200 backend tests passed**, including **8/8 original TQK8
  tests**; Ruff passed; frontend typecheck and ESLint passed; **32 frontend
  tests passed** across 10 files; production Vite build passed;
- `docker compose ps`: backend and frontend healthy on rebuilt images;
- lightweight HTTP checks on the final recreated services: backend health 200
  in 1.76 ms, quantum readiness 200 in 2.59 ms, and frontend 200 in 4.70 ms;
- `git diff --check`: passed;
- opaque before/after SHA-256 manifest: all **39 historical study artifact
  files unchanged**, including the sealed-test payloads and ledger.

One earlier diagnostic invocation used `docker compose exec` without the
Makefile's read-only `/workspace/docker-compose.yml` test mount: 196 tests
passed and only `test_docker_healthcheck.py` failed with `StopIteration`
because that fixture path was absent. The official `make test` invocation
provided the required mount and the same test passed. There were no skips.
The only non-blocking warnings remain Starlette's `BlockingPortal` deprecation
and Vite's existing chunk-size advisory.

## Explicitly deferred

- production phase encoding and scaler fitting (1D.2);
- QNG training and checkpoints (1D.3);
- any real test opening or held-out evaluation (1D.4);
- workbench training controls (1D.5);
- AFSE and neural stages.
