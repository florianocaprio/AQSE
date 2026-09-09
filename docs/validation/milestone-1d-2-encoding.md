# AQSE Milestone 1D.2 — Encoding validation record

## Evidence status

This record separates the approved phase-direct policy from measured local
evidence. The materialization used only TRAIN observations/labels and
VALIDATION observations from the explicit canonical archive. It calculated no
alignment loss, predictive metric, classifier fit, QNG update or theta
optimization. The fixed non-zero theta below is solely a deterministic circuit
integration fixture.

## Canonical identities

| Item | Measured identity |
| --- | --- |
| Development dataset | `aqse-development-064acca20fc788c6` |
| Dataset scientific digest | `064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7` |
| Feature-profile fingerprint | `cc6f91470912cfc25025bb6190673f67cd3f56fc8265a7f4da311ad3c0eaeb17` |
| Encoding policy | `aqse.tqk8.encoding.phase-direct.v1` |
| Fitting population | `aqse-fitting-population-7e0d9449d2d571f9` |
| Fitting-population digest | `7e0d9449d2d571f9bc195301c577b236432d6fcd1f1c5f7e661436531ac802bd` |
| Scaler | `aqse-angle-scaler-fc1d44f00f4cc1be` |
| Encoder | `aqse-encoder-f9cf4bc12a767410` |
| Encoder digest | `f9cf4bc12a767410c8759dd01c5bb232ffc3877269af3ed219f3b6484db14e51` |
| TRAIN bank | `aqse-train-bank-32b5f93897676535` |
| TRAIN bank digest | `32b5f938976765355f0cd2886bee88082a693a06816f2bae25b9ec268973d8dd` |
| VALIDATION bank | `aqse-validation-bank-105467fa5a4a5cc7` |
| VALIDATION bank digest | `105467fa5a4a5cc7c867f7f77d92eaa61d231949e7c2356e8f3817c541c0f047` |

Artifacts are outside Git. Encoder and bank writes passed their load-back
verification before atomic publication; subsequent overwrite attempts are
rejected.

## Train-only fit evidence

The archive verified 72 TRAIN episodes with 19 eligible windows each: 1,368
rows. All were supplied once to the unchanged protected `AngleScaler.fit()`.
The frozen statistics are:

| Coordinate | Mean | Scale | Active output use |
| --- | ---: | ---: | --- |
| amplitude | 7.743500173149733 | 1.1540838936245088 | protected scaler |
| phase | -0.00651237462452469 | 1.8058499079753036 | stored only; replaced by direct phase |
| frequency | 5.959186022137233 | 1.190160981497035 | protected scaler |
| variance | 33.86264842037067 | 9.177137096851304 | protected scaler |
| drift | 0.0050669209981008476 | 0.6747568658951649 | protected scaler |
| snr | 11.648439567067744 | 5.224200410569491 | protected scaler |
| spectral_peak | 18.19469058648047 | 5.8855700013486585 | protected scaler |
| temperature | 293.0275565113102 | 0.5671899773403217 | protected scaler |

Dedicated regressions demonstrate repeatable identity from identical TRAIN
content, rejection of non-TRAIN fitting, immutability under VALIDATION
transforms and perturbations, query-order invariance, exact non-phase
delegation, canonical phase wrapping, and rejection of object, non-finite and
wrong-shape inputs.

Negative contract cases cover dataset identity, full profile fingerprint,
feature order, units, phase origin, encoding policy, scaler ID, TQK8 source
identity, 8-input contract and corrupted artifact digest. Every case is
rejected before a usable encoded result is returned.

## Bank evidence

The TRAIN artifact records 32 rows from 32 distinct TRAIN lineages, one fixed
ordinal-9 window each, using seed `1001004` and exactly 16/16 labels. The
VALIDATION artifact records all 24 validation lineages, one ordinal-9 window
each. Tests reproduce both artifacts deterministically and demonstrate that
arbitrary feature-value perturbation cannot change bank membership. No
overlapping windows from one episode are treated as independent rows.

No TEST bank exists. The builder has no TEST semantic-load call and its opaque
seal test fails if `AQSE_ALLOW_TEST_OPEN=1`.

## Exact-state integration evidence

Tolerance was fixed at absolute `1e-12` before the diagnostic. The fixture used
the protected VQC and deterministic non-zero theta
`linspace(-0.45, 0.55, 16)`. This is a loss-free infrastructure diagnostic,
not training or evaluation.

| Measurement | Result | Gate |
| --- | ---: | ---: |
| `phase` and `phase + 2*pi`, NumPy density max absolute delta | `2.221490086301386e-16` | `<= 1e-12` |
| `phase` and `phase + 2*pi`, Qiskit density max absolute delta | `3.885812369975715e-16` | `<= 1e-12` |
| Seam fidelity, NumPy, epsilon `1e-8` rad | `0.9999999999999993` | `1 - K <= 1e-12` |
| Seam fidelity, Qiskit, epsilon `1e-8` rad | `0.9999999999999996` | `1 - K <= 1e-12` |
| NumPy/Qiskit kernel max absolute delta | `2.220446049250313e-16` | `<= 1e-12` |

All numerical gates pass without circuit changes.

## Seal and protected assets

After publication, the canonical TEST state remains `sealed`. Semantic TEST
access count is exactly **0**. Its ledger contains exactly one entry,
`sequence=0,event=sealed`, and has SHA-256
`210077e41754c47eebe572660f07b7cdaf8653ade2945fccd368e04d64a432c6`.
No held-out values, labels, quality statistics, kernels or metrics were read or
computed.

| Protected/effective source | SHA-256 |
| --- | --- |
| `tqk8.py` | `cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689` |
| `sampler_qng.py` | `7489c5c2b3eb0783499c333dc0826994b146d7cf5631469a0520bac9570b2ea6` |
| Original TQK8 test | `c87c56c0eb3f1ea20d2dd124427aea96f868ae4674b5b390e2688e9da8bb0c09` |
| TQK8 notebook | `9d7e0337af7c039894fc6c53567de2883063b32c89e368753bd01e1d9e9e82be` |
| `encoding.py` materialized source | `8d0846513d68dfc7334d1de330b09108e5f51608b1e20111d8333d7d27bb3cd2` |
| `encoding_models.py` materialized source | `3ab5801790d44f0186a8218d381a1bd4fc6cbc0ed3305c4cbef7fd91e14b2144` |
| `banks.py` materialized source | `1a276d3ec37334100510fb2088cd25ec944e6be52dea6f63ab3313e8c92c6b5a` |

Materialization reported `repository_base_sha=unrecorded` and
`repository_dirty=null` because Git was unavailable in that container. This
operational limitation is explicit and does not replace the effective source
hashes stored in the artifact.

## Validation gate

The final code-quality gate measured:

- complete backend: **217/217 passed**;
- original protected TQK8 regression: **8/8 passed**;
- complete training suite: **67/67 passed**;
- dedicated 1D.2 encoding/bank/seal suite: **16/16 passed**;
- frontend: **32/32 passed** across 10 test files;
- Ruff, TypeScript typecheck and ESLint: passed;
- Vite production build: passed;
- `git diff --check`: passed;
- final Docker backend/frontend: healthy;
- lightweight backend health: HTTP 200 in 2.986 ms;
- lightweight quantum readiness: HTTP 200 in 1.652 ms;
- frontend: HTTP 200 in 3.038 ms.

The one Starlette `BlockingPortal` deprecation and existing Vite chunk-size
advisory are non-blocking and are unrelated to this change set.

Milestone 1D.3 training, theta updates, QNG runs, checkpoints, test opening,
evaluation, AFSE, neural processing and GUI training integration remain
unimplemented.
