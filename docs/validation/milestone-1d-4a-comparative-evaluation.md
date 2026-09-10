# AQSE Milestone 1D.4a — Comparative evaluation validation record

## Evidence status

This record documents measured TRAIN/VALIDATION results and the resulting
model-selection freeze. It does not close Gate G5 or authorize TEST access.
The entry branch was `codex/milestone-1d-tqk-training` at approved HEAD
`1b4435b2699066211aa2e9bb8e9cd8cd0b8af82b`, with local/remote divergence
`0/0` and a clean working tree.

## Freeze chronology and identities

The protocol was written before the first VALIDATION-label access. The first
attempt to invoke its script by file path failed at import time and wrote
nothing; the corrected module invocation then published the immutable protocol.

| Artifact | ID | Scientific digest | Payload SHA-256 |
| --- | --- | --- | --- |
| Protocol | `aqse-comparative-protocol-a671952c35f0ffe8` | `a671952c35f0ffe80707a4f6fed48ff3ff17c679a3392ce4c1fc697f087eff9d` | `a9ce09bf5d1bc515b34224adb29b26348743fd84b898a01d112fa10d6b65a988` |
| Comparative evaluation | `aqse-comparative-evaluation-5b0aeed2cdd9aeba` | `5b0aeed2cdd9aeba51bf99ef43ff304a2770c2c422b127d9e07ee66a4c3c4fb0` | `1d0e6fa5b3a81eff1f79782a94147a2e538161f4b861143dcf78f751f38fb959` |
| Model-selection freeze | `aqse-model-selection-freeze-433ca4c0cae8fabf` | `433ca4c0cae8fabf06120f7e929c0e0471d338f48f120aad15f18611f7b4016f` | `b4f5b294351efe09ed9a5bbb8b7b1787bbc144d016caab8dc6e9365ca7c3ef03` |

Protocol creation was recorded at `2026-09-09T19:34:20.192557Z`, comparative
publication at `2026-09-09T19:36:12.991613Z`, and selection freeze at
`2026-09-09T19:36:13.051045Z`. Git was unavailable inside the materializing
container, so execution provenance honestly records
`repository_base_sha=unrecorded` and `repository_dirty=null`.

The comparison-input fingerprint is
`aqse-comparison-input-0def6d051fdecdf8`, digest
`0def6d051fdecdf814557126c08972e0026bb12a14e74db37474d92b73525c38`.
Its identities bind both raw and encoded matrices, separately loaded labels,
bank order, window IDs and disjoint lineage IDs.

## Candidate budget and selected results

Exactly 355 candidates were evaluated: 1 SNR threshold, 9 RBF settings, 15
fixed-theta TQK settings, 165 ordinary-gradient settings and 165 QNG settings.
No VALIDATION value was used to change that budget.

| Method | TRAIN BA | VALIDATION BA | 95% lineage interval | VALIDATION macro-F1 | Confusion `[-1,+1]` | Selected configuration |
| --- | ---: | ---: | --- | ---: | --- | --- |
| SNR threshold | 1.000000 | 1.000000 | `[1.000, 1.000]` | 1.000000 | `[[12,0],[0,12]]` | threshold `11.433898484324114` |
| Circular RBF-SVC | 0.968750 | 0.875000 | `[0.750, 1.000]` | 0.873016 | `[[12,0],[3,9]]` | `C=10`, `gamma=0.01` |
| Fixed-theta TQK | 1.000000 | 0.958333 | `[0.875, 1.000]` | 0.958261 | `[[12,0],[1,11]]` | seed `1001008`, checkpoint 0, `C=1` |
| Ordinary-gradient TQK | 1.000000 | 0.958333 | `[0.875, 1.000]` | 0.958261 | `[[12,0],[1,11]]` | seed `1001008`, checkpoint 0, `C=1` |
| QNG TQK | 1.000000 | 0.958333 | `[0.875, 1.000]` | 0.958261 | `[[12,0],[1,11]]` | seed `1001008`, checkpoint 0, `C=1` |

The selected quantum initialization has TRAIN alignment loss
`0.7649299334650025` and VALIDATION alignment loss `0.654453626961554`.
Every checkpoint index from 0 through 10 had an available best VALIDATION
balanced accuracy of `0.958333` for both GD and QNG; the frozen earliest-step
tie-break therefore selected checkpoint 0. The result is not evidence that
either optimization method improved validation performance.

SNR alone separates this simulated noise-regime dataset perfectly on TRAIN and
VALIDATION. This confirms a strong task-specific baseline and makes a quantum
advantage claim inappropriate. No method is declared globally selected or
superior; the freeze retains one configuration per comparator for a possible
later TEST evaluation.

## Measured compute

The isolated evaluation container recorded total wall time `30,839.585 ms`.

| Method family | Wall time | Quantum coordinates evaluated |
| --- | ---: | ---: |
| SNR threshold | `87.899 ms` | 0 |
| Circular RBF-SVC | `54.860 ms` | 0 |
| Fixed-theta TQK | `216.776 ms` | 5 |
| Ordinary-gradient TQK | `11,613.267 ms` | 55 |
| QNG TQK | `9,675.199 ms` | 55 |

These are local exact-simulator engineering timings, not physical-QPU costs or
shots. The seed-`1001005` QNG checkpoint history came from the designated 1D.3
run; it was not rerun.

## TEST seal and scientific assets

Before protocol freeze, after comparison assembly, after evaluation and after
artifact publication, TEST remained `sealed`. The ledger still has exactly its
single genesis event and SHA-256
`210077e41754c47eebe572660f07b7cdaf8653ade2945fccd368e04d64a432c6`.
`AQSE_ALLOW_TEST_OPEN` was not enabled. No TEST bank was created and no TEST
semantic payload was read.

Protected identities remain:

| Asset | SHA-256 |
| --- | --- |
| `tqk8.py` | `cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689` |
| `sampler_qng.py` | `7489c5c2b3eb0783499c333dc0826994b146d7cf5631469a0520bac9570b2ea6` |
| Original TQK8 tests | `c87c56c0eb3f1ea20d2dd124427aea96f868ae4674b5b390e2688e9da8bb0c09` |
| TQK8 notebook | `9d7e0337af7c039894fc6c53567de2883063b32c89e368753bd01e1d9e9e82be` |

## Software validation

Final validation measured:

- complete backend: **247/247 passed**;
- original protected TQK8 regression: **8/8 passed**;
- complete training suite: **97/97 passed**;
- dedicated 1D.4a suite: **9/9 passed**;
- frontend: **32/32 passed**;
- Ruff, TypeScript typecheck, ESLint and Vite production build: passed;
- `git diff --check`: passed;
- rebuilt Docker backend/frontend: healthy.

The upstream Starlette `BlockingPortal` deprecation and existing Vite
chunk-size advisory remain known, non-blocking warnings.
