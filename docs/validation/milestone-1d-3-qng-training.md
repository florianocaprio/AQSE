# AQSE Milestone 1D.3 — QNG training validation record

## Evidence status

This record separates the approved 1D.3 protocol, the implemented software,
and measurements from the single designated canonical TRAIN trajectory. It
does not close Gate G4, select a model, open TEST, use VALIDATION labels, or
authorize Milestone 1D.4.

Entry branch was `codex/milestone-1d-tqk-training`; the approved entry HEAD was
`ab5bff99fa9779a143e236595c0d4c1e78ff349f`. The working tree was clean and
local/remote divergence was `0/0`. `main`, `origin/main`, and the
`milestone-1c` tag were not modified.

## Canonical identity and preflight evidence

| Item | Measured identity |
| --- | --- |
| Dataset | `aqse-development-064acca20fc788c6` |
| Dataset digest | `064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7` |
| Encoder | `aqse-encoder-f9cf4bc12a767410` |
| Encoder digest | `f9cf4bc12a767410c8759dd01c5bb232ffc3877269af3ed219f3b6484db14e51` |
| TRAIN bank | `aqse-train-bank-32b5f93897676535` |
| TRAIN bank digest | `32b5f938976765355f0cd2886bee88082a693a06816f2bae25b9ec268973d8dd` |
| Training-input fingerprint | `aqse-training-input-e853259fac7c0eba` |
| Training-input digest | `e853259fac7c0ebab47ef29ea53b42d14a64659588accc79864365b2c96d715a` |
| Shape and balance | `32 x 8`; 16/16 TRAIN labels |

The mandatory canonical `N=3` wrapper comparison produced zero delta for both
the final theta and every protected history field. The one-step Qiskit/NumPy
comparison measured kernel delta `1.5543122344752192e-15`, loss delta `0.0`,
gradient delta `2.7755575615628914e-17`, metric delta
`5.551115123125783e-17`, and candidate-theta delta
`5.551115123125783e-17`. These pass the predeclared `1e-12` fidelity and
`1e-10` gradient/metric/update thresholds.

## Designated canonical run

| Measurement | Result |
| --- | --- |
| Artifact ID | `aqse-qng-run-f00c702ad790df2b` |
| Content digest | `f00c702ad790df2bc9e3f4aade14d823565fe9b50bb33996a4313c28784673d2` |
| Backend | exact `numpy_statevector` |
| Theta policy / seed | `aqse.theta-init.uniform-v1` / `1001005` |
| Accepted updates | 10/10 |
| Stop reason | `MAX_UPDATES_REACHED` |
| Initial TRAIN alignment loss | `0.8263338624724166` |
| Final TRAIN alignment loss | `0.5515703229976026` |
| Runner wall time | `2067.119792991434 ms` |
| End-to-end API lifecycle | approximately `2752 ms` |
| Peak process RSS | `97820672` bytes |
| Differential calls | 10 |
| Gram calls | 11 |
| Counted statevector evaluations | 10,912 |

The exact final candidate theta is:

```text
[-0.05328002850299152, -0.9286886710860751,
 -0.080694032472177,   -0.08370451638500866,
 -0.17040355167748014, -1.3668242607926595,
 -0.5166098396779399,  -0.7460687471424304,
  0.34282047880523026,  0.7980780561187012,
  0.04997629043545576, -0.609810519929328,
 -0.1992609993674046,  -1.3161432592428204,
  0.5588304704331543,   0.5912905770423051]
```

### Accepted-step history

| Step | Loss before | Loss after | Gradient norm | Step norm | FS min eigenvalue | Time ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8263338625 | 0.8182897659 | 0.0872176554 | 0.0898548045 | 0.0615504061 | 204.659 |
| 1 | 0.8182897659 | 0.8056996663 | 0.1076163907 | 0.1163634050 | 0.0615398340 | 204.188 |
| 2 | 0.8056996663 | 0.7866120350 | 0.1318001708 | 0.1471690263 | 0.0615236315 | 203.209 |
| 3 | 0.7866120350 | 0.7596042823 | 0.1572526079 | 0.1789647838 | 0.0615008128 | 197.993 |
| 4 | 0.7596042823 | 0.7253417487 | 0.1787660124 | 0.2057124451 | 0.0614716280 | 210.360 |
| 5 | 0.7253417487 | 0.6874548962 | 0.1897645274 | 0.2206288044 | 0.0614389556 | 208.093 |
| 6 | 0.6874548962 | 0.6504399142 | 0.1876541242 | 0.2220593363 | 0.0614081632 | 207.636 |
| 7 | 0.6504399142 | 0.6161826319 | 0.1782813235 | 0.2167711687 | 0.0613838983 | 206.736 |
| 8 | 0.6161826319 | 0.5836708757 | 0.1708950962 | 0.2132358822 | 0.0613671592 | 206.726 |
| 9 | 0.5836708757 | 0.5515703230 | 0.1688583037 | 0.2133572647 | 0.0613562955 | 211.136 |

Each accepted step counted one protected differential evaluation, one Gram
evaluation and 1,088 statevector evaluations. Theta values above are raw
optimizer coordinates; no range clamp or modulo normalization was applied.

Load-back validation passed. File hashes are:

| File | SHA-256 |
| --- | --- |
| `run.json` | `770bf5533adce51ccaf596e672fe1190dea609c8f3ba1087d5ca56e9862bd106` |
| `execution.json` | `3b845d8d4b7541db0dab388e4d749d5c62bc7d9cc9ca2a3dfd58cb50a3a40c59` |
| `manifest.json` | `513064126a66acba9a1d0611b08e98a3ec842415706d79b264f550893098137f` |

Git was unavailable inside the materializing runtime, so execution provenance
records `repository_base_sha=unrecorded` and `repository_dirty=null`. The
artifact independently binds the effective hashes of `tqk8.py`, assembly,
models and runner.

## Operational deviation retained for audit

The first job completed before the live contention probes were issued. A
second `POST /api/training/jobs`, intended only to verify HTTP 429 while the
first job was active, therefore received HTTP 201 and completed before its
cancellation request arrived. It published
`aqse-qng-run-5ae026e633def66b` with digest
`5ae026e633def66b477537147962e1b9b35658e6edcff25357231c2660614551`.

This second artifact is retained, immutable, and explicitly non-canonical. It
was not started to improve or choose a loss curve. After removing timing, peak
RSS and identity fields, its normalized scientific payload is byte-identical
to the designated run (comparison SHA-256
`2231f353a44cab6f087adb60638ba0142f6406ee7abf253d049fa27dd5eaafc3`).
No third run was started. This deviation prevents claiming a clean
exactly-one-execution audit even though exactly one artifact is designated
canonical.

The lightweight endpoint measurements captured immediately afterward were
`2.637 ms` for `/api/health`, `3.324 ms` for `/api/quantum/health`, and
`2.940 ms` for `/api/network/health`; because the first run had already ended,
they are not represented as live concurrent canonical-run measurements.
Automated lifecycle tests instead hold a job in `RUNNING`, execute a real
one-frame sensor session within the one-second bound, keep both lightweight
health endpoints callable, and verify 429 for diagnostics and a second job.

## TEST seal and protected assets

After both executions, canonical TEST remains `sealed`; its semantic-access
count remains zero. The ledger still contains exactly the single sealed genesis
entry and SHA-256
`210077e41754c47eebe572660f07b7cdaf8653ade2945fccd368e04d64a432c6`.
No held-out values, labels, statistics, kernels or metrics were read.

| Protected asset | SHA-256 |
| --- | --- |
| `tqk8.py` | `cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689` |
| `sampler_qng.py` | `7489c5c2b3eb0783499c333dc0826994b146d7cf5631469a0520bac9570b2ea6` |
| Original TQK8 tests | `c87c56c0eb3f1ea20d2dd124427aea96f868ae4674b5b390e2688e9da8bb0c09` |
| TQK8 notebook | `9d7e0337af7c039894fc6c53567de2883063b32c89e368753bd01e1d9e9e82be` |

## Validation status

Final validation measured:

- complete backend: **232/232 passed**;
- original protected TQK8 regression: **8/8 passed**;
- complete training suite: **82/82 passed**;
- dedicated 1D.3 training/service suite: **15/15 passed**;
- frontend: **32/32 passed** across 10 test files;
- Ruff, TypeScript typecheck, ESLint and Vite production build: passed;
- `git diff --check`: passed;
- rebuilt Docker backend/frontend: healthy;
- lightweight backend health: HTTP 200 in `1.891 ms`;
- lightweight quantum readiness: HTTP 200 in `2.441 ms`;
- network health: HTTP 200 in `2.800 ms`;
- frontend: HTTP 200 in `4.267 ms`;
- standalone real one-frame simulator step: `0.920 ms`.

The upstream Starlette `BlockingPortal` deprecation and existing Vite
chunk-size advisory are known non-blocking warnings.

The observed TRAIN loss decrease is an optimization result only. It is not
held-out predictive validation and does not authorize checkpoint promotion,
1D.4, AFSE or a neural model.
