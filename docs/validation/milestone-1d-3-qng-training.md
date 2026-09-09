# AQSE Milestone 1D.3 — QNG training validation record

## Evidence status

This record separates the approved 1D.3 protocol, the implemented software,
the designated canonical TRAIN execution, and the corrective audit/identity
evidence. It does not close Gate G4, select a model, open TEST, use VALIDATION
labels, or authorize Milestone 1D.4.

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
was not started to improve or choose a loss curve. No third candidate-training
run was started during corrective consolidation.

## Corrective trajectory identity and execution intent

The two historical `run.json` documents were loaded read-only and independently
reduced to the versioned deterministic scientific payload. They resolve exactly
to the same identity:

| Item | Value |
| --- | --- |
| Trajectory ID | `aqse-qng-trajectory-08f0020e9dbcd171` |
| Trajectory content digest | `08f0020e9dbcd171b59447545e8e01464cd7793fecbed250df899b19c46f3d8e` |
| Audit mapping digest | `2fba36334a8a224169ba97a153536fbe293333114396d285953827e54d6f8ea2` |
| Designated execution | `aqse-qng-run-f00c702ad790df2b` |
| Equivalent non-canonical execution | `aqse-qng-run-5ae026e633def66b` |
| Mapping `mapping.json` SHA-256 | `1b83c804f2b7a3525e4314507b530e829cfe8362219af229ef866cd729a0509c` |
| Mapping `manifest.json` SHA-256 | `37867a7f766d3c1cdd1742fbdfcfdf8d6cfd5392e90ba291d69e0cd1e7b7c9c2` |

The trajectory payload includes frozen input identity/digests, backend and
protected-source semantics, initialization, optimizer contract, every accepted
scientific step and aggregate evaluation counters. It excludes run/content
identity, job UUIDs, paths, timestamps, timing, RSS and execution provenance.
Changing theta, scientific input or optimizer contract changes the identity;
changing excluded execution measurements does not.

The immutable audit mapping was written outside Git without touching either
historical execution. Their six recorded file hashes remained byte-identical.
The canonical historical execution is also bound retrospectively to durable
intent `aqse-1d3-canonical-training-v1`; replaying that intent returned HTTP 200
with the original job/run and did not create a worker. Its immutable claim and
terminal-result digests are respectively
`57da9c95b7959a1d7e043f7c6b12bc813a234200a180c3e63184e9a5b8549acd`
and `92c552fd40c4ba0fa0d1ab466c317dea0fb711c042ef4746378a182c4bbc62cf`.
The API accepts no scientific overrides in an execution intent.

## Real-load responsiveness benchmark

One predeclared engineering benchmark was run once with the real 32-row TRAIN
bank and exact NumPy QNG implementation. It held the same heavy-quantum slot as
candidate training but used the separate `engineering_benchmark` purpose and
published no candidate artifact. Training-run directories were identical
before and after the benchmark.

| Measurement under active real QNG load | Result |
| --- | ---: |
| Intent | `aqse-1d3-g4-real-load-benchmark` |
| Accepted updates | 10 |
| Compute wall time | `2366.940 ms` |
| End-to-end wall time | `2371.525 ms` |
| Peak process RSS | `99,123,200` bytes |
| Differential / Gram / statevector calls | `10 / 11 / 10,912` |
| `/api/health` | HTTP 200, `1.732 ms` |
| `/api/quantum/health` | HTTP 200, `4.702 ms` |
| `/api/network/health` | HTTP 200, `5.237 ms` |
| Real one-frame simulator step | HTTP 200, `5.148 ms` |
| Quantum diagnostics contention | HTTP 429, `4.060 ms` |
| Quantum preview contention | HTTP 429, `4.001 ms` |

This is an engineering responsiveness measurement, not a second canonical
trajectory, checkpoint candidate, loss comparison or model-selection result.

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

Final corrective validation measured:

- complete backend: **238/238 passed**;
- original protected TQK8 regression: **8/8 passed**;
- complete training suite: **88/88 passed**;
- dedicated 1D.3 training/service/corrective suite: **21/21 passed**;
- corrective identity, intent and benchmark regression: **6/6 passed**;
- frontend: **32/32 passed** across 10 test files;
- Ruff, TypeScript typecheck, ESLint and Vite production build: passed;
- `git diff --check`: passed;
- rebuilt Docker backend/frontend: healthy;
- the real-load measurements are reported above.

The upstream Starlette `BlockingPortal` deprecation and existing Vite
chunk-size advisory are known non-blocking warnings.

The observed TRAIN loss decrease is an optimization result only. It is not
held-out predictive validation and does not authorize checkpoint promotion,
1D.4, AFSE or a neural model.
