# AQSE Milestone 1D.4b — Held-out validation record

## Entry state and chronology

Milestone 1D.4b entered on branch `codex/milestone-1d-tqk-training` at the
approved local/remote HEAD
`81045762d69c6a82c63e47196a2ae36899786233`, divergence `0/0`, with a clean
working tree. `main`, `origin/main`, and all tags were unchanged.

Before any TEST semantic access:

- the canonical ledger contained only sequence 0 and had SHA-256
  `210077e41754c47eebe572660f07b7cdaf8653ade2945fccd368e04d64a432c6`;
- the backend suite passed 254/254 and Ruff passed;
- the protected source hashes matched their prior freeze;
- the fixture-based 1D.4b contract and ledger tests passed;
- the immutable G5 authorization
  `aqse-g5-authorization-2007f6ca3fec727a` was published with no TEST values;
- the ledger remained byte-identical after authorization publication.

One pre-gate backend invocation omitted the read-only `docker-compose.yml`
mount required by `test_docker_healthcheck.py`: 253 tests passed and that one
environmental test could not locate its fixture. Repeating the suite with the
same mount used by `make test` produced 254/254. No code or TEST state changed
because of the failed invocation.

The final operation ran once in an isolated container with
`AQSE_ALLOW_TEST_OPEN=1`. That variable was not persisted to an `.env` file or
Docker Compose. It was absent from every later validation command.

## Immutable evidence graph

| Artifact | ID | Scientific digest |
| --- | --- | --- |
| Dataset | `aqse-development-064acca20fc788c6` | `064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7` |
| Encoder | `aqse-encoder-f9cf4bc12a767410` | `f9cf4bc12a767410c8759dd01c5bb232ffc3877269af3ed219f3b6484db14e51` |
| Protocol | `aqse-comparative-protocol-a671952c35f0ffe8` | `a671952c35f0ffe80707a4f6fed48ff3ff17c679a3392ce4c1fc697f087eff9d` |
| 1D.4a evaluation | `aqse-comparative-evaluation-5b0aeed2cdd9aeba` | `5b0aeed2cdd9aeba51bf99ef43ff304a2770c2c422b127d9e07ee66a4c3c4fb0` |
| Selection freeze | `aqse-model-selection-freeze-433ca4c0cae8fabf` | `433ca4c0cae8fabf06120f7e929c0e0471d338f48f120aad15f18611f7b4016f` |
| G5 authorization | `aqse-g5-authorization-2007f6ca3fec727a` | `2007f6ca3fec727ab63f1d3601114be3c5bdfb4e995a4e4c654be727a5211e05` |
| TEST reference bank | `aqse-test-bank-2e330d59973e5b37` | `2e330d59973e5b377aa1f5792659fe31cc171de66f1bf5bb760f4ca46e82a617` |
| Held-out input | `aqse-held-out-input-25977d5500002ed9` | `25977d5500002ed9b42082b071a2f0b988caf9dd7b9b872ae459799975c4b897` |
| Final held-out result | `aqse-final-held-out-7f338dafc2c02f09` | `7f338dafc2c02f0959b33caf8c3033cf7a971ba455d4c8593bbde91774404228` |

All three newly published artifact families are outside Git, atomically
written without pickle or overwrite, restricted by file allowlists and
SHA-256 manifests, and validated by load-back. Git was unavailable in the
materializing container, so execution metadata honestly records
`repository_base_sha=unrecorded` and `repository_dirty=null`.

## TEST ledger and access proof

The final ledger has exactly three valid chained entries:

| Sequence | Event | Entry SHA-256 | Reason |
| ---: | --- | --- | --- |
| 0 | sealed | `b88efed1f589c0a1a3dc3873de2f2d0ea4ac0a7dc21ce64d614aa4ff021919d6` | `test partition sealed at dataset creation` |
| 1 | opened | `274eaa553eef917a566421c40c54ae8ef865b8a9d335d6e095287976d5cf1687` | `freeze=aqse-model-selection-freeze-433ca4c0cae8fabf; fixed-test-bank-and-observations` |
| 2 | opened | `1487220e9d0346ffd19dc3a232dd8f9ec1ce11cafd28db4a1b2719e2b5f1bd49` | `freeze=aqse-model-selection-freeze-433ca4c0cae8fabf; final-held-out-labels` |

Final ledger SHA-256:
`e0d4898141d8be07f4a4f1af7582791eae61994ced52bb7b3dba30a7ddda4b70`.
Semantic TEST load count is exactly two. TEST generation-plan access count is
zero. No canonical TEST semantic read occurred during post-run validation;
those checks load only manifests, safe result artifacts and the ledger.

## Cohort, compatibility and measured results

The TEST bank contains 24 distinct lineages, all at ordinal 9. Every fixed
window was eligible, giving 24 eligible, zero abstained, and 100% coverage for
the common five-method cohort. TEST lineages are disjoint from TRAIN and
VALIDATION; no duplicate observable content crosses partitions. The encoder,
TRAIN bank, TRAIN-only RBF preprocessing and selected theta identities match
their freezes. Each quantum TEST-by-TRAIN cross-kernel is `(24, 32)`.

| Method | BA, 95% interval | Macro-F1, 95% interval | Confusion `[-1,+1]` | Recall `-1/+1` | AUC | Runtime |
| --- | --- | --- | --- | --- | ---: | ---: |
| SNR threshold | `0.958333 [0.875000,1.000000]` | `0.958261 [0.873016,1.000000]` | `[[12,0],[1,11]]` | `1.000000/0.916667` | 0.993056 | `1,931.511 ms` |
| Circular RBF-SVC | `0.833333 [0.666667,0.958333]` | `0.833333 [0.666667,0.958261]` | `[[10,2],[2,10]]` | `0.833333/0.833333` | 0.930556 | `1,867.938 ms` |
| Fixed-theta TQK | `1.000000 [1.000000,1.000000]` | `1.000000 [1.000000,1.000000]` | `[[12,0],[0,12]]` | `1.000000/1.000000` | 1.000000 | `1,930.585 ms` |
| Ordinary-gradient TQK | `1.000000 [1.000000,1.000000]` | `1.000000 [1.000000,1.000000]` | `[[12,0],[0,12]]` | `1.000000/1.000000` | 1.000000 | `1,950.866 ms` |
| QNG TQK | `1.000000 [1.000000,1.000000]` | `1.000000 [1.000000,1.000000]` | `[[12,0],[0,12]]` | `1.000000/1.000000` | 1.000000 | `1,878.927 ms` |

Total local wall time was `9,563.448 ms`. The quantum records each account for
1,024 TRAIN-kernel and 768 TEST-by-TRAIN kernel coordinates. Fixed, GD and QNG
theta, kernel, prediction digest and score digest are identical within the
frozen tolerance.

`no_winner_predeclared=true` and
`test_results_not_used_for_model_selection=true`. No hyperparameter, theta,
checkpoint, preprocessing or fitting population changed after viewing TEST.
The result does not justify a quantum-advantage claim: the simulation is
strongly SNR-separable, the held-out sample is small, and the three quantum
methods are the same frozen checkpoint-0 evaluator.

## Software and service validation

Final measured validation:

- complete backend: **255/255 passed**;
- complete training suite: **105/105 passed**;
- dedicated 1D.4b suite: **8/8 passed**;
- original protected TQK8 regression: **8/8 passed**;
- frontend: **32/32 passed** across 10 files;
- Ruff, TypeScript typecheck, ESLint, Vite production build and
  `git diff --check`: passed;
- Docker backend and frontend: healthy after rebuild and restart.

Protected identities remain:

| Asset | SHA-256 |
| --- | --- |
| `tqk8.py` | `cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689` |
| `sampler_qng.py` | `7489c5c2b3eb0783499c333dc0826994b146d7cf5631469a0520bac9570b2ea6` |
| Original TQK8 tests | `c87c56c0eb3f1ea20d2dd124427aea96f868ae4674b5b390e2688e9da8bb0c09` |
| TQK8 notebook | `9d7e0337af7c039894fc6c53567de2883063b32c89e368753bd01e1d9e9e82be` |

Known non-blocking warnings remain the upstream Starlette `BlockingPortal`
deprecation and Vite's existing chunk-size advisory. Milestone 1D.5 was not
started.
