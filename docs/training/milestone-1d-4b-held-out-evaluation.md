# AQSE Milestone 1D.4b — Final held-out evaluation

## Scope and authorization

Gate G5 was explicitly closed before TEST was opened. Milestone 1D.4b executes
the single held-out evaluation procedure already frozen in 1D.4a. It does not
change a method selection, search a hyperparameter, rerun GD or QNG training,
refit on TRAIN+VALIDATION, promote a model, start AFSE, or integrate a model
with the GUI.

The immutable authorization was published outside Git before the first TEST
semantic read:

- authorization ID: `aqse-g5-authorization-2007f6ca3fec727a`;
- authorization digest:
  `2007f6ca3fec727ab63f1d3601114be3c5bdfb4e995a4e4c654be727a5211e05`;
- approved entry commit: `81045762d69c6a82c63e47196a2ae36899786233`;
- purpose: `milestone-1d4b-final-held-out-evaluation`;
- initial TEST ledger SHA-256:
  `210077e41754c47eebe572660f07b7cdaf8653ade2945fccd368e04d64a432c6`;
- `contains_test_values=false`.

The authorization binds the canonical dataset, encoder, TRAIN bank,
comparative protocol, 1D.4a evaluation, model-selection freeze, exact five
selections, metric/bootstrap policy, fixed TEST cohort rule and expected
two-call semantic access plan.

## TEST access and cohort

The final operation made exactly two semantic calls, in this order:

1. `load_observations(TEST)`, reason
   `freeze=aqse-model-selection-freeze-433ca4c0cae8fabf; fixed-test-bank-and-observations`;
2. `load_labels(TEST)`, reason
   `freeze=aqse-model-selection-freeze-433ca4c0cae8fabf; final-held-out-labels`.

`load_generation_channel(TEST)` was never called. TEST was not semantically
loaded again after the operation and was not resealed. The final chained ledger
contains exactly three events and has SHA-256
`e0d4898141d8be07f4a4f1af7582791eae61994ced52bb7b3dba30a7ddda4b70`.

The immutable TEST reference bank is
`aqse-test-bank-2e330d59973e5b37`, digest
`2e330d59973e5b377aa1f5792659fe31cc171de66f1bf5bb760f4ca46e82a617`.
It contains all 24 distinct TEST lineages, always uses zero-based window
ordinal 9, is independent of TEST labels, and permits no fallback or nearby
window substitution. All 24 fixed windows were valid for quantum processing,
so the common cohort has 24 eligible rows, zero abstentions, and coverage
`1.0` for every method.

Before final publication, the operation verified lineage disjointness from
TRAIN and VALIDATION, no duplicate observable content across partitions,
encoder compatibility, the frozen TRAIN-only RBF preprocessing, a quantum
TEST-by-TRAIN kernel shape of `(24, 32)`, and identical fixed/GD/QNG theta and
`C=1` configurations.

## Frozen reconstruction

Every evaluator was reconstructed from TRAIN only:

- SNR used the frozen threshold `11.433898484324114` and predicted `+1` when
  SNR was at or below that threshold;
- circular RBF-SVC used the frozen TRAIN-only mean/scale snapshot with `C=10`
  and `gamma=0.01`; no scaler was fitted on TEST or TRAIN+VALIDATION;
- fixed-theta, ordinary-gradient and QNG TQK used the frozen 32-row encoded
  TRAIN reference, seed `1001008`, checkpoint 0, the exact frozen theta and
  precomputed-kernel SVC `C=1`;
- neither ordinary-gradient nor QNG training was rerun.

The three quantum method-family selections are numerically the same
checkpoint-0 model. Their TEST kernels, predictions and scores were verified
identical within absolute tolerance `1e-12`.

## Held-out results

The safe immutable final artifact is
`aqse-final-held-out-7f338dafc2c02f09`, digest
`7f338dafc2c02f0959b33caf8c3033cf7a971ba455d4c8593bbde91774404228`.
The held-out input identity is `aqse-held-out-input-25977d5500002ed9`, digest
`25977d5500002ed9b42082b071a2f0b988caf9dd7b9b872ae459799975c4b897`.

Intervals are percentile 95% intervals from 2,000 stratified resamples of
independent TEST lineages with frozen seed `1001010`.

| Method | Balanced accuracy (95%) | Macro-F1 (95%) | Confusion `[-1,+1]` | Recall `-1/+1` | ROC-AUC |
| --- | --- | --- | --- | --- | ---: |
| SNR threshold | `0.958333 [0.875000, 1.000000]` | `0.958261 [0.873016, 1.000000]` | `[[12,0],[1,11]]` | `1.000000 / 0.916667` | 0.993056 |
| Circular RBF-SVC | `0.833333 [0.666667, 0.958333]` | `0.833333 [0.666667, 0.958261]` | `[[10,2],[2,10]]` | `0.833333 / 0.833333` | 0.930556 |
| Fixed-theta TQK | `1.000000 [1.000000, 1.000000]` | `1.000000 [1.000000, 1.000000]` | `[[12,0],[0,12]]` | `1.000000 / 1.000000` | 1.000000 |
| Ordinary-gradient TQK | `1.000000 [1.000000, 1.000000]` | `1.000000 [1.000000, 1.000000]` | `[[12,0],[0,12]]` | `1.000000 / 1.000000` | 1.000000 |
| QNG TQK | `1.000000 [1.000000, 1.000000]` | `1.000000 [1.000000, 1.000000]` | `[[12,0],[0,12]]` | `1.000000 / 1.000000` | 1.000000 |

Prediction and score digests are:

| Method | Prediction SHA-256 | Score SHA-256 |
| --- | --- | --- |
| SNR threshold | `2865fbfc2dd6abe50f1bfbe59cafa8da723b0ae707569ea90175c62ce1b6fdfb` | `970776fe79d61f5f8197052d847f1d57f2290d9e4e13c4230f1d93f5ebdefa49` |
| Circular RBF-SVC | `1b4b828cfd58f797cd99f8dddb0b5c6239ede637d1d04a9d736c8e7dfc87fd38` | `642eb5cda4f63b81ec27b5d718c4f657dedd991b10f2741e5b2e3a1b35ea4d95` |
| Fixed/GD/QNG TQK | `d7eab86bd8c81852888e35c1677613d13f6e7d01b01ffd60498a272995345013` | `220e871cb0960ecefaef2053879c219c03f4b0a312d10eb2628f289323eed90e` |

## Compute accounting and interpretation

The isolated local exact-state operation recorded `9,563.448 ms` total wall
time. Per-method times, including deterministic bootstrap work, were:

| Method | Wall time | TRAIN kernel coordinates | TEST×TRAIN coordinates |
| --- | ---: | ---: | ---: |
| SNR threshold | `1,931.511 ms` | 0 | 0 |
| Circular RBF-SVC | `1,867.938 ms` | 0 | 0 |
| Fixed-theta TQK | `1,930.585 ms` | 1,024 | 768 |
| Ordinary-gradient TQK | `1,950.866 ms` | 1,024 | 768 |
| QNG TQK | `1,878.927 ms` | 1,024 | 768 |

These are local engineering timings, not physical-QPU costs or shot counts.
The simulated task remains strongly separable through SNR and has only 24
independent TEST lineages. The three quantum results are identical because
1D.4a froze the same checkpoint-0 theta and SVC setting for each family, not
because GD or QNG improved it. No winner is declared, TEST results are not used
for model selection, and no quantum-advantage claim is made.

Milestone 1D.5 remains blocked pending separate authorization.
