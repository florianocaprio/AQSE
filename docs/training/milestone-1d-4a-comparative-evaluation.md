# AQSE Milestone 1D.4a — VALIDATION comparison and selection freeze

## Scope and authority

Gate G4 is closed. Milestone 1D.4a performs the authorized comparative
evaluation on the frozen TRAIN and VALIDATION partitions and freezes one
configuration per approved method. It does not open TEST, create a TEST bank,
promote a model, refit after selection, implement AFSE, or connect a selected
theta to the workbench.

The comparison protocol was published immutably at
`2026-09-09T19:34:20.192557Z`, before the first semantic access to VALIDATION
labels:

- protocol ID: `aqse-comparative-protocol-a671952c35f0ffe8`;
- digest: `a671952c35f0ffe80707a4f6fed48ff3ff17c679a3392ce4c1fc697f087eff9d`;
- comparison input: `aqse-comparison-input-0def6d051fdecdf8`;
- TRAIN: 32 fixed central windows from 32 independent lineages, 16/16 labels;
- VALIDATION: 24 fixed central windows from 24 independent lineages, 12/12 labels;
- TRAIN and VALIDATION lineage sets are disjoint.

The frozen 1D.2 encoder is reused without refitting. The QNG primary seed
reuses the designated 1D.3 historical run rather than rerunning it.

## Predeclared methods and budgets

The five methods receive the approved observable eight-feature information.
The classical RBF method represents phase as sine/cosine, producing nine
coordinates without adding information, and fits its standardization on TRAIN
only.

| Method | Frozen training and evaluator budget |
| --- | --- |
| SNR threshold | Predict elevated when SNR is at or below a TRAIN-selected midpoint; maximize TRAIN balanced accuracy then macro-F1, then choose the lowest threshold |
| Circular RBF-SVC | `C ∈ {0.1, 1, 10}` and `gamma ∈ {0.01, 0.1, 1}` |
| Fixed-theta TQK | Five frozen theta initializations and precomputed-kernel SVC `C ∈ {0.1, 1, 10}` |
| Ordinary-gradient TQK | Five seeds, checkpoints 0–10, 10 full-batch updates, learning rate `0.2`, maximum step norm `0.4`, no line search, protected alignment gradient |
| QNG TQK | Five seeds, checkpoints 0–10, protected `fit_qng`, learning rate `0.2`, damping `0.001`, maximum step norm `0.4`, unchanged Armijo behavior |

The matched seed schedule is exactly `1001005, 1001006, 1001007, 1001008,
1001009`. The exact NumPy statevector backend is used for quantum training and
kernel evaluation. The protected circuit, loss, gradient, Fubini--Study metric
and QNG implementation are unchanged.

## Selection rule

VALIDATION selects one candidate independently within each method family. The
predeclared ordering is:

1. maximize VALIDATION balanced accuracy;
2. maximize VALIDATION macro-F1;
3. choose the earliest checkpoint;
4. choose the lowest SVC `C`;
5. choose the lowest RBF `gamma`;
6. choose the lowest seed.

The evaluation contains all 355 candidates; no candidate was added or removed
after viewing VALIDATION. Confidence intervals use 2,000 deterministic,
stratified bootstrap resamples of independent VALIDATION lineages with seed
`1001010`. They are descriptive selection-set intervals, not TEST confidence
claims.

## Frozen selections

| Method | Selected configuration | VALIDATION balanced accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| SNR threshold | threshold `11.433898484324114` | 1.000000 | 1.000000 |
| Circular RBF-SVC | `C=10`, `gamma=0.01` | 0.875000 | 0.873016 |
| Fixed-theta TQK | seed `1001008`, checkpoint 0, `C=1` | 0.958333 | 0.958261 |
| Ordinary-gradient TQK | seed `1001008`, checkpoint 0, `C=1` | 0.958333 | 0.958261 |
| QNG TQK | seed `1001008`, checkpoint 0, `C=1` | 0.958333 | 0.958261 |

Checkpoint 0 is the pre-update initialization. Its selection for both GD and
QNG is a measured negative result: within the frozen budget, later trained
checkpoints did not improve the primary VALIDATION score over the selected
initialization. The three quantum selections are therefore the same numerical
theta/evaluator configuration under different method-family candidate spaces.
They must not be presented as evidence that QNG improved predictive accuracy.

The perfect SNR result demonstrates that this simulated first task is strongly
and directly separable by its task-specific observable. It is useful for
pipeline calibration but weak evidence for a quantum representation advantage.

## Final evaluation procedure frozen for later review

Gate G5 remains open. If and only if a separate external review closes it, the
future final evaluation will:

1. create one TEST bank from all 24 TEST lineages using fixed window ordinal 9;
2. reconstruct each selected model from TRAIN only, with no post-VALIDATION refit;
3. evaluate each of the five frozen selections exactly once;
4. report balanced accuracy, macro-F1, confusion matrix, per-class recall,
   AUC only when both classes and valid scores exist, eligibility/abstention,
   lineage bootstrap intervals and actual compute cost;
5. retain negative or inconclusive results and predeclare no winner.

This procedure is metadata only. No TEST observation, label, feature, quality
statistic, bank, kernel, prediction or metric exists in Milestone 1D.4a.
