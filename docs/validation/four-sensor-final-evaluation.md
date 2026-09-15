# AQSE four-sensor held-out evaluation

Date: 2026-09-15
Study: `aqse-four-sensor-study-v1`
Status: completed; TEST ledger closed
Scientific scope: simulation-only research evidence; not field validation

## Objective and frozen order

This study evaluates the already-frozen AQSE network model with exactly four
magnetometer nodes. It does not train, select, tune, or refit a model.

The operations were completed in the following irreversible order:

1. generate and publish a 20-episode noncanonical quality pilot;
2. accept the pilot only for structural and measurement-quality checks;
3. publish the protocol freeze, including the exact existing model binding,
   metrics, thresholds, geometry schedule, random seeds, and TEST access plan;
4. generate and publish 100 new held-out episodes;
5. publish an explicit one-run authorization;
6. open observations and labels once, evaluate the frozen model, publish the
   result, and close the ledger.

The final ledger is:

`sealed -> observations_opened -> labels_opened -> evaluation_published`

## Design

- Four classes, 25 independent episodes each:
  `NORMAL`, `ENVIRONMENT_COMPATIBLE`, `DEVICE_COMPATIBLE`, and
  `MIXED_OR_AMBIGUOUS`.
- Four planar geometries, 25 episodes each: square, rectangle, rotated square,
  and irregular quadrilateral.
- Four focal sensors, 25 episodes each: S1 through S4.
- Every scenario contains all four geometries and all four focal sensors, with
  6/7 occupancy because 25 is not divisible by four.
- Every scenario contains all 16 geometry/focal combinations.
- Sampling: 100 Hz for 28 s; reference `[0, 8)` s; injected event `[14, 24)` s;
  primary State8 window `[18, 22)` s.
- The pilot and final corpus use separate generation domains, seeds, episode
  identities, and generative lineages. Pilot rows are not reusable in TEST.

## Frozen model

| Item | Frozen value |
|---|---|
| Selection freeze | `aqse-demo-freeze-39c61760d233f694` |
| Network bundle | `aqse-demo-bundle-cd888ecfb28b2145` |
| Bundle digest | `cd888ecfb28b2145a1df484ff4f2367d92934807a8da9a98b57fe8bbb67a50fd` |
| Task | `aqse.network-pattern.v1` |
| State8 profile | `aqse.network-state8.v1` |
| Theta candidate | `theta0` |
| Accepted QNG updates | `0` |

Consequently, this run evaluates the frozen TQK/AFSE pipeline with its selected
deterministic initial theta. It is **not** evidence for a QNG-trained theta.

## Artifact identities

| Artifact | Identity |
|---|---|
| Pilot | `aqse-four-sensor-pilot-99390dbc150f4559` |
| Pilot assessment digest | `493dcbc589d39d93055ec9a62cb2eda79fc184d422f43712fe21d60b6febf480` |
| Protocol freeze | `aqse-four-sensor-protocol-freeze-33983fd9cb41e8f2` |
| Final TEST corpus | `aqse-four-sensor-final-test-61fa2ebad1099658` |
| Model binding | `aqse-four-sensor-model-binding-b7c04b9ee9e4c302` |
| TEST authorization | `aqse-four-sensor-test-authorization-3c1b979d46121ea0` |
| Scientific evaluation | `aqse-four-sensor-final-037ff015340ff45f` |
| Evaluation artifact | `aqse-four-sensor-final-evaluation-c823a6a31fe4f0d4` |

## Primary results

Intervals are predeclared 95% episode-level stratified percentile-bootstrap
intervals with 2,000 replicates.

| Method | Balanced accuracy (95% CI) | Macro-F1 (95% CI) | Coverage |
|---|---:|---:|---:|
| Frozen quantum AFSE + MLP | 0.630 (0.550–0.710) | 0.621 (0.528–0.706) | 0.900 |
| Raw State8 + same MLP | 0.550 (0.480–0.620) | 0.466 (0.403–0.528) | 1.000 |
| Observable p99 rule | 0.630 (0.560–0.700) | 0.591 (0.494–0.681) | 1.000 |

The quantum pipeline improves balanced accuracy over the raw MLP by `+0.080`,
but its paired 95% interval is `[0.000, 0.160]`. The lower bound does not exceed
zero, so the predeclared quantum-advantage rule is not met. Its macro-F1 gain
over the raw MLP is `+0.154`, with paired interval `[0.064, 0.234]`.

Against the observable p99 rule, the balanced-accuracy difference is `0.000`
with interval `[-0.120, 0.110]`.

## Per-class results for the frozen quantum pipeline

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.500 | 0.720 | 0.590 | 25 |
| ENVIRONMENT_COMPATIBLE | 0.500 | 0.520 | 0.510 | 25 |
| DEVICE_COMPATIBLE | 0.821 | 0.920 | 0.868 | 25 |
| MIXED_OR_AMBIGUOUS | 0.900 | 0.360 | 0.514 | 25 |

Confusion matrix (rows are truth; columns use the class order shown above):

| Truth / prediction | NORMAL | ENVIRONMENT | DEVICE | MIXED |
|---|---:|---:|---:|---:|
| NORMAL | 18 | 7 | 0 | 0 |
| ENVIRONMENT | 12 | 13 | 0 | 0 |
| DEVICE | 1 | 0 | 23 | 1 |
| MIXED | 5 | 6 | 5 | 9 |

The principal errors are not random: environment-compatible episodes are often
collapsed into NORMAL, and mixed episodes are split across three simpler
classes. Device-compatible episodes are the only class that already exceeds
the frozen recall threshold.

## Geometry and focal-sensor robustness

| Geometry | Quantum balanced accuracy | Raw balanced accuracy | N |
|---|---:|---:|---:|
| Square | 0.756 | 0.548 | 25 |
| Rectangle | 0.649 | 0.565 | 25 |
| Irregular | 0.655 | 0.542 | 25 |
| Rotated square | 0.464 | 0.542 | 25 |

| Focal sensor | Quantum balanced accuracy | Raw balanced accuracy | N |
|---|---:|---:|---:|
| S1 | 0.673 | 0.548 | 25 |
| S2 | 0.613 | 0.542 | 25 |
| S3 | 0.631 | 0.589 | 25 |
| S4 | 0.601 | 0.524 | 25 |

The focal-sensor spread is moderate. Geometry dependence is much stronger: the
rotated square is the clearest robustness failure and the ordinary square is
the only geometry above the nominal 0.75 balanced-accuracy threshold. These
subgroup estimates have only 25 episodes each and should be treated as
diagnostic rather than definitive rankings.

There is also a design limitation in this first frozen protocol: within the
`DEVICE_COMPATIBLE` class, device mode is determined by `replicate mod 4`, while
geometry follows another deterministic replicate schedule. Device mode and
geometry are therefore not fully independent. The geometry table is suitable
for hypothesis generation, but it cannot support a causal claim that geometry
alone produced the observed difference.

## Frozen success criteria

| Criterion | Result |
|---|---|
| Balanced accuracy >= 0.75 | Failed |
| Macro-F1 >= 0.75 | Failed |
| Every class recall >= 0.70 | Failed |
| NORMAL recall >= 0.80 | Failed |
| Coverage >= 0.90 | Passed (boundary value) |
| NORMAL false-positive episode rate <= 0.20 | Failed (0.28) |
| Quantum advantage over raw baseline | Not demonstrated |

## Scientific conclusion

The experiment is internally auditable and supports one useful conclusion:
the existing frozen model is not sufficiently robust for the predeclared
four-sensor operating domain. The point estimate and the complete primary 95%
interval are below the 0.75 target. This is evidence against declaring the
current model validated under this simulation protocol, not evidence that the
architecture can never work.

The result does show a promising representation effect relative to the raw
same-architecture MLP, especially in macro-F1 and in recovering some mixed
episodes that the raw model never recognizes. However, the primary paired
balanced-accuracy interval touches zero, coverage is lower, NORMAL recall is
lower, and a simple observable rule ties the quantum pipeline in balanced
accuracy. A quantum advantage therefore cannot be claimed.

The quantum-versus-raw comparison is a system-level comparison rather than a
causal ablation of the quantum circuit: the quantum path also includes the AFSE
representation, while the reference model consumes raw State8 values. In
addition, the predeclared accuracy metrics use the model's underlying class
prediction and report abstention separately as coverage. A post-hoc descriptive
check of the already-published output finds 59 correct results among the 90
non-abstained episodes (0.656 selective accuracy); this was not a confirmatory
metric and must not be promoted to one after TEST access.

## Recommended next decision

Do **not** add more identically distributed held-out episodes to this already
opened study. More repetitions would mainly narrow an estimate that is already
below the frozen target, while repeated inspection would weaken the clean-test
interpretation.

The defensible next step is a new, separately versioned development study:

1. create TRAIN/VALIDATION data restricted to the same four-sensor geometries;
2. address geometry invariance and the `ENVIRONMENT`/`MIXED` separation using
   only TRAIN/VALIDATION;
3. include stronger frozen classical comparators (for example an RBF-SVC) and
   report multiple training seeds;
4. compare `theta0` and the protected QNG candidate without changing TQK8;
5. counterbalance device mode independently of geometry;
6. freeze the improved model and then evaluate once on a new independent TEST
   corpus, never reusing these 100 episodes for selection.

Any analyses suggested by the observed rotated-square weakness must be labelled
post-hoc and may guide development only; they cannot be used to reinterpret this
TEST as confirmatory evidence.

## Execution timing

| Stage | Wall time |
|---|---:|
| Pilot generation and publication | 15.47 s |
| Protocol freeze publication | 0.07 s |
| Final TEST generation and publication | 80.61 s |
| Binding and authorization | 0.08 s |
| Single TEST access and evaluation | 1.27 s |
| Evaluation publication | 0.12 s |
| Total | 97.63 s |
