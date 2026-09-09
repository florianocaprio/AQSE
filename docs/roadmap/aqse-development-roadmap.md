# AQSE Development Roadmap

## Document status

This document is the canonical development sequence for AQSE. It records what
the repository currently implements, what is being designed, and which future
increments still require explicit scientific approval. A roadmap entry is not
authorization to implement it, and the presence of a worksheet, DTO, protocol,
or placeholder does not make a scientific stage complete.

Repository state verified at the start of Milestone 1D:

- repository: `florianocaprio/AQSE`;
- current milestone branch: `codex/milestone-1d-tqk-training`;
- branch HEAD at the approved 1D.3 entry gate:
  `ab5bff99fa9779a143e236595c0d4c1e78ff349f`;
- underlying `main` baseline: `1cb07cdccabf9c655a63f2b23aab37383ae90b69`;
- `main` and `origin/main` also resolve to `1cb07cdccabf9c655a63f2b23aab37383ae90b69`;
- the `milestone-1c` tag remains at the peeled commit
  `95c5483c0192ba7605713c428e02b2527ab1f919`;
- commit `1cb07cd` is the post-tag infrastructure reliability patch that
  separates lightweight health checks from explicitly requested quantum
  diagnostics. It is part of the 1D branch baseline, but it does not alter the
  Milestone 1C scientific scope.

## Milestone sequence and approval status

| Milestone | Scope | Verifiable outcome | Status and authority |
| --- | --- | --- | --- |
| **1A** | Author-supplied TQK8 scientific engine | Original 8-qubit VQC, fidelity kernel, alignment loss, numerical derivatives, Fubini--Study metric, and QNG implementation are present with their original tests | **Complete baseline** |
| **1B** | Initial magnetometer simulator | Scalar finite simulation and the legacy eight-feature harmonic profile are available | **Complete baseline** |
| **1C** | Vector/network workbench | Causal 1--8-node simulator, observation/truth separation, feature windows, quality/provenance controls, and bounded fixed-theta kernel preview are implemented | **Complete, merged, and tagged** at `95c5483` |
| **1D** | Controlled datasets and TQK training validation | Reproducible labelled experiments, compatible encoding, bounded training, held-out comparison, checkpoints, and controlled workbench integration | **Current milestone; 1D.3 corrective consolidation implemented and awaiting final G4 review** |
| **1E** | Local Functional Embedding / AFSE | A mathematically approved, fixed-size, versioned local representation | **Future and unapproved** |
| **1F** | Classical model | A frozen classical model consuming the approved AFSE representation | **Future and unapproved** |
| **1G** | Continuous inference and network analysis | Causal predictions, latency/queue accounting, network-level evaluation, and controlled model promotion | **Future and unapproved** |

Physical sensors, a second sensor technology, noisy or shot-based quantum
simulation, IBM Runtime, and physical-QPU execution are separate future
validation tracks. They are not implied by completion of the local simulator.

## Milestone 1D increments

Milestone 1D is deliberately incremental. Completion of one increment does not
authorize the next one.

| Increment | Intended content | Exit condition | Current authority |
| --- | --- | --- | --- |
| **1D.0** | Scientific contract, compatibility audit, baseline validation, and isolated non-production diagnostic probes | Evidence and open decisions are reviewed; Floriano gives an explicit scientific decision on the proposed task, phase treatment, data independence, budgets, and boundaries | **Completed decision gate** |
| **1D.1** | Immutable experiment datasets, independent label channel, durable lineage, exclusions, and grouped train/validation/test splits | Deterministic regeneration, replay deduplication, raw-interval isolation, and group isolation are demonstrated | **Approved and canonically frozen as archive v2** |
| **1D.2** | Approved input encoding and phase policy, fitted-preprocessing boundary, compatibility rules, and baseline preparation | Periodicity, seam continuity, train-only fitting, and incompatible-artifact rejection are demonstrated | **Scientifically approved; Gate G3 closed** |
| **1D.3** | Bounded training jobs using the unchanged author-supplied QNG implementation | Wrapper equivalence, deterministic trajectory identity, execution-intent idempotence, real accepted-step history, cancellation, bounded concurrency, and resource limits are validated | **Corrective consolidation implemented; awaiting final G4 review** |
| **1D.4** | Held-out evaluation and matched classical/quantum comparisons | Model selection is frozen before test opening; negative and inconclusive results are preserved | **Planned; not authorized** |
| **1D.5** | Workbench controls, checkpoint persistence, and explicit application of compatible trained theta | No automatic training or promotion; stale results and downstream invalidation are enforced | **Planned; not authorized** |
| **1D.6** | Consolidation, regression, local acceptance, and external review | Full validation passes and a separately authorized review/merge decision is made | **Planned; not authorized** |

Through 1D.3 one explicit, bounded TRAIN-only job endpoint and one designated
immutable candidate-theta trajectory exist. A deterministic scientific
trajectory identity maps the designated execution and the retained accidental
non-canonical duplicate without rewriting either artifact. Durable versioned
execution intents prevent replay from starting a second worker. A separately
typed, predeclared real-load engineering benchmark validates responsiveness
without publishing a candidate artifact. There is still no active model,
held-out result, validation-selected theta, AFSE mathematics, classical/neural
model, GUI training control or automatic checkpoint application. The
phase-direct encoder and candidate theta are not connected to the Milestone 1C
preview.

## Training and inference are separate paths

### Controlled training path

```text
labelled training observations
        -> causal feature extraction
        -> training-fitted encoding/scaler
        -> VQC(theta)
        -> TQK Gram matrix
        -> centred-alignment loss
        -> supplied QNG optimizer
        -> immutable candidate-theta run artifact
```

Labels are inputs to the training objective only. Truth, simulator seeds,
scenario names, hidden fault state, and future samples do not enter feature
extraction, encoding, or the VQC as predictor inputs. QNG updates theta; it does
not label samples, tune sensor hardware, or form an inference layer.

### Frozen inference path

```text
measured observation
        -> frozen causal feature profile
        -> frozen compatible encoding/scaler
        -> frozen VQC(theta*) and TQK
        -> future approved AFSE representation
        -> future frozen classical model
        -> prediction with lineage and validity status
```

The current Milestone 1C path stops at an explicit fixed-theta fidelity-kernel
preview. A Gram matrix is relational batch geometry, not AFSE, and the existing
demonstration SVC is a possible research evaluator rather than the final AQSE
classical model. The implemented 1D.3 output is a versioned candidate-theta run
artifact, not a fabricated local embedding or deployed prediction.

## Proposed first 1D experiment

The leading proposal for scientific approval is a narrow binary experiment:
discriminate nominal versus elevated simulated device-noise conditions under a
controlled harmonic excitation, using labels `-1` and `+1` generated separately
from declared simulator interventions.

This would validate the dataset, training, and comparison machinery. It would
not prove universal environmental-versus-instrumental diagnosis or quantum
advantage. Amplitude, frequency, phase, temperature, and other nuisances must
vary independently of class within a declared domain. An observationally
equivalent physical-versus-instrumental counterexample must be retained as a
negative control: identical predictive observations must produce identical
features, kernels, and predictions even when hidden truth labels differ.

The proposal, label semantics, signal/noise ranges, eligibility policy, and
comparison budget remain decisions for the 1D.0 approval gate. The candidate
continuous-network relational feature profile must not be silently activated to
make the task easier.

## Scientific and compatibility rules

### Protected scientific contract

- Preserve the author-supplied `tqk8.py`, `sampler_qng.py`, original quantum
  tests, fixtures, notebooks, and scientific artifacts byte-for-byte unless
  Floriano explicitly authorizes a scientific change.
- Preserve the actual TQK8 contract: eight ordered inputs, eight qubits,
  sixteen theta parameters, seven CZ gates, RY upload and later RZ re-upload,
  fidelity kernel, centred-alignment loss, empirical Fubini--Study metric, and
  the existing QNG step semantics.
- Reuse the supplied loss, gradient/metric, and QNG implementation through an
  application wrapper. Do not reconstruct or silently replace the mathematics.
- Preserve the validated sensor-response order, noise/filter/clipping order,
  deterministic random streams, SI-unit boundaries, causal acquisition,
  replay semantics, and strict observation/truth separation.

### Versioned compatibility tuple

Every dataset, trained kernel, representation, and model must record enough
lineage to reject incompatible combinations. At minimum this includes:

```text
full_git_sha
observation_schema_version + physical_units
calibration_and_pose_provenance
dataset_id + generative_lineage_id + split_policy_version
feature_profile_id + extractor_version + feature_order
encoding_policy_id
scaler_id + scaler_snapshot + fitting_dataset_id
vqc_tqk_source_hashes + quantum_backend_semantics
theta_version + initial/final/selected_theta
reference_dataset_or_bank_id
future_afse_method_and_version
future_downstream_model_version
```

Missing or incompatible fields make an artifact stale; components must not be
silently combined. In particular:

- the implemented feature order remains
  `[amplitude, phase, frequency, variance, drift, snr, spectral_peak, temperature]`;
- length eight alone does not make different feature profiles compatible;
- scaling and reference-bank fitting use training data only;
- query batch membership or order must not refit or change frozen inference;
- deterministic replays, overlapping windows, synchronized sensors, and paired
  variants sharing one generative realization remain in the same data split;
- changing a feature profile, phase policy, scaler, theta, VQC/TQK semantics,
  reference bank, AFSE method, or downstream model creates a new incompatible
  version and invalidates dependent cached artifacts;
- datasets and checkpoints are generated artifacts and remain outside Git.

### Phase-policy boundary

The legacy preview retains the unchanged `AngleScaler` behavior. The approved
offline 1D.2 policy retains the original scaler outputs for seven columns and replaces
the encoded phase column with a canonically wrapped observed phase in
`[-pi, pi)`. It is versioned as `aqse.tqk8.encoding.phase-direct.v1`; actual
NumPy/Qiskit state and kernel tests demonstrate periodicity, seam continuity
and cross-engine agreement at absolute tolerance `1e-12`. It does not overwrite
legacy semantics or add a ninth quantum input.

## Stage gates

| Gate | Required evidence or decision | What remains blocked |
| --- | --- | --- |
| **G0 — 1D.0 entry** | Clean dedicated branch, verified baseline/tag distinction, protected hashes, and existing baseline validation | All production training work |
| **G1 — scientific approval** | Floriano explicitly approves the first task, labels, observational limits, phase policy direction, provenance strategy, and resource budgets after reviewing 1D.0 evidence | 1D.1--1D.6 |
| **G2 — dataset freeze** | Dataset manifest, independent labels, grouped splits, duplicate/raw-overlap checks, eligibility reporting, and sealed test policy pass review | Production encoding and optimization |
| **G3 — representation compatibility** | Approved encoding is periodic where required, fitted on training only, versioned, and rejects incompatible artifacts | Training jobs and checkpoints |
| **G4 — bounded training** | Unchanged QNG wrapper equivalence, deterministic trajectory identity, execution-intent idempotence, concurrency/cancellation behavior, deterministic initialization, actual history, and real-load compute/memory/responsiveness limits are validated | Evaluation claims and UI application |
| **G5 — model selection freeze** | Classical and quantum comparisons use matched information and declared budgets; validation selects the checkpoint before the test set is opened | Held-out claims and model promotion |
| **G6 — controlled integration** | Explicit checkpoint application, stale-result invalidation, artifact lineage, simulator responsiveness, and honest UI labels are validated | Release or merge |
| **G7 — external acceptance** | Full regression, local build, manual acceptance, external scientific/code review, and separate merge authorization | Merge into `main`, release, or subsequent milestone |

No acceptance gate requires QNG or the quantum kernel to outperform a classical
baseline. Negative or inconclusive results are valid scientific outcomes.

## Git and review workflow

1. Use `codex/milestone-1d-tqk-training` for all separately authorized 1D
   increments. Its verified starting point includes infrastructure commit
   `1cb07cd`; do not move the `milestone-1c` tag from `95c5483`.
2. Begin each increment from a clean working tree, inspect ancestry and remote
   state, and isolate only that increment's intentional files.
3. Run the relevant backend, frontend, lint, typecheck, build, scientific
   regression, hash, and `git diff --check` validations before any commit.
4. Commit and push one reviewable increment only after that increment has been
   explicitly authorized and validated. Use a normal fast-forward push.
5. Do not rebase or rewrite published history, amend published commits,
   force-push, move tags, merge into `main`, create a release, or open a pull
   request without separate authorization.
6. Keep secrets, `.env` files, generated datasets/checkpoints, runtime files,
   caches, logs, ZIP archives, temporary probes, and build artifacts out of
   Git.
7. Completion of each authorized increment ends at its review gate. Work must
   stop before the next increment until Floriano explicitly authorizes it.

The final milestone workflow remains: push the reviewed branch, external code
and scientific review, local build, manual acceptance, and only then a
separately authorized merge. A successful test run never implies automatic
promotion.

## Related contracts

- [Milestone 1D scientific training plan](../training/milestone-1d-scientific-plan.md)
- [Milestone 1D.2 phase-direct encoding](../training/milestone-1d-2-encoding.md)
- [Milestone 1D.2 validation record](../validation/milestone-1d-2-encoding.md)
- [Milestone 1D.3 bounded QNG training](../training/milestone-1d-3-qng-training.md)
- [Milestone 1D.3 validation record](../validation/milestone-1d-3-qng-training.md)
- [Milestone 1D.0 design and compatibility audit](../validation/milestone-1d-design-audit.md)
- [Canonical AQSE pipeline](../architecture/canonical-aqse-pipeline.md)
- [Scientific scope and limitations](../architecture/scientific-scope-and-limitations.md)
- [Feature profiles and quality](../features/feature-profiles-and-quality.md)
- [Quantum preview contract](../quantum/quantum-preview-contract.md)
- [AFSE boundary](../quantum/afse-boundary.md)
- [Milestone 1C validation record](../validation/milestone-1c.md)
