# AQSE end-to-end research demonstrator — delivery record

## Delivery status

**Status at this documentation snapshot:** the bounded end-to-end v1 research
demonstrator is implemented, canonically prepared and operationally validated
on the dedicated branch. Final manual acceptance and merge review remain with
Floriano; no PR or merge is included.

Scientific label: **research / not validated for field deployment**.

This page is the concise hand-off. Detailed commands, metrics, hashes and
acceptance observations belong in
[`validation/end-to-end-demo.md`](validation/end-to-end-demo.md).

## Implemented outcome

The source tree now contains one connected application path:

```text
continuous simulated ObservationFrame
  → frozen observed reference
  → local/network State8
  → TRAIN-fitted 8-coordinate AngleScaler
  → author's protected 8-qubit VQC and fidelity TQK
  → regularised fixed-landmark AFSE
  → TRAIN-fitted compact MLP
  → live score / uncertainty / quality / context status
```

The draft also contains the separate, explicit training path, an immutable
network study, model-selection freeze, one-time TEST ledger, safe bundle
registry and atomic bundle application. The existing eight-workbook GUI reads
these real API payloads; it does not draw placeholder AFSE vectors, classifier
scores or optimizer curves.

### Implemented component identities

| Component | Frozen identity or policy |
| --- | --- |
| Local features | `aqse.local-state8.v1` |
| Network features | `aqse.network-state8.v1` |
| State8 encoding | `aqse.state8.angle-scaler-all8.v1` |
| Study | `aqse-network-demo-v1` |
| AFSE | `aqse.afse.nystrom-ridge32.v1` |
| MLP | `aqse.classical.mlp-32x16-tanh-lbfgs.v1` |
| Local task | `aqse.local-change.v1` |
| Network task | `aqse.network-pattern.v1` |
| Scientific label | `research / not validated for field deployment` |

## Operating recipe

First preparation:

```sh
cp .env.example .env
make prepare-demo
make demo
```

Normal later startup:

```sh
make demo
```

Validation and shutdown:

```sh
make test
make acceptance
make down
```

`make prepare-demo` is the only normal command that may generate the new study,
fit candidates, freeze VALIDATION selection and execute the single authorised
new TEST evaluation. `make demo` loads existing artifacts and does none of
those operations automatically.

## Bounded scientific protocol

- 160 independent 28-second episodes at 100 Hz;
- four scenario strata × node counts 1–8 × five independent replicates;
- fixed split 96 TRAIN / 32 VALIDATION / 32 TEST before simulation;
- observed reference `[0,8)` and primary focal-node window `[18,22)`;
- separate observation and label channels;
- exactly two theta candidates per task: theta0 and at most ten accepted
  protected-QNG updates;
- up to 32 distinct TRAIN QNG rows and up to 32 distinct balanced TRAIN AFSE
  landmarks;
- TRAIN-only fitting; VALIDATION-only candidate choice; one frozen new TEST
  evaluation;
- no historical TEST reopening and no selection change after TEST.

The verified canonical study is
`aqse-network-study-5f33f5c4d856361f`, selection freeze
`aqse-demo-freeze-39c61760d233f694` and final evaluation
`aqse-demo-final-ad5fb1055eb68ead`. Local selection used the protected-QNG
candidate after 10 accepted steps; network selection used theta0 under the
predeclared tie rule. The detailed record contains the exact theta vectors,
supports, confusions, intervals and artifact hashes.

## Operational behavior

- N=1 uses the local change task and disclaims causal attribution.
- N=2 uses the local task and explicitly marks attribution ambiguity.
- N≥3 uses the network task only with valid aligned peer context; otherwise it
  degrades locally or abstains.
- Invalid reference, incomplete/missing/stuck/clipped observations,
  unreliable pose, undefined correlations and non-finite features cannot enter
  quantum inference.
- The worker keeps a bounded 1,600-frame buffer and a newest-window queue depth
  of at most one; it reports skips, age and latency.
- Training and inference share one heavy exact-state admission slot. Sensor
  acquisition remains live while inference reports a paused/busy state.
- Applying a compatible local/network selection changes the complete bundle
  atomically and invalidates cached/pending results.
- Backend/frontend restart reloads persistent registry artifacts; in-memory
  sensor sessions are intentionally recreated.

## Verification summary

This table must contain only final commands actually run against the delivery
candidate. `PENDING` is not a pass.

| Verification | Result | Evidence |
| --- | --- | --- |
| `make build` | PASS | native `linux/arm64` images |
| `make test` | PASS | backend 369/369; Ruff; TypeScript; ESLint; frontend 47/47; Vite build |
| `make prepare-demo` | PASS and idempotent rerun | canonical study/freeze/final IDs above; no TEST reopen on rerun |
| `make demo` and Compose health | PASS | backend/frontend healthy; persisted active pair restored after restart |
| `make acceptance` | PASS | 182 HTTP calls; N=1/2/4/8, events, abstention, replay, training and bundle controls |
| Browser 1440×900 | PASS | live four-node observation and connected model output; all worksheets opened; clean console |
| Browser 390×844 | PASS | all eight worksheets at 390 px after one responsive correction; clean console |
| Eight-node 600 s soak | PASS for the bounded runtime invariants | 600.206 s; 588/0 complete/skipped windows; queue max 0; health 120/120; RSS growth disclosed |
| Protected hashes | PASS | all five expected values below matched exactly |
| Branch push/alignment | Deferred to post-commit delivery report | exact final SHA and 0/0 alignment are reported after push |

## Scientific interpretation and limits

Functional completion and predictive performance are separate statements.
The connected demonstrator is functional, but the quantum→AFSE path **hurt**
both held-out TEST comparisons after helping on VALIDATION:

- local TEST: BA 0.5833 and macro-F1 0.5897 versus raw 0.6875/0.6952;
- network TEST: BA 0.7083 and macro-F1 0.7009 versus raw 0.7500/0.7565.

Both paired delta intervals include zero. NORMAL recall was 0.25 local and
0.3333 network, and the NORMAL replay false-positive episode rate was 1.0 for
both tasks. These negative findings were preserved without another seed,
threshold, model or hyperparameter search and without changing the frozen
selection.

The following limits remain regardless of measured scores:

- the four network labels express pattern compatibility inside a simulator,
  not guaranteed physical causation;
- a shared device offset and common field change can be observationally
  equivalent;
- one sensor cannot identify environment versus device and two sensors cannot
  identify the faulty side from discrepancy alone;
- model scores are not calibrated probabilities;
- OOD uses a TRAIN residual p99 heuristic, not a calibrated guarantee;
- AFSE is a classical Nyström map built from a quantum kernel, not a new
  quantum algorithm or physical sensor state;
- exact statevector execution is not a QPU and establishes no hardware speed,
  sensitivity or quantum advantage;
- the study is small, simulated and evaluated at the independent-episode
  level; degenerate bootstrap intervals do not prove zero uncertainty;
- the completed ten-minute local soak demonstrates one machine/run only
  and does not guarantee hard real-time behavior.

## Protected evidence

Expected byte-identical hashes:

| Asset | SHA-256 |
| --- | --- |
| `tqk8.py` | `cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689` |
| `sampler_qng.py` | `7489c5c2b3eb0783499c333dc0826994b146d7cf5631469a0520bac9570b2ea6` |
| original TQK8 test | `c87c56c0eb3f1ea20d2dd124427aea96f868ae4674b5b390e2688e9da8bb0c09` |
| original notebook | `9d7e0337af7c039894fc6c53567de2883063b32c89e368753bd01e1d9e9e82be` |
| historical TEST ledger | `e0d4898141d8be07f4a4f1af7582791eae61994ced52bb7b3dba30a7ddda4b70` |

All values matched on the final recheck. `main`, tags and protected assets were
not changed by this delivery.

## Known implementation disclosure

A transient pre-freeze development probe simulated provisional N=3 examples
while diagnosing window selection. It wrote no artifacts, performed no metric
or tuning and is invalidated by the canonical generation domain separator
`aqse-network-demo-v1/canonical-generation/v3`. The disclosure and disjoint
fixture namespace are preserved in the detailed validation record.

## Final acceptance condition

The delivery is ready for final manual acceptance and merge review after the
final evidence commit is pushed, the local/remote SHA equality and clean status
are reported, and the negative scientific result above remains unchanged. No
PR or merge is part of this delivery.
