# AQSE end-to-end research demonstrator — delivery record

## Delivery status

**Status at this documentation snapshot:** functional draft implemented in the
working tree; canonical preparation and final operational acceptance are not
declared complete until the measured fields below are populated from actual
runs.

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

Actual artifact IDs, candidate choices, accepted steps, class supports,
confusions, coverage and metrics are intentionally omitted here until
`make prepare-demo` has completed and the artifacts have been verified.

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
| `make build` | **PENDING FINAL RUN** | see validation record |
| `make test` | **PENDING FINAL RUN** | backend/frontend counts pending |
| `make prepare-demo` | **PENDING CANONICAL RUN** | artifact IDs/metrics pending |
| `make demo` and Compose health | **PENDING FINAL RUN** | backend/frontend state pending |
| `make acceptance` | **PENDING FINAL RUN** | API report pending |
| Browser 1440×900 | **PENDING FINAL RUN** | screenshot/report path pending |
| Browser 390×844 | **PENDING FINAL RUN** | screenshot/report path pending |
| Eight-node 600 s soak | **PENDING FINAL RUN** | RSS/queue/latency report pending |
| Protected hashes | **PENDING FINAL RECHECK** | expected values below |
| Branch push/alignment | **PENDING FINAL PUSH** | local/remote SHA pending |

## Scientific interpretation and limits

Functional completion and predictive performance are separate statements.
The code can be a complete research demonstrator even when measured accuracy is
weak or the AFSE model loses to the raw-feature MLP. The final record must say
whether the quantum→AFSE path helped, tied or hurt; it must not start another
seed/data/hyperparameter search after seeing TEST.

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
- the ten-minute local soak, when complete, demonstrates one machine/run only
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

These values are expectations until the final validation record states the
actual recheck. `main`, tags and protected assets must not be changed by this
delivery.

## Known implementation disclosure

A transient pre-freeze development probe simulated provisional N=3 examples
while diagnosing window selection. It wrote no artifacts, performed no metric
or tuning and is invalidated by the canonical generation domain separator
`aqse-network-demo-v1/canonical-generation/v3`. The disclosure and disjoint
fixture namespace are preserved in the detailed validation record.

## Final acceptance condition

The delivery becomes ready for final manual acceptance and merge review only
after all PENDING rows above have actual evidence, the branch is pushed with a
clean working tree, local/remote HEAD match, and the final scientific result is
reported without post-TEST tuning. No PR or merge is part of this delivery.
