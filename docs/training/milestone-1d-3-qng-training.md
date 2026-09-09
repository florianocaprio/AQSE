# AQSE Milestone 1D.3 — Bounded QNG training contract

## Scope and status

Milestone 1D.3 implements one explicitly invoked, bounded QNG optimization
path over the frozen TRAIN quantum bank. It delegates every optimizer update
to the unchanged author-supplied `fit_qng` implementation. The result is an
immutable candidate-theta trajectory outside Git; it is not a promoted model,
an active workbench checkpoint, held-out evaluation, or evidence of quantum
advantage.

The increment is implemented on `codex/milestone-1d-tqk-training` and awaits
external scientific review. Gate G4 is not declared closed by this document.
Milestone 1D.4, validation-based model selection, TEST opening, AFSE, neural
processing, physical-QPU work and GUI integration remain blocked.

## Frozen scientific inputs

| Item | Frozen value |
| --- | --- |
| Dataset | `aqse-development-064acca20fc788c6` |
| Dataset digest | `064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7` |
| Feature-profile fingerprint | `cc6f91470912cfc25025bb6190673f67cd3f56fc8265a7f4da311ad3c0eaeb17` |
| Encoder | `aqse-encoder-f9cf4bc12a767410` |
| Encoder digest | `f9cf4bc12a767410c8759dd01c5bb232ffc3877269af3ed219f3b6484db14e51` |
| Scaler | `aqse-angle-scaler-fc1d44f00f4cc1be` |
| Encoding policy | `aqse.tqk8.encoding.phase-direct.v1` |
| TRAIN bank | `aqse-train-bank-32b5f93897676535` |
| TRAIN bank digest | `32b5f938976765355f0cd2886bee88082a693a06816f2bae25b9ec268973d8dd` |
| Optimization matrix | exactly `32 x 8`, 32 distinct lineages |
| Labels | exactly 32 TRAIN labels, balanced 16/16 as `-1/+1` |

Assembly follows the frozen bank order and joins observations and labels by
episode, window and lineage identity. It loads only the TRAIN observation and
label partitions. Generation plans, simulator seeds, configured noise, hidden
causes, VALIDATION labels and TEST content cannot enter `X_train`.

## Initialization and optimizer protocol

The initialization policy is `aqse.theta-init.uniform-v1`: NumPy
`default_rng(1001005)` draws exactly 16 independent values from
`Uniform(-0.8, 0.8)`. The resulting immutable coordinate is:

```text
[-0.012546994208710416, -0.25396301734270066,
 -0.0587826243874503,   -0.3918649792046132,
 -0.14525278571459332,  -0.4976652971644949,
 -0.36525990994512353,  -0.39147835650363344,
  0.4166832520401502,    0.40065220107971666,
  0.019719475214645676,  -0.7446722630591751,
 -0.03960170131429952,  -0.13783836459272913,
  0.599190349075206,     0.6055058504024162]
```

The canonical backend is the exact NumPy statevector engine. Qiskit is used
only for the bounded cross-engine preflight. The protected optimizer contract
is fixed to 10 maximum accepted updates, learning rate `0.2`, damping `0.001`,
maximum step norm `0.4`, and the original Armijo/backtracking behavior.

The wrapper invokes `fit_qng(..., steps=1, verbose=False)` between cooperative
cancellation points. It records a step only when the protected function
returns its real history row. An empty history stops honestly with
`NO_ACCEPTED_UPDATE`; theta is never clamped, wrapped, normalized or silently
retried.

The future matched-seed schedule reserved for 1D.4 is
`1001005, 1001006, 1001007, 1001008, 1001009`. It is documented only and was
not executed by this increment.

## Preflight gates

The canonical 32-row TRAIN bank passed the mandatory `N=3` equivalence gate:

| Measurement | Result | Tolerance |
| --- | ---: | ---: |
| Accepted updates | 3 versus 3 | exact |
| Final theta maximum absolute delta | `0.0` | `1e-12` |
| Complete history maximum absolute delta | `0.0` | `1e-12` |

The same bank and theta0 passed the NumPy/Qiskit one-step gate:

| Measurement | Maximum absolute delta | Tolerance |
| --- | ---: | ---: |
| Initial fidelity Gram matrix | `1.5543122344752192e-15` | `1e-12` |
| Initial alignment loss | `0.0` | `1e-10` |
| Alignment gradient | `2.7755575615628914e-17` | `1e-10` |
| Empirical FS metric | `5.551115123125783e-17` | `1e-10` |
| First candidate theta | `5.551115123125783e-17` | `1e-10` |

These are exact-simulator consistency checks, not physical-QPU measurements.

## Runtime and artifact contracts

`POST /api/training/jobs` admits the one frozen job contract without accepting
browser-supplied arrays or hyperparameters. Status and cooperative cancellation
are explicit job-ID operations. The bounded process-local registry retains at
most 16 records and exposes `CREATED`, `RUNNING`, `COMPLETED`, `CANCELLED` and
`FAILED`. No active model or automatic theta application exists.

Training, quantum preview and explicit diagnostics share one non-blocking
heavy-quantum admission slot. Conflicting heavy work receives HTTP 429.
`/api/health`, `/api/quantum/health` and sensor-network execution do not acquire
that slot and remain independent of training. Docker health invokes only
`/api/health`.

Run artifacts are bounded JSON written below the configured artifact root.
Publication stages and validates `run.json`, `execution.json` and a digest
manifest, makes the payload read-only, then atomically renames it. Existing
identities cannot be overwritten. Loading rejects unexpected files, symlinks,
size/digest changes, source or compatibility mismatches.

## Scientific boundary

The recorded loss is TRAIN alignment only. A decrease along this trajectory
does not establish predictive accuracy, sensor sensitivity, generalization,
model selection or quantum advantage. VALIDATION labels were not loaded and
TEST remains semantically unopened. The final theta is a candidate coordinate
from one frozen optimization trajectory and is not active in the workbench.

Measured evidence and the run-history record are maintained separately in
[the Milestone 1D.3 validation record](../validation/milestone-1d-3-qng-training.md).
