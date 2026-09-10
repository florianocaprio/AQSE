# AQSE Milestone 1D.2 — Phase-direct training encoding

## Scope and status

Milestone 1D.2 implements a versioned, offline encoding boundary for the
canonical development dataset. It also freezes row-reference banks for later
training and validation. It does not train theta, calculate alignment loss,
fit a classifier, open the held-out test partition, add a production API, or
connect this encoding to the Milestone 1C fixed-theta preview.

The only accepted scientific input is:

- dataset ID `aqse-development-064acca20fc788c6`;
- scientific digest
  `064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7`;
- feature-profile fingerprint
  `cc6f91470912cfc25025bb6190673f67cd3f56fc8265a7f4da311ad3c0eaeb17`.

The builder requires this dataset path explicitly. It performs no latest-file
or directory-order discovery, and therefore cannot silently choose a pilot,
historical v1 archive, or non-canonical artifact.

## Approved encoding policy

The raw order remains exactly:

```text
[amplitude, phase, frequency, variance, drift, snr,
 spectral_peak, temperature]
```

Policy `aqse.tqk8.encoding.phase-direct.v1` delegates first to the protected
`AngleScaler.transform()` fitted on all eligible TRAIN windows. It preserves
that output exactly for indices `[0, 2, 3, 4, 5, 6, 7]` and replaces only index
1 with:

```text
((phase + pi) mod (2*pi)) - pi
```

The canonical interval is `[-pi, pi)`, including the representation
`+pi -> -pi`. The stored phase mean and scale document the protected scaler
fit, but `phase_scaler_statistics_used=false` for the active output phase
coordinate.

The phase reference remains `window_start`. Canonical wrapping makes the
coordinate circularly consistent for the VQC; it does not synchronize sensor
clocks and does not create a global phase reference.

## Train-only fit and compatibility

`AngleScaler.fit()` is imported from the author-supplied TQK8 source and is
called once on every quantum-eligible TRAIN window. Transform calls are pure
with respect to the frozen mean and scale. Pilot, validation, test, query rows,
and the bounded training bank never enter fitting.

Before transformation, the input contract must match the artifact exactly:

- canonical dataset ID and scientific digest;
- complete feature-profile fingerprint and payload, including extractor,
  feature order and units, channel, sampling, window/hop geometry, quality
  policy and `window_start` phase origin;
- encoding policy and scaler artifact IDs;
- protected TQK8 source identity and the 8-input, 8-qubit, 16-parameter
  circuit contract.

Mismatch is rejected rather than coerced. The legacy preview encoding remains
unchanged and incompatible with the phase-direct space.

## Immutable artifacts

The encoder is stored as bounded canonical JSON under the configured artifact
root, normally `../AQSE-artifacts/encoders/`. Scientific identity is separated
from creation time, runtime versions and repository availability. Files are
written into a private staging directory, verified, made read-only and then
atomically renamed. Existing artifact IDs are never overwritten. Pickle,
executable serialization, browser storage and theta are not used.

The canonical materialization produced:

- fitting population `aqse-fitting-population-7e0d9449d2d571f9`:
  72 TRAIN episodes and 1,368 eligible windows;
- scaler `aqse-angle-scaler-fc1d44f00f4cc1be`;
- encoder `aqse-encoder-f9cf4bc12a767410`;
- encoder digest
  `f9cf4bc12a767410c8759dd01c5bb232ffc3877269af3ed219f3b6484db14e51`.

Operational Git provenance is explicitly `repository_base_sha=unrecorded` and
`repository_dirty=null` because Git was unavailable inside the materialization
container. Effective source hashes remain part of the scientific artifact.

## Frozen quantum banks

The TRAIN bank uses seed `1001004`, a separate deterministic permutation for
each label class, and fixed zero-based window ordinal 9. It contains 32
distinct episodes/lineages: 16 nominal and 16 elevated. Labels are used only
to select the balanced membership and are not copied into row references.
Feature magnitudes, SNR, kernels, loss and validation results are not selection
inputs.

The VALIDATION bank contains all 24 validation lineages, also at ordinal 9,
without using labels to choose membership. In either bank an ineligible fixed
window causes failure; no nearby window is substituted.

- TRAIN bank `aqse-train-bank-32b5f93897676535`, digest
  `32b5f938976765355f0cd2886bee88082a693a06816f2bae25b9ec268973d8dd`;
- VALIDATION bank `aqse-validation-bank-105467fa5a4a5cc7`, digest
  `105467fa5a4a5cc7c867f7f77d92eaa61d231949e7c2356e8f3817c541c0f047`.

There is no TEST bank in 1D.2. A future, explicitly authorized 1D.4 opening may
apply the same fixed central-window rule once per test lineage. Until then the
test ledger must remain untouched.

## Offline command

With the artifact root mounted as configured by Docker Compose:

```sh
docker compose run --rm backend python scripts/build_training_encoding.py \
  --dataset-path /artifacts/aqse-development-064acca20fc788c6 \
  --artifact-root /artifacts
```

The command checks dataset identity and the one-event sealed TEST ledger before
and after every permitted TRAIN/VALIDATION semantic operation. Re-running it
against existing immutable IDs fails rather than overwriting them.

## Later classical baseline boundary

A future matched classical comparator may consume the same observable raw
information, represent phase as `sin(phase), cos(phase)` or an equivalent
circular distance, and fit preprocessing on TRAIN only. It must retain the
same episode/group partitions and explicitly distinguish the 32-row quantum
training bank from any richer full-TRAIN classical comparison. No classical
model is fitted or evaluated in 1D.2.

QNG execution, theta initialization or updates, checkpoints, held-out metrics,
AFSE, neural models and workbench training controls remain unimplemented and
require later, explicit authorization.
