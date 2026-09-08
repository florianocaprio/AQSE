# Milestone 1C Validation Record

## Record identity

- Date: 2026-09-08
- Branch: `codex/milestone-1c-aqse-workbench`
- Protected baseline: `29a2b93`
- Host: macOS with Docker Desktop
- Container architecture: native `aarch64`
- Commit created by this work: none

This record covers the implemented Milestone 1C sensor workbench and the
compatible subset of the vertical 1–8 magnetometer-network specification. It
does not promote architecture-only components to scientific implementations.

## Automated validation

| Command | Result |
| --- | --- |
| `docker compose build` | PASS; backend and frontend images built natively |
| `docker compose run --rm backend python -m pytest` | PASS; 136 tests |
| `docker compose run --rm backend ruff check .` | PASS |
| `docker compose run --rm --no-deps frontend sh -c "CI=true pnpm install --frozen-lockfile && pnpm run typecheck && pnpm run lint && pnpm run test && pnpm run build"` | PASS |
| `pnpm run typecheck` | PASS |
| `pnpm run lint` | PASS; zero ESLint warnings |
| `pnpm run test` | PASS; 26 tests in 7 files |
| `pnpm run build` | PASS; 650 modules transformed |
| `git diff --check` | PASS |

Pytest emits one dependency deprecation warning from Starlette's use of the
AnyIO `BlockingPortal` alias. The production frontend build emits one
non-blocking Vite warning because its single JavaScript chunk is approximately
718.21 kB before gzip and 209.73 kB after gzip.

The first Docker frontend check used an image and named dependency volume from
before Vitest was added, so TypeScript could not resolve the `vitest` module.
The images were rebuilt and dependency installation was made explicit,
frozen-lockfile, and non-interactive in `make test` and the development service.
The complete gate then passed.

Browser QA initially exposed repeated Recharts zero-size warnings caused by
mounting charts in hidden worksheets. The application now mounts only the
active worksheet; a fresh browser session with visible charts produced no
console warning or error.

The final release audit also exposed two frontend lifecycle/provenance issues.
Reset and replay now always renew the SSE subscription after success so its
cursor cannot remain ahead of the backend's reset frame sequence. Stream
actions carry a revision token, stale queued batches are rejected, and a new
revision starts explicitly from frame zero. Experiment
records now retain immutable request snapshots, while blind mode prevents those
potentially latent scenario inputs from being rendered or exported. Dedicated
regression tests cover the SSE decision, deep snapshot detachment, single
ledger emission, and the blind-mode disclosure policy.

## Docker and endpoint verification

Both services reached Docker `healthy` state with loopback-only bindings:

- frontend: `127.0.0.1:3000 -> 3000/tcp`;
- backend: `127.0.0.1:8000 -> 8000/tcp`.

The following requests returned the expected status:

| Request | HTTP result | Evidence |
| --- | ---: | --- |
| `GET /api/health` | 200 | `status=ok`, `service=AQSE Backend` |
| `GET /api/quantum/health` | 200 | Qiskit and NumPy exact-state paths ready |
| `GET /api/sensors` | 200 | legacy magnetometer remains listed |
| `GET /api/sensors/magnetometer/defaults` | 200 | legacy configuration returned |
| `POST /api/sensors/magnetometer/simulate` | 200 | 400 samples and the original 8-feature result |
| `GET /api/sensors/vector-magnetometer/scenarios` | 200 | six finite-vector scenarios |
| `GET /api/sensors/vector-magnetometer/defaults` | 200 | finite-vector configuration returned |
| `POST /api/sensors/vector-magnetometer/simulate` | 200 | 200 samples with separate generator truth |
| `GET /api/workbench/capabilities` | 200 | implemented and pending boundaries returned |
| `GET /api/quantum/circuit` | 200 | 8 qubits, 8 features, 16 theta parameters |
| `GET /api/network/health` | 200 | bounded in-memory store and SSE reported |
| `GET /api/network/presets` | 200 | 1/2/4/8-node and causal demo presets returned |
| `GET /api/network/field-providers` | 200 | synthetic providers ready; WMM unavailable |
| `GET /` on the frontend | 200 | Vite application served |

The manual container end-to-end path used the explicit 5 Hz
`quantum_preview_signal` preset and returned:

- session creation 201 and deletion 204;
- 250 observation frames and 250 truth frames from separate endpoints;
- no truth fields or active causes in observation frames;
- four complete feature windows, all eligible for quantum preview;
- a 4 x 4 `qiskit_statevector` fidelity kernel;
- an unchanged 16-value executed theta snapshot;
- the scope label `Fixed-theta infrastructure preview; no QNG training`.

An exploratory call using `harmonic_reference` as a finite-vector scenario was
correctly rejected with HTTP 422 because that identifier belongs to the
continuous-network validation preset, not the six finite-vector scenarios. The
finite-vector verification was repeated with `combined_stress` and passed.

## Continuous-network soak

Executed command:

```sh
docker compose run --rm backend python scripts/network_soak.py \
  --duration-s 1200 \
  --nodes 8 \
  --seed 42 \
  --sampling-rate-hz 100 \
  --ui-refresh-rate-hz 5 \
  --time-scale 1
```

Measured result:

| Metric | Value |
| --- | ---: |
| Elapsed wall time | 1200.0107 s |
| Frames generated | 120001 |
| Effective rate | 99.999945 Hz |
| Final simulation lag | 0.0 s |
| Nodes | 8 |
| Buffer size / capacity | 12000 / 12000 frames |
| Overwritten frames | 108001 |
| Initial RSS | 54,964,224 bytes |
| Final RSS | 690,397,184 bytes |
| Maximum sampled RSS | 689,922,048 bytes |
| Average process CPU | 18.1899% |
| Logical CPUs | 12 |
| Python | 3.12.14 |
| Container platform | Linux aarch64 |

The retained buffer reached its configured bound and older frames were
overwritten. Memory therefore includes the full 120-second, eight-node buffer;
approximately 690 MB is a measured local-demonstrator cost and an explicit
future optimization target, not an unbounded-growth claim.

## Browser QA

The real Dockerized application was exercised in English at 1440 x 900,
1280 x 800, and 390 x 844. The narrow and desktop layouts had matching
`clientWidth` and `scrollWidth`, so no document-level horizontal overflow was
present.

Verified interactions:

1. load the `Quantum preview harmonic signal` preset;
2. create, start, stream, pause, stop, and delete a network session;
3. display observed node signals, network overlay, node difference, and the
   separate truth validation channel;
4. extract twenty valid causal feature windows from observed S1 data;
5. display all eight canonical values with units and the quality ledger;
6. edit theta 0 to `0.4` and theta 8 to `-0.3`;
7. run a 16 x 16 Qiskit fixed-theta preview and inspect its real heatmap,
   diagnostics, scaler snapshot, raw values, and encoded angles;
8. verify QNG, AFSE, Neural, and output surfaces remain honestly unavailable or
   architecture-only;
9. verify blind mode hides request snapshots and disables scenario and
   experiment-ledger exports containing latent inputs;
10. verify a fresh session with visible Recharts charts has no browser console
    warning or error.

A final clean-browser regression exercised `create -> step -> reset -> step` in
blind mode. The renewed SSE subscription delivered the new frame `1`
immediately, remained `CONNECTED`, and produced four distinct provenance
records (create, step, reset, step) without rendering the stored request
snapshots. The test session was then deleted and `/api/network/health` returned
`active_sessions=0`.

The executed 16 x 16 preview reported a minimum fidelity of `0.158942`, maximum
of `1.0`, maximum diagonal deviation `2.2204e-15`, maximum symmetry deviation
`2.2204e-16`, minimum eigenvalue `0.0793613`, and backend duration `12.9292 ms`.
These are infrastructure-preview measurements, not model-performance metrics.

## Protected scientific source

`git diff --exit-code HEAD` reported no change for the author-owned TQK8/QNG
sources and original quantum test. Their SHA-256 values remain:

| File | SHA-256 |
| --- | --- |
| `backend/app/quantum/user_pipeline/tqk8.py` | `cef11f0d0617e5e12b2904c0aa65868ef655a853db48a99c73a803440d371689` |
| `backend/app/quantum/user_pipeline/sampler_qng.py` | `7489c5c2b3eb0783499c333dc0826994b146d7cf5631469a0520bac9570b2ea6` |
| `backend/tests/quantum/test_tqk8.py` | `c87c56c0eb3f1ea20d2dd124427aea96f868ae4674b5b390e2688e9da8bb0c09` |

The original notebook, TQK8 README, and supplied TQK8 validation artifacts also
have no diff from the protected baseline.

## Explicit limitations and deferred vertical scope

- QNG source exists but sensor-integrated loss, training, cancellation,
  checkpoint promotion, and theta update are not connected.
- AFSE mathematics, neural/classical task models, diagnosis, localization,
  tracking, uncertainty, correlation matrix, continuous-network PSD, coverage
  map, and comparative performance evaluation are not implemented.
- Circular and user-defined trajectories, recovery transients, quantization,
  physical transport delay, shot/noisy simulation, and physical QPU execution
  remain future contracts.
- World Magnetic Model support is an unavailable adapter boundary; no
  coefficients or location-accuracy claim are bundled.
- Blind mode is a local presentation barrier, not an authorization or API
  security boundary.
- Browser page-lifecycle session deletion is best-effort after a browser crash
  or abrupt host shutdown; the registry remains bounded to sixteen sessions.
- No claim of quantum advantage, hardware sensitivity improvement, diagnosis
  accuracy, or experimental instrument fidelity is made.

These omissions preserve the approved Milestone 1C boundary and prevent the
vertical specification from silently introducing unapproved scientific logic.
