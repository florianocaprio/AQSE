# Milestone 1B Validation Record

- Date: 2026-09-08
- Branch: `codex/milestone-1b`
- Result: validation complete and ready for formal freeze

## Validated scope

- Deterministic quantum-magnetometer time-series simulation.
- Separate raw acquisition, spectral preprocessing and 8D feature extraction.
- Sensor catalog, defaults and simulation REST endpoints.
- English sensor configuration and result-inspection dashboard.
- Magnetic-field and peak-preserving frequency-spectrum plots.
- Sixteen browser-local TQK8 parameter controls with no execution connection.
- Existing TQK8 implementation and scientific tests unchanged.

No sensor feature is normalized, sent to the quantum engine, persisted or used
for training. This milestone adds no QPU access, database, queue, GPU dependency
or WebSocket transport.

## Automated verification

| Command | Result |
| --- | --- |
| `pnpm install --no-frozen-lockfile` in `frontend/` | lockfile regenerated; Recharts 3.10.1 installed |
| `pnpm run build` in `frontend/` | TypeScript and Vite production build passed; 620 modules transformed |
| `python3 -m compileall -q backend/app backend/tests` | passed |
| `docker compose config --quiet` | passed |
| `make build` | backend and frontend images built successfully |
| `make test` | 39/39 backend tests passed; frontend production build passed |
| `make up` | backend and frontend started successfully |
| `git diff --check` | passed with no whitespace errors |
| `git diff --exit-code HEAD -- backend/app/quantum backend/tests/quantum docs/quantum docs/validation/tqk8 docs/notebooks` | passed; no scientific-quantum changes |

The backend suite ran in Linux on Python 3.12.14 and included all eight existing
`tests/quantum/test_tqk8.py` cases. The only pytest warning was an upstream
Starlette use of a deprecated AnyIO alias.

The frontend production bundle passed. Vite reported a non-blocking chunk-size
warning for the Recharts-containing JavaScript bundle (approximately 571 kB
minified, 172 kB gzip). This is acceptable for the local demonstrator and does
not affect correctness.

## Runtime verification

Both Compose services reported `healthy`:

| Service | Port | Result |
| --- | ---: | --- |
| Backend | `8000` | healthy |
| Frontend | `3000` | healthy |

Docker image inspection reported native `arm64/linux` images for both services;
no AMD64 platform override is present.

The following checks returned HTTP 200:

- `GET http://localhost:8000/api/health`
- `GET http://localhost:8000/api/quantum/health`
- `GET http://localhost:8000/api/sensors`
- `GET http://localhost:8000/api/sensors/magnetometer/defaults`
- `POST http://localhost:8000/api/sensors/magnetometer/simulate`
- `GET http://localhost:8000/docs`
- `GET http://localhost:3000`
- `GET http://localhost:3000/api/health` through the Vite proxy

## Browser verification

The Docker-served frontend was tested in the in-app browser. The verified flow
covered:

1. backend, TQK8 and Qiskit statevector statuses displayed as `READY`;
2. default simulation producing 400 raw samples;
3. two rendered charts and exactly eight ordered feature cards;
4. parameter edits marking prior results as stale;
5. a second simulation replacing the stale result;
6. all 16 theta controls present, editable and resettable;
7. theta values remaining a `LOCAL DRAFT` with no execution action;
8. no browser console errors or warnings.

The final services remain running for local inspection.
