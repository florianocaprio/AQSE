from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

BENCHMARK_INTENT = {
    "schema_version": "aqse.qng-execution-intent.v1",
    "intent_id": "aqse-1d3-g4-real-load-benchmark",
    "purpose": "engineering_benchmark",
}


def _request(
    base_url: str,
    method: str,
    path: str,
    payload: object | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    return {
        "status": status,
        "latency_ms": elapsed_ms,
        "body": json.loads(body) if body else None,
    }


def _feature_request() -> dict[str, object]:
    sampling_rate = 100.0
    time_s = [index / sampling_rate for index in range(400)]
    signal = [
        25.0 + 8.0 * math.sin(2.0 * math.pi * 8.0 * value)
        for value in time_s
    ]
    return {
        "series": {
            "acquisition_id": "g4-responsiveness-acquisition",
            "sensor_id": "g4-responsiveness-sensor",
            "sampling_rate_hz": sampling_rate,
            "time_s": time_s,
            "measured_field": [[value, 2.0, 0.0] for value in signal],
            "field_unit": "nT",
            "temperature_k": [293.15 + 0.2 * value for value in time_s],
            "saturation_mask": [[False, False, False] for _ in time_s],
        },
        "channel": "x",
        "window": {"duration_s": 1.0, "overlap_fraction": 0.5},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the one AQSE 1D.3 G4 real-load responsiveness benchmark."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--artifact-root", type=Path, default=Path("/artifacts"))
    arguments = parser.parse_args()
    base_url = arguments.base_url.rstrip("/")
    run_root = arguments.artifact_root.resolve() / "training-runs"
    runs_before = sorted(path.name for path in run_root.iterdir() if path.is_dir())

    extracted = _request(
        base_url,
        "POST",
        "/api/features/vector-magnetometer/extract",
        _feature_request(),
    )
    if extracted["status"] != 200:
        raise RuntimeError(f"feature preparation failed: {extracted}")
    feature_body = extracted["body"]
    preview_payload = {
        "mode": "self_reference",
        "backend": "numpy",
        "feature_profile": feature_body["profile"],
        "reference_dataset_id": "g4-responsiveness-reference",
        "reference_windows": feature_body["windows"][:2],
        "query_windows": [],
        "theta": [0.0] * 16,
    }
    defaults = _request(base_url, "GET", "/api/network/defaults?node_count=1")
    session = _request(base_url, "POST", "/api/network/sessions", defaults["body"])
    if session["status"] != 201:
        raise RuntimeError(f"network session preparation failed: {session}")
    session_id = session["body"]["status"]["session_id"]

    try:
        started = _request(
            base_url,
            "POST",
            "/api/training/benchmarks/responsiveness",
            BENCHMARK_INTENT,
        )
        if started["status"] != 201:
            raise RuntimeError(f"benchmark was not newly admitted: {started}")
        deadline = time.monotonic() + 10.0
        while True:
            status = _request(
                base_url,
                "GET",
                "/api/training/benchmarks/responsiveness",
            )
            state = status["body"]["state"]
            if state == "RUNNING":
                break
            if state in {"COMPLETED", "FAILED"} or time.monotonic() >= deadline:
                raise RuntimeError(
                    f"benchmark did not expose a RUNNING contention window: {status}"
                )
            time.sleep(0.01)

        probes = {
            "backend_health": ("GET", "/api/health", None),
            "quantum_health": ("GET", "/api/quantum/health", None),
            "network_health": ("GET", "/api/network/health", None),
            "simulator_step": (
                "POST",
                f"/api/network/sessions/{session_id}/step",
                {"frames": 1},
            ),
            "diagnostics_conflict": ("POST", "/api/quantum/diagnostics", None),
            "preview_conflict": ("POST", "/api/quantum/preview", preview_payload),
        }
        with ThreadPoolExecutor(max_workers=len(probes)) as pool:
            futures = {
                name: pool.submit(_request, base_url, method, path, payload)
                for name, (method, path, payload) in probes.items()
            }
            measured = {name: future.result() for name, future in futures.items()}

        expected_status = {
            "backend_health": 200,
            "quantum_health": 200,
            "network_health": 200,
            "simulator_step": 200,
            "diagnostics_conflict": 429,
            "preview_conflict": 429,
        }
        for name, expected in expected_status.items():
            if measured[name]["status"] != expected:
                raise RuntimeError(
                    f"unexpected {name} status: {measured[name]['status']} != {expected}"
                )

        while True:
            terminal = _request(
                base_url,
                "GET",
                "/api/training/benchmarks/responsiveness",
            )
            if terminal["body"]["state"] in {"COMPLETED", "FAILED"}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("engineering benchmark did not finish")
            time.sleep(0.05)
        if terminal["body"]["state"] != "COMPLETED":
            raise RuntimeError(f"engineering benchmark failed: {terminal}")

        runs_after = sorted(path.name for path in run_root.iterdir() if path.is_dir())
        if runs_after != runs_before:
            raise RuntimeError("engineering benchmark changed candidate run artifacts")
        print(
            json.dumps(
                {
                    "intent": BENCHMARK_INTENT,
                    "candidate_run_directories_unchanged": True,
                    "probes": {
                        name: {
                            "status": result["status"],
                            "latency_ms": result["latency_ms"],
                        }
                        for name, result in measured.items()
                    },
                    "benchmark": terminal["body"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        _request(base_url, "DELETE", f"/api/network/sessions/{session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
