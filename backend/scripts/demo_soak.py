#!/usr/bin/env python3
"""Run a real-wall-clock soak against the live AQSE analysis worker."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = (
    BACKEND_ROOT.parent if BACKEND_ROOT.name == "backend" else BACKEND_ROOT
)
DEFAULT_REPORT_ROOT = Path(
    os.environ.get(
        "AQSE_ARTIFACT_ROOT",
        str(REPOSITORY_ROOT.parent / "AQSE-artifacts"),
    )
) / "validation"


class SoakError(RuntimeError):
    """Raised when the live soak cannot preserve its bounded runtime contract."""


class JsonHttpClient:
    def __init__(self, base_url: str, *, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, object] | None = None,
        expected: tuple[int, ...] = (200,),
    ) -> tuple[int, Any, float]:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        started = time.perf_counter()
        try:
            with urlopen(
                Request(url, method=method, data=data, headers=headers),
                timeout=self.timeout_s,
            ) as response:
                raw = response.read()
                status = response.status
        except HTTPError as exc:
            raw = exc.read()
            status = exc.code
        except (TimeoutError, URLError) as exc:
            raise SoakError(f"{method} {path} failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        try:
            body = json.loads(raw) if raw else None
        except json.JSONDecodeError as exc:
            raise SoakError(f"{method} {path} returned invalid JSON") from exc
        if status not in expected:
            raise SoakError(
                f"{method} {path} returned HTTP {status}; expected {expected}: {body}"
            )
        return status, body, elapsed_ms


def _backend_rss_bytes() -> tuple[int | None, str]:
    """Read aggregate RSS for the long-lived backend processes when available."""

    proc = Path("/proc")
    rss_total = 0
    matched = 0
    if proc.is_dir():
        for process_dir in proc.iterdir():
            if not process_dir.name.isdigit():
                continue
            try:
                command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ")
                is_backend = (
                    b"uvicorn app.main:app" in command
                    or b"multiprocessing.spawn" in command
                    or b"multiprocessing.resource_tracker" in command
                )
                if not is_backend:
                    continue
                resident_pages = int(
                    (process_dir / "statm").read_text(encoding="utf-8").split()[1]
                )
                rss_total += resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
                matched += 1
            except (FileNotFoundError, OSError, ValueError, IndexError):
                continue
    if matched:
        return rss_total, "backend-process-rss-sum"

    candidates = (
        (Path("/sys/fs/cgroup/memory.current"), "cgroup-v2-memory-usage-fallback"),
        (
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            "cgroup-v1-memory-usage-fallback",
        ),
    )
    for path, source in candidates:
        try:
            return int(path.read_text(encoding="utf-8").strip()), source
        except (FileNotFoundError, OSError, ValueError):
            continue
    statm = Path("/proc/self/statm")
    try:
        resident_pages = int(statm.read_text(encoding="utf-8").split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE")), "process-statm-fallback"
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None, "unavailable"


def summarize_rss(samples: list[tuple[float, int]]) -> dict[str, int | float | None]:
    if not samples:
        return {
            "initial_bytes": None,
            "final_bytes": None,
            "minimum_bytes": None,
            "maximum_bytes": None,
            "growth_bytes": None,
            "ols_slope_bytes_per_hour": None,
        }
    times = [sample[0] for sample in samples]
    values = [sample[1] for sample in samples]
    mean_time = sum(times) / len(times)
    mean_value = sum(values) / len(values)
    denominator = sum((value - mean_time) ** 2 for value in times)
    slope_per_second = (
        0.0
        if denominator == 0.0
        else sum(
            (sample_time - mean_time) * (sample_value - mean_value)
            for sample_time, sample_value in samples
        )
        / denominator
    )
    return {
        "initial_bytes": values[0],
        "final_bytes": values[-1],
        "minimum_bytes": min(values),
        "maximum_bytes": max(values),
        "growth_bytes": values[-1] - values[0],
        "ols_slope_bytes_per_hour": slope_per_second * 3_600.0,
    }


def _safe_report_path(report_dir: Path) -> Path:
    resolved = report_dir.expanduser().resolve()
    if resolved == REPOSITORY_ROOT.resolve() or REPOSITORY_ROOT.resolve() in resolved.parents:
        raise SoakError("soak reports must be written outside the Git tree")
    resolved.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return resolved / f"demo-soak-{timestamp}.json"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cleanup(client: JsonHttpClient, session_id: str) -> None:
    client.request(
        "POST",
        f"/api/demo/analysis/{session_id}/stop",
        expected=(200, 404),
    )
    client.request(
        "POST",
        f"/api/network/sessions/{session_id}/stop",
        expected=(200, 404),
    )
    client.request(
        "DELETE",
        f"/api/network/sessions/{session_id}",
        expected=(204, 404),
    )


def run_soak(
    *,
    base_url: str,
    duration_s: float,
    node_count: int,
    seed: int,
    sample_interval_s: float,
) -> dict[str, Any]:
    client = JsonHttpClient(base_url)
    _, registry, _ = client.request("GET", "/api/demo/registry")
    if not registry.get("prepared") or not registry.get("active"):
        raise SoakError(registry.get("preparation_detail", "run make prepare-demo"))
    _, configuration, _ = client.request(
        "GET",
        "/api/network/defaults",
        query={"node_count": node_count},
    )
    configuration.update(
        {
            "session_name": "AQSE real-wall-clock analysis soak",
            "random_seed": seed,
            "sampling_rate_Hz": 100.0,
            "ui_refresh_rate_Hz": 5.0,
            "time_scale": 1.0,
            "buffer_duration_s": 120.0,
        }
    )
    _, created, _ = client.request(
        "POST",
        "/api/network/sessions",
        payload=configuration,
        expected=(201,),
    )
    session_id = created["status"]["session_id"]
    rss_samples: list[tuple[float, int]] = []
    health_latencies_ms: list[float] = []
    queue_depth_max = 0
    state_counts: dict[str, int] = {}
    started_utc = datetime.now(timezone.utc)
    interrupted = False
    latest_analysis: dict[str, Any] | None = None
    latest_session: dict[str, Any] | None = None
    rss_source = "unavailable"
    started = time.monotonic()
    try:
        client.request(
            "POST",
            f"/api/demo/analysis/{session_id}/start",
            payload={"acquire_reference": True},
        )
        client.request("POST", f"/api/network/sessions/{session_id}/start")
        deadline = started + duration_s
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            _, health, health_ms = client.request("GET", "/api/health")
            if health.get("status") != "ok":
                raise SoakError(f"backend health failed during soak: {health}")
            health_latencies_ms.append(health_ms)
            _, latest_analysis, _ = client.request(
                "GET", f"/api/demo/analysis/{session_id}"
            )
            _, latest_session, _ = client.request(
                "GET", f"/api/network/sessions/{session_id}"
            )
            state = latest_analysis["state"]
            state_counts[state] = state_counts.get(state, 0) + 1
            if state == "failed":
                raise SoakError(f"analysis worker failed: {latest_analysis.get('error')}")
            queue_depth_max = max(queue_depth_max, latest_analysis["queue_depth"])
            rss, source = _backend_rss_bytes()
            rss_source = source
            if rss is not None:
                rss_samples.append((now - started, rss))
            time.sleep(min(sample_interval_s, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        try:
            _, latest_analysis, _ = client.request(
                "GET", f"/api/demo/analysis/{session_id}"
            )
            _, latest_session, _ = client.request(
                "GET", f"/api/network/sessions/{session_id}"
            )
        finally:
            _cleanup(client, session_id)
    elapsed_s = time.monotonic() - started
    if latest_analysis is None or latest_session is None:
        raise SoakError("soak ended without runtime observations")
    if not interrupted and elapsed_s + 0.05 < duration_s:
        raise SoakError("soak ended before the requested real-wall-clock duration")
    if not interrupted and latest_analysis["completed_window_count"] == 0:
        raise SoakError("analysis produced no complete causal windows during the soak")
    buffer = latest_session["status"]["buffer"]
    if buffer["size"] > buffer["capacity"] or queue_depth_max > 1:
        raise SoakError("bounded buffer or newest-window queue invariant failed")
    rss_summary = summarize_rss(rss_samples)
    sorted_health = sorted(health_latencies_ms)
    health_p95 = (
        None
        if not sorted_health
        else sorted_health[min(len(sorted_health) - 1, int(len(sorted_health) * 0.95))]
    )
    return {
        "schema_version": "aqse.demo-soak-report.v1",
        "started_at_utc": started_utc.isoformat().replace("+00:00", "Z"),
        "result": "interrupted" if interrupted else "completed",
        "research_runtime_claim": (
            "one local bounded observation; this is not a hard real-time guarantee"
        ),
        "prediction_input_boundary": (
            "observation and analysis REST endpoints only; simulator truth not requested"
        ),
        "requested_duration_s": duration_s,
        "elapsed_wall_s": elapsed_s,
        "node_count": node_count,
        "random_seed": seed,
        "sampling_rate_Hz": 100.0,
        "time_scale": 1.0,
        "session": {
            "frames_generated": latest_session["status"]["latest_frame_id"],
            "simulation_lag_s": latest_session["status"]["simulation_lag_s"],
            "buffer_size": buffer["size"],
            "buffer_capacity": buffer["capacity"],
            "overwritten_frames": buffer["overwritten_frames"],
        },
        "analysis": {
            "state": latest_analysis["state"],
            "state_counts": state_counts,
            "completed_window_count": latest_analysis["completed_window_count"],
            "skipped_window_count": latest_analysis["skipped_window_count"],
            "quality_abstention_count": latest_analysis["quality_abstention_count"],
            "queue_depth_max": queue_depth_max,
            "latency_p50_ms": latest_analysis["latency_p50_ms"],
            "latency_p95_ms": latest_analysis["latency_p95_ms"],
            "result_age_ms": latest_analysis["result_age_ms"],
        },
        "health": {
            "poll_count": len(health_latencies_ms),
            "failure_count": 0,
            "maximum_latency_ms": max(health_latencies_ms, default=None),
            "p95_latency_ms": health_p95,
        },
        "rss": {
            "source": rss_source,
            "sample_count": len(rss_samples),
            **rss_summary,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8_600)
    parser.add_argument("--sample-interval-s", type=float, default=5.0)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    if not 1 <= args.nodes <= 8:
        parser.error("--nodes must be between 1 and 8")
    if args.sample_interval_s <= 0.0:
        parser.error("--sample-interval-s must be positive")
    try:
        report = run_soak(
            base_url=args.base_url,
            duration_s=args.duration_s,
            node_count=args.nodes,
            seed=args.seed,
            sample_interval_s=args.sample_interval_s,
        )
        path = _safe_report_path(args.report_dir)
        _write_report(path, report)
    except (OSError, SoakError, ValueError) as exc:
        print(f"AQSE soak failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"result": report["result"], "report_path": str(path)}, sort_keys=True))
    return 130 if report["result"] == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
