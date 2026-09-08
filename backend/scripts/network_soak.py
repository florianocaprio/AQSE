#!/usr/bin/env python3
"""Run the AQSE network simulator continuously and emit bounded-resource metrics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.network.defaults import default_network_configuration  # noqa: E402
from app.network.models import NetworkSessionConfiguration  # noqa: E402
from app.network.session import NetworkSession  # noqa: E402


def _maximum_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _current_rss_bytes() -> int:
    """Return current resident memory where the host exposes it.

    Docker's Linux runtime provides ``/proc``.  On other supported local hosts,
    fall back to the process high-water mark rather than adding a dependency
    solely for validation telemetry.
    """

    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text(encoding="utf-8").split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    return _maximum_rss_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=1_200.0)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-rate-hz", type=float, default=100.0)
    parser.add_argument("--ui-refresh-rate-hz", type=float, default=5.0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    if not 1 <= args.nodes <= 8:
        parser.error("--nodes must be between 1 and 8")

    base_configuration = default_network_configuration(args.nodes)
    configuration = NetworkSessionConfiguration.model_validate(
        {
            **base_configuration.model_dump(),
            "random_seed": args.seed,
            "sampling_rate_Hz": args.sampling_rate_hz,
            "ui_refresh_rate_Hz": args.ui_refresh_rate_hz,
            "time_scale": args.time_scale,
        }
    )
    session = NetworkSession(configuration)
    initial_rss_bytes = _current_rss_bytes()
    started = time.monotonic()
    process_cpu_started = time.process_time()
    session.start()
    try:
        while time.monotonic() - started < args.duration_s:
            remaining_s = max(0.0, args.duration_s - (time.monotonic() - started))
            time.sleep(min(1.0, remaining_s))
    except KeyboardInterrupt:
        pass
    finally:
        status = session.stop().status

    elapsed_s = max(time.monotonic() - started, 1.0e-12)
    process_cpu_s = max(time.process_time() - process_cpu_started, 0.0)
    metrics = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "node_count": args.nodes,
        "random_seed": configuration.random_seed,
        "sampling_rate_Hz": configuration.sampling_rate_Hz,
        "ui_refresh_rate_Hz": configuration.ui_refresh_rate_Hz,
        "time_scale": configuration.time_scale,
        "requested_duration_s": args.duration_s,
        "elapsed_wall_s": elapsed_s,
        "process_cpu_s": process_cpu_s,
        "average_cpu_percent": process_cpu_s / elapsed_s * 100.0,
        "frames_generated": status.latest_frame_id,
        "effective_frame_rate_Hz": status.latest_frame_id / elapsed_s,
        "simulation_lag_s": status.simulation_lag_s,
        "buffer_size": status.buffer.size,
        "buffer_capacity": status.buffer.capacity,
        "overwritten_frames": status.buffer.overwritten_frames,
        "initial_rss_bytes": initial_rss_bytes,
        "final_rss_bytes": _current_rss_bytes(),
        "maximum_rss_bytes": _maximum_rss_bytes(),
    }
    print(json.dumps(metrics, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
