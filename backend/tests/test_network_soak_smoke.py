from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_network_soak_script_smoke() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(backend_root / "scripts" / "network_soak.py"),
            "--duration-s",
            "0.05",
            "--nodes",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    metrics = json.loads(completed.stdout)

    assert metrics["node_count"] == 2
    assert metrics["random_seed"] == 42
    assert metrics["sampling_rate_Hz"] == 100.0
    assert metrics["ui_refresh_rate_Hz"] == 5.0
    assert metrics["time_scale"] == 1.0
    assert metrics["frames_generated"] >= 1
    assert metrics["buffer_size"] <= metrics["buffer_capacity"]
    assert metrics["initial_rss_bytes"] > 0
    assert metrics["final_rss_bytes"] > 0
    assert metrics["maximum_rss_bytes"] > 0
    assert metrics["process_cpu_s"] >= 0.0
    assert metrics["average_cpu_percent"] >= 0.0
