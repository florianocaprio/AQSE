from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.training.generation import (  # noqa: E402
    build_development_dataset,
    run_pilot,
)
from app.training.storage import artifact_root, write_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the sealed AQSE Milestone 1D.1 pilot and development artifacts."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Override AQSE_ARTIFACT_ROOT for this offline run.",
    )
    arguments = parser.parse_args()
    root = arguments.artifact_root.resolve() if arguments.artifact_root else artifact_root()

    try:
        pilot_path, pilot_manifest, pilot_execution = write_dataset(run_pilot(), root=root)
    except Exception as exc:
        print(f"Pilot failed; development generation was not started: {exc}", file=sys.stderr)
        return 1

    development_path, development_manifest, development_execution = write_dataset(
        build_development_dataset(), root=root
    )
    report = {
        "artifact_root": str(root),
        "pilot": {
            "path": str(pilot_path),
            "dataset_id": pilot_manifest.dataset_id,
            "scientific_digest": pilot_manifest.scientific_digest,
            "feature_profile_fingerprint": pilot_manifest.feature_profile_fingerprint,
            "partitions": [
                item.model_dump(mode="json") for item in pilot_manifest.partition_summaries
            ],
            "duration_ms": pilot_execution.duration_ms,
            "peak_memory_bytes": pilot_execution.peak_memory_bytes,
            "total_bytes": pilot_execution.total_bytes,
        },
        "development": {
            "path": str(development_path),
            "dataset_id": development_manifest.dataset_id,
            "scientific_digest": development_manifest.scientific_digest,
            "feature_profile_fingerprint": development_manifest.feature_profile_fingerprint,
            "test_state": development_manifest.test_state,
            "partitions": [
                item.model_dump(mode="json") for item in development_manifest.partition_summaries
            ],
            "duration_ms": development_execution.duration_ms,
            "peak_memory_bytes": development_execution.peak_memory_bytes,
            "total_bytes": development_execution.total_bytes,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
