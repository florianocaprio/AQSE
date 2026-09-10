from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.training.evaluation_protocol import build_evaluation_protocol
from app.training.evaluation_storage import (
    load_evaluation_protocol,
    write_evaluation_protocol,
)
from app.training.seal import verify_canonical_test_seal
from app.training.storage import artifact_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the AQSE 1D.4a comparison protocol before VALIDATION access."
    )
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    if os.environ.get("AQSE_ALLOW_TEST_OPEN") == "1":
        raise PermissionError("AQSE_ALLOW_TEST_OPEN must remain disabled for 1D.4a")
    root = (args.artifact_root or artifact_root()).resolve()
    dataset_path = root / "aqse-development-064acca20fc788c6"
    seal_before = verify_canonical_test_seal(dataset_path)
    protocol = build_evaluation_protocol()
    path, execution = write_evaluation_protocol(protocol, root=root)
    loaded, loaded_execution = load_evaluation_protocol(
        path,
        expected_protocol_id=protocol.protocol_id,
    )
    if loaded != protocol or loaded_execution != execution:
        raise RuntimeError("comparative protocol load-back differs from the freeze")
    if verify_canonical_test_seal(dataset_path) != seal_before:
        raise RuntimeError("canonical TEST seal changed during protocol freeze")
    print(
        json.dumps(
            {
                "protocol_id": protocol.protocol_id,
                "content_digest": protocol.content_digest,
                "path": str(path),
                "created_at_utc": execution.created_at_utc,
                "seed_schedule": protocol.seed_schedule,
                "methods": protocol.methods,
                "test_state": seal_before[0].test_state,
                "test_ledger_sha256": seal_before[1],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
