from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.training.evaluation_storage import (
    load_comparative_evaluation,
    load_evaluation_protocol,
    load_model_selection_freeze,
)
from app.training.held_out import (
    EVALUATION_ID,
    FREEZE_ID,
    INITIAL_LEDGER_SHA256,
    PROTOCOL_ID,
    build_g5_authorization,
)
from app.training.held_out_storage import (
    load_g5_authorization,
    read_test_ledger_opaque,
    write_g5_authorization,
)
from app.training.storage import artifact_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish the immutable AQSE 1D.4b G5 TEST authorization."
    )
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    if os.environ.get("AQSE_ALLOW_TEST_OPEN") == "1":
        raise PermissionError("G5 authorization must be published before TEST is enabled")
    root = (args.artifact_root or artifact_root()).resolve()
    dataset_path = root / "aqse-development-064acca20fc788c6"
    ledger = read_test_ledger_opaque(dataset_path)
    if len(ledger.entries) != 1 or ledger.file_sha256 != INITIAL_LEDGER_SHA256:
        raise RuntimeError("G5 authorization requires the exact sealed genesis ledger")
    if (root / "test-banks").exists() or (root / "final-held-out-evaluations").exists():
        raise RuntimeError("held-out artifacts already exist before G5 authorization")
    protocol, _ = load_evaluation_protocol(
        root / "evaluation-protocols" / PROTOCOL_ID,
        expected_protocol_id=PROTOCOL_ID,
    )
    evaluation, _ = load_comparative_evaluation(
        root / "comparative-evaluations" / EVALUATION_ID,
        expected_evaluation_id=EVALUATION_ID,
    )
    freeze, _ = load_model_selection_freeze(
        root / "model-selection-freezes" / FREEZE_ID,
        expected_freeze_id=FREEZE_ID,
    )
    authorization = build_g5_authorization(
        protocol,
        evaluation,
        freeze,
        initial_ledger_sha256=ledger.file_sha256,
    )
    path, execution = write_g5_authorization(authorization, root=root)
    loaded, loaded_execution = load_g5_authorization(
        path,
        expected_authorization_id=authorization.authorization_id,
    )
    if loaded != authorization or loaded_execution != execution:
        raise RuntimeError("G5 authorization load-back differs from publication")
    if read_test_ledger_opaque(dataset_path) != ledger:
        raise RuntimeError("G5 authorization unexpectedly changed the TEST ledger")
    print(
        json.dumps(
            {
                "authorization_id": authorization.authorization_id,
                "content_digest": authorization.content_digest,
                "path": str(path),
                "entry_commit": authorization.entry_commit,
                "gate_state": authorization.gate_state,
                "initial_ledger_sha256": ledger.file_sha256,
                "initial_ledger_event_count": len(ledger.entries),
                "contains_test_values": authorization.contains_test_values,
                "expected_semantic_loads": [
                    item.model_dump(mode="json")
                    for item in authorization.expected_semantic_loads
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
