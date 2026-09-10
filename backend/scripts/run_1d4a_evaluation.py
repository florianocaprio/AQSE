from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.training.canonical import file_sha256
from app.training.evaluation import build_model_selection_freeze, evaluate_comparison
from app.training.evaluation_assembly import assemble_comparison_input
from app.training.evaluation_protocol import build_evaluation_protocol
from app.training.evaluation_storage import (
    load_comparative_evaluation,
    load_evaluation_protocol,
    load_model_selection_freeze,
    write_comparative_evaluation,
    write_model_selection_freeze,
)
from app.training.run_storage import load_training_run_artifact
from app.training.seal import verify_canonical_test_seal
from app.training.storage import artifact_root

PRIMARY_QNG_RUN_ID = "aqse-qng-run-f00c702ad790df2b"
TRAINING_INPUT_ID = "aqse-training-input-e853259fac7c0eba"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen AQSE 1D.4a TRAIN/VALIDATION comparison once."
    )
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    if os.environ.get("AQSE_ALLOW_TEST_OPEN") == "1":
        raise PermissionError("AQSE_ALLOW_TEST_OPEN must remain disabled for 1D.4a")
    root = (args.artifact_root or artifact_root()).resolve()
    dataset_path = root / "aqse-development-064acca20fc788c6"
    seal_before = verify_canonical_test_seal(dataset_path)
    expected_protocol = build_evaluation_protocol()
    protocol_path = root / "evaluation-protocols" / expected_protocol.protocol_id
    protocol, protocol_execution = load_evaluation_protocol(
        protocol_path,
        expected_protocol_id=expected_protocol.protocol_id,
    )
    if protocol != expected_protocol:
        raise RuntimeError("stored comparative protocol differs from the code freeze")

    comparison = assemble_comparison_input(
        dataset_path=dataset_path,
        encoder_path=root / "encoders" / protocol.encoder_id,
        train_bank_path=root / "banks" / protocol.train_bank_id,
        validation_bank_path=root / "banks" / protocol.validation_bank_id,
    )
    if verify_canonical_test_seal(dataset_path) != seal_before:
        raise RuntimeError("canonical TEST seal changed during comparison assembly")
    primary_run = load_training_run_artifact(
        root / "training-runs" / PRIMARY_QNG_RUN_ID,
        expected_run_id=PRIMARY_QNG_RUN_ID,
        expected_training_input_fingerprint=TRAINING_INPUT_ID,
    )
    outcome = evaluate_comparison(
        comparison,
        protocol,
        primary_run,
        test_ledger_sha256=seal_before[1],
    )
    if verify_canonical_test_seal(dataset_path) != seal_before:
        raise RuntimeError("canonical TEST seal changed during comparative evaluation")
    evaluation_path, execution = write_comparative_evaluation(
        outcome.artifact,
        method_runtimes=outcome.method_runtimes,
        total_wall_time_ms=outcome.total_wall_time_ms,
        root=root,
    )
    freeze = build_model_selection_freeze(protocol, outcome.artifact)
    freeze_path, freeze_execution = write_model_selection_freeze(freeze, root=root)
    loaded_evaluation, loaded_execution = load_comparative_evaluation(
        evaluation_path,
        expected_evaluation_id=outcome.artifact.evaluation_id,
    )
    loaded_freeze, loaded_freeze_execution = load_model_selection_freeze(
        freeze_path,
        expected_freeze_id=freeze.freeze_id,
    )
    if loaded_evaluation != outcome.artifact or loaded_execution != execution:
        raise RuntimeError("comparative evaluation load-back differs from publication")
    if loaded_freeze != freeze or loaded_freeze_execution != freeze_execution:
        raise RuntimeError("model-selection freeze load-back differs from publication")
    seal_after = verify_canonical_test_seal(dataset_path)
    if seal_after != seal_before:
        raise RuntimeError("canonical TEST seal changed during 1D.4a publication")

    print(
        json.dumps(
            {
                "protocol": {
                    "id": protocol.protocol_id,
                    "digest": protocol.content_digest,
                    "created_at_utc": protocol_execution.created_at_utc,
                },
                "comparison_input": comparison.identity.model_dump(mode="json"),
                "evaluation": {
                    "id": outcome.artifact.evaluation_id,
                    "digest": outcome.artifact.content_digest,
                    "candidate_count": outcome.artifact.candidate_count,
                    "path": str(evaluation_path),
                    "evaluation_json_sha256": file_sha256(
                        evaluation_path / "evaluation.json"
                    ),
                    "total_wall_time_ms": execution.total_wall_time_ms,
                    "method_runtimes": [
                        item.model_dump(mode="json") for item in execution.method_runtimes
                    ],
                },
                "selection_freeze": {
                    "id": freeze.freeze_id,
                    "digest": freeze.content_digest,
                    "path": str(freeze_path),
                    "freeze_json_sha256": file_sha256(freeze_path / "freeze.json"),
                    "created_at_utc": freeze_execution.created_at_utc,
                    "selections": [
                        item.model_dump(mode="json") for item in freeze.selections
                    ],
                },
                "test_state": seal_after[0].test_state,
                "test_ledger_sha256": seal_after[1],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
