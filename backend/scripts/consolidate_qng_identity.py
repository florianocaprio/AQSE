from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.training.canonical import file_sha256  # noqa: E402
from app.training.execution_intent import TrainingExecutionIntent  # noqa: E402
from app.training.intent_storage import FileExecutionIntentStore  # noqa: E402
from app.training.run_models import TrainingJobState, TrainingJobView  # noqa: E402
from app.training.run_storage import load_training_run_artifact  # noqa: E402
from app.training.storage import artifact_root  # noqa: E402
from app.training.trajectory import (  # noqa: E402
    build_trajectory_audit_mapping,
    derive_trajectory_identity,
)
from app.training.trajectory_storage import write_trajectory_audit_mapping  # noqa: E402

DESIGNATED_EXECUTION_ID = "aqse-qng-run-f00c702ad790df2b"
EQUIVALENT_EXECUTION_ID = "aqse-qng-run-5ae026e633def66b"
TRAINING_INPUT_FINGERPRINT = "aqse-training-input-e853259fac7c0eba"
CANONICAL_EXECUTION_INTENT_ID = "aqse-1d3-canonical-training-v1"
CANONICAL_JOB_ID = "qng-job-5fb8b8f43e0c46678b059b0de77efc5b"


def _first_difference(left: Any, right: Any, path: str = "content") -> str | None:
    if type(left) is not type(right):
        return f"{path}: types {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: keys differ"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(
                left_item,
                right_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    return None if left == right else f"{path}: {left!r} != {right!r}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the additive AQSE 1D.3 trajectory/intent audit records."
    )
    parser.add_argument("--artifact-root", type=Path, default=artifact_root())
    arguments = parser.parse_args()
    root = arguments.artifact_root.resolve()
    runs = {
        execution_id: load_training_run_artifact(
            root / "training-runs" / execution_id,
            expected_run_id=execution_id,
            expected_training_input_fingerprint=TRAINING_INPUT_FINGERPRINT,
        )
        for execution_id in (DESIGNATED_EXECUTION_ID, EQUIVALENT_EXECUTION_ID)
    }
    identities = {
        execution_id: derive_trajectory_identity(run)
        for execution_id, run in runs.items()
    }
    designated = identities[DESIGNATED_EXECUTION_ID]
    equivalent = identities[EQUIVALENT_EXECUTION_ID]
    if designated != equivalent:
        difference = _first_difference(
            designated.content.model_dump(mode="json"),
            equivalent.content.model_dump(mode="json"),
        )
        raise RuntimeError(f"historical v1 scientific trajectories differ at {difference}")

    run_hashes = {
        execution_id: file_sha256(
            root / "training-runs" / execution_id / "run.json"
        )
        for execution_id in runs
    }
    mapping = build_trajectory_audit_mapping(
        designated,
        designated_execution_id=DESIGNATED_EXECUTION_ID,
        equivalent_execution_ids=(EQUIVALENT_EXECUTION_ID,),
        execution_run_sha256=run_hashes,
    )
    mapping_path = write_trajectory_audit_mapping(mapping, root=root)

    intent = TrainingExecutionIntent(intent_id=CANONICAL_EXECUTION_INTENT_ID)
    intent_store = FileExecutionIntentStore(root)
    execution_metadata = json.loads(
        (root / "training-runs" / DESIGNATED_EXECUTION_ID / "execution.json").read_bytes()
    )
    recorded_at = execution_metadata["created_at_utc"]
    claimed, _ = intent_store.claim(
        intent,
        job_id=CANONICAL_JOB_ID,
        claimed_at_utc=recorded_at,
    )
    terminal = TrainingJobView(
        job_id=claimed.claim.job_id,
        state=TrainingJobState.COMPLETED,
        created_at_utc=recorded_at,
        updated_at_utc=recorded_at,
        accepted_update_count=runs[DESIGNATED_EXECUTION_ID].accepted_update_count,
        stop_reason=runs[DESIGNATED_EXECUTION_ID].stop_reason,
        run_artifact_id=DESIGNATED_EXECUTION_ID,
    )
    intent_record = intent_store.complete(intent, terminal)
    print(
        json.dumps(
            {
                "trajectory_id": designated.trajectory_id,
                "trajectory_digest": designated.content_digest,
                "mapping_digest": mapping.mapping_digest,
                "mapping_path": str(mapping_path),
                "designated_execution_id": DESIGNATED_EXECUTION_ID,
                "equivalent_execution_ids": [EQUIVALENT_EXECUTION_ID],
                "execution_run_sha256": run_hashes,
                "canonical_execution_intent": intent.model_dump(mode="json"),
                "intent_claim_digest": intent_record.claim.claim_digest,
                "intent_result_digest": intent_record.result.result_digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
