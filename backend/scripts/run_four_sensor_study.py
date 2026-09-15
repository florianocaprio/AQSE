from __future__ import annotations

import json

from app.four_sensor_study.workflow import run_four_sensor_study


def main() -> None:
    result = run_four_sensor_study(
        progress=lambda message: print(f"[AQSE 4S] {message}", flush=True)
    )
    quantum = result.evaluation["metrics"]["frozen_quantum_afse_mlp"]
    raw = result.evaluation["metrics"]["raw_state8_mlp"]
    observable = result.evaluation["metrics"]["observable_p99_rule"]
    summary = {
        "pilot_artifact_id": result.pilot_manifest["artifact_id"],
        "pilot_assessment_digest": result.pilot_assessment["assessment_digest"],
        "pilot_accepted": result.pilot_assessment["accepted"],
        "protocol_freeze_id": result.protocol_freeze_manifest["artifact_id"],
        "final_test_dataset_id": result.dataset_manifest["artifact_id"],
        "model_binding_id": result.model_binding_manifest["artifact_id"],
        "authorization_id": result.authorization_manifest["artifact_id"],
        "evaluation_id": result.evaluation["evaluation_id"],
        "evaluation_artifact_id": result.evaluation_manifest["artifact_id"],
        "test_ledger_events": [item["event"] for item in result.ledger_entries],
        "frozen_quantum_afse_mlp": {
            "balanced_accuracy": quantum["balanced_accuracy"],
            "macro_f1": quantum["macro_f1"],
            "coverage": quantum["coverage"],
        },
        "raw_state8_mlp": {
            "balanced_accuracy": raw["balanced_accuracy"],
            "macro_f1": raw["macro_f1"],
            "coverage": raw["coverage"],
        },
        "observable_p99_rule": {
            "balanced_accuracy": observable["balanced_accuracy"],
            "macro_f1": observable["macro_f1"],
            "coverage": observable["coverage"],
        },
        "success_criteria": result.evaluation["success_criteria"],
        "all_primary_criteria_met": result.evaluation["all_primary_criteria_met"],
        "quantum_advantage_demonstrated": result.evaluation[
            "quantum_advantage_demonstrated"
        ],
        "timings_s": result.timings_s,
        "artifact_paths": {
            "pilot": str(result.pilot_path),
            "protocol_freeze": str(result.protocol_freeze_path),
            "final_test_dataset": str(result.dataset_path),
            "model_binding": str(result.model_binding_path),
            "authorization": str(result.authorization_path),
            "evaluation": str(result.evaluation_path),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
