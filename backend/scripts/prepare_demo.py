from __future__ import annotations

import json

from app.demo.provisioning import prepare_demo


def main() -> None:
    result = prepare_demo(progress=lambda message: print(f"[AQSE] {message}", flush=True))
    summary = {
        "study_artifact_id": result.manifest.artifact_id,
        "study_content_digest": result.manifest.content_digest,
        "selection_freeze_id": result.freeze.freeze_id,
        "final_evaluation_id": result.final_evaluation.evaluation_id,
        "bundle_ids": [bundle.bundle_id for bundle in result.bundles],
        "active_local_bundle_id": result.freeze.selections[0].selected_bundle_id,
        "active_network_bundle_id": result.freeze.selections[1].selected_bundle_id,
        "reused_study": result.reused_study,
        "test_access": "single completed evaluation; no automatic reopen",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
