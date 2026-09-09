from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.training.assembly import assemble_training_input
from app.training.encoding_storage import load_encoder_artifact
from app.training.evaluation_assembly import assemble_comparison_input
from app.training.evaluation_storage import (
    load_comparative_evaluation,
    load_evaluation_protocol,
    load_model_selection_freeze,
)
from app.training.held_out import (
    ACTOR,
    DATASET_DIGEST,
    DATASET_ID,
    ENCODER_DIGEST,
    ENCODER_ID,
    EVALUATION_ID,
    FREEZE_ID,
    INITIAL_LEDGER_SHA256,
    LABEL_REASON,
    OBSERVATION_REASON,
    PROFILE_FINGERPRINT,
    PROTOCOL_ID,
    TRAIN_BANK_ID,
    assemble_held_out_input,
    build_test_bank,
    evaluate_held_out,
    prepare_frozen_methods,
)
from app.training.held_out_storage import (
    TestLedgerSnapshot,
    load_final_held_out,
    load_g5_authorization,
    load_test_bank,
    read_test_ledger_opaque,
    write_final_held_out,
    write_test_bank,
)
from app.training.models import DatasetPartition, TestAccessAuthorization
from app.training.storage import (
    artifact_root,
    load_labels,
    load_observations,
    verify_archive_opaque,
)


def _require_ledger_state(
    dataset_path: Path,
    *,
    count: int,
    reasons: tuple[str, ...],
) -> TestLedgerSnapshot:
    ledger = read_test_ledger_opaque(dataset_path)
    if len(ledger.entries) != count:
        raise RuntimeError(f"TEST ledger expected {count} entries")
    if tuple(item.reason for item in ledger.entries[1:]) != reasons:
        raise RuntimeError("TEST ledger reasons differ from the authorized access plan")
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the single authorized AQSE 1D.4b held-out evaluation."
    )
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    if os.environ.get("AQSE_ALLOW_TEST_OPEN") != "1":
        raise PermissionError("the single 1D.4b TEST operation requires explicit enablement")
    root = (args.artifact_root or artifact_root()).resolve()
    dataset_path = root / DATASET_ID
    if (root / "test-banks").exists() or (root / "final-held-out-evaluations").exists():
        raise RuntimeError("held-out evidence already exists; TEST evaluation cannot be rerun")
    ledger_before = _require_ledger_state(dataset_path, count=1, reasons=())
    if ledger_before.file_sha256 != INITIAL_LEDGER_SHA256:
        raise RuntimeError("initial TEST ledger differs from the G5 freeze")

    authorization_paths = sorted((root / "g5-authorizations").glob("*"))
    if len(authorization_paths) != 1:
        raise RuntimeError("exactly one immutable G5 authorization is required")
    authorization, _ = load_g5_authorization(
        authorization_paths[0],
        expected_authorization_id=authorization_paths[0].name,
    )
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
    encoder = load_encoder_artifact(
        root / "encoders" / ENCODER_ID,
        expected_artifact_id=ENCODER_ID,
        expected_dataset_id=DATASET_ID,
        expected_dataset_digest=DATASET_DIGEST,
        expected_profile_fingerprint=PROFILE_FINGERPRINT,
    )
    if encoder.artifact.content_digest != ENCODER_DIGEST:
        raise RuntimeError("frozen encoder digest changed")
    comparison = assemble_comparison_input(
        dataset_path=dataset_path,
        encoder_path=root / "encoders" / ENCODER_ID,
        train_bank_path=root / "banks" / TRAIN_BANK_ID,
        validation_bank_path=root
        / "banks"
        / "aqse-validation-bank-105467fa5a4a5cc7",
    )
    training_input = assemble_training_input(
        dataset_path=dataset_path,
        encoder_path=root / "encoders" / ENCODER_ID,
        train_bank_path=root / "banks" / TRAIN_BANK_ID,
    )
    train_observations = load_observations(dataset_path, DatasetPartition.TRAIN)
    validation_observations = load_observations(
        dataset_path,
        DatasetPartition.VALIDATION,
    )
    prepared = prepare_frozen_methods(comparison, evaluation, freeze)
    manifest = verify_archive_opaque(dataset_path)

    test_observations = load_observations(
        dataset_path,
        DatasetPartition.TEST,
        authorization=TestAccessAuthorization(
            actor=ACTOR,
            reason=OBSERVATION_REASON,
        ),
    )
    ledger_after_observations = _require_ledger_state(
        dataset_path,
        count=2,
        reasons=(OBSERVATION_REASON,),
    )
    bank = build_test_bank(test_observations, manifest, authorization)  # type: ignore[arg-type]
    bank_path, bank_execution = write_test_bank(bank, root=root)
    loaded_bank, loaded_bank_execution = load_test_bank(
        bank_path,
        expected_bank_id=bank.artifact_id,
    )
    if loaded_bank != bank or loaded_bank_execution != bank_execution:
        raise RuntimeError("TEST bank load-back differs from publication")

    test_labels = load_labels(
        dataset_path,
        DatasetPartition.TEST,
        authorization=TestAccessAuthorization(actor=ACTOR, reason=LABEL_REASON),
    )
    ledger_final = _require_ledger_state(
        dataset_path,
        count=3,
        reasons=(OBSERVATION_REASON, LABEL_REASON),
    )
    if ledger_final.entries[1].previous_entry_sha256 != ledger_before.entries[0].entry_sha256:
        raise RuntimeError("TEST observation ledger event is not chained to genesis")
    if (
        ledger_final.entries[2].previous_entry_sha256
        != ledger_after_observations.entries[1].entry_sha256
    ):
        raise RuntimeError("TEST label ledger event is not chained to observation access")

    held_out = assemble_held_out_input(
        observations=test_observations,
        labels=test_labels,
        bank=bank,
        encoder=encoder,
        training_input=training_input,
        comparison=comparison,
        reference_observations=(train_observations, validation_observations),
    )
    outcome = evaluate_held_out(
        prepared=prepared,
        held_out=held_out,
        authorization=authorization,
        protocol=protocol,
        evaluation=evaluation,
        freeze=freeze,
        bank=bank,
        ledger_entries=ledger_final.entries,
        final_ledger_sha256=ledger_final.file_sha256,
        backend_root=Path(__file__).resolve().parents[1],
    )
    final_path, execution = write_final_held_out(
        outcome.artifact,
        method_runtimes=outcome.method_runtimes,
        total_wall_time_ms=outcome.total_wall_time_ms,
        root=root,
    )
    loaded, loaded_execution = load_final_held_out(
        final_path,
        expected_artifact_id=outcome.artifact.artifact_id,
    )
    if loaded != outcome.artifact or loaded_execution != execution:
        raise RuntimeError("final held-out load-back differs from publication")
    if read_test_ledger_opaque(dataset_path) != ledger_final:
        raise RuntimeError("artifact publication unexpectedly changed the TEST ledger")

    print(
        json.dumps(
            {
                "authorization": {
                    "id": authorization.authorization_id,
                    "digest": authorization.content_digest,
                },
                "test_bank": {
                    "id": bank.artifact_id,
                    "digest": bank.content_digest,
                    "total": bank.row_count,
                    "eligible": bank.eligible_count,
                    "abstained": bank.abstained_count,
                },
                "ledger": {
                    "before_sha256": ledger_before.file_sha256,
                    "final_sha256": ledger_final.file_sha256,
                    "event_count": len(ledger_final.entries),
                    "reasons": [item.reason for item in ledger_final.entries],
                },
                "final_evaluation": {
                    "id": outcome.artifact.artifact_id,
                    "digest": outcome.artifact.content_digest,
                    "path": str(final_path),
                    "results": [
                        item.model_dump(mode="json") for item in outcome.artifact.results
                    ],
                    "method_runtimes": [
                        item.model_dump(mode="json") for item in execution.method_runtimes
                    ],
                    "total_wall_time_ms": execution.total_wall_time_ms,
                },
                "semantic_test_load_count": outcome.artifact.semantic_test_load_count,
                "generation_plan_access_count": (
                    outcome.artifact.generation_plan_access_count
                ),
                "fixed_gradient_qng_identical": (
                    outcome.artifact.fixed_gradient_qng_identical
                ),
                "no_winner_predeclared": outcome.artifact.no_winner_predeclared,
                "test_results_not_used_for_model_selection": (
                    outcome.artifact.test_results_not_used_for_model_selection
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
