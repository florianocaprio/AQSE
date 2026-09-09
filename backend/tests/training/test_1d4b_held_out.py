from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.training.assembly import assemble_training_input
from app.training.canonical import canonical_json_bytes
from app.training.evaluation_assembly import assemble_comparison_input
from app.training.evaluation_protocol import build_evaluation_protocol
from app.training.evaluation_storage import (
    load_comparative_evaluation,
    load_model_selection_freeze,
)
from app.training.held_out import (
    ACTOR,
    EVALUATION_ID,
    FREEZE_ID,
    INITIAL_LEDGER_SHA256,
    LABEL_REASON,
    OBSERVATION_REASON,
    PROTECTED_HASHES,
    assemble_held_out_input,
    build_g5_authorization,
    build_test_bank,
    evaluate_held_out,
    prepare_frozen_methods,
)
from app.training.held_out_storage import (
    load_final_held_out,
    load_g5_authorization,
    load_test_bank,
    read_test_ledger_opaque,
    write_final_held_out,
    write_g5_authorization,
    write_test_bank,
)
from app.training.models import (
    DatasetPartition,
    EpisodeLabel,
)
from app.training.models import TestAccessAuthorization as AccessAuthorization
from app.training.models import TestAccessLedgerEntry as LedgerEntry
from app.training.storage import (
    artifact_root,
    load_labels,
    load_observations,
    verify_archive_opaque,
    write_dataset,
)


def _frozen_graph():  # type: ignore[no-untyped-def]
    root = artifact_root()
    protocol = build_evaluation_protocol()
    evaluation, _ = load_comparative_evaluation(
        root / "comparative-evaluations" / EVALUATION_ID,
        expected_evaluation_id=EVALUATION_ID,
    )
    freeze, _ = load_model_selection_freeze(
        root / "model-selection-freezes" / FREEZE_ID,
        expected_freeze_id=FREEZE_ID,
    )
    comparison = assemble_comparison_input(
        dataset_path=root / protocol.dataset_id,
        encoder_path=root / "encoders" / protocol.encoder_id,
        train_bank_path=root / "banks" / protocol.train_bank_id,
        validation_bank_path=root / "banks" / protocol.validation_bank_id,
    )
    training_input = assemble_training_input(
        dataset_path=root / protocol.dataset_id,
        encoder_path=root / "encoders" / protocol.encoder_id,
        train_bank_path=root / "banks" / protocol.train_bank_id,
    )
    authorization = build_g5_authorization(
        protocol,
        evaluation,
        freeze,
        initial_ledger_sha256=INITIAL_LEDGER_SHA256,
    )
    return root, protocol, evaluation, freeze, comparison, training_input, authorization


def _observations(*, offset: float, partition: DatasetPartition, invalid: int = 0):
    count = 24 if partition is DatasetPartition.TEST else 3
    rng = np.random.default_rng(int(offset * 1000) + count)
    features = rng.normal(loc=offset, size=(count, 19, 8)).astype(np.float64)
    valid_mask = np.ones((count, 19), dtype=np.bool_)
    valid_mask[:invalid, 9] = False
    episode_ids = tuple(f"episode-{partition.value}-{index:02d}" for index in range(count))
    lineage_ids = tuple(f"lineage-{partition.value}-{index:02d}" for index in range(count))
    windows = tuple(
        tuple(
            SimpleNamespace(window_id=f"{episode_id}:x:{ordinal}")
            for ordinal in range(19)
        )
        for episode_id in episode_ids
    )
    return SimpleNamespace(
        dataset_id="aqse-development-064acca20fc788c6",
        partition=partition,
        episode_ids=episode_ids,
        lineage_ids=lineage_ids,
        time_s=np.tile(np.arange(10, dtype=np.float64), (count, 1)) + offset,
        measured_field_T=rng.normal(loc=offset, size=(count, 10, 3)),
        temperature_K=rng.normal(loc=293.0 + offset, size=(count, 10)),
        saturation_mask=np.zeros((count, 10, 3), dtype=np.bool_),
        features=features,
        valid_mask=valid_mask,
        windows=windows,
    )


def _labels(observations) -> tuple[EpisodeLabel, ...]:  # type: ignore[no-untyped-def]
    return tuple(
        EpisodeLabel(
            episode_id=episode_id,
            lineage_id=lineage_id,
            target=-1 if index < 12 else 1,
        )
        for index, (episode_id, lineage_id) in enumerate(
            zip(observations.episode_ids, observations.lineage_ids, strict=True)
        )
    )


class _IdentityEncoder:
    artifact = object()

    def transform(self, values, *, context):  # type: ignore[no-untyped-def]
        del context
        return np.asarray(values, dtype=np.float64)


def _ledger_entry(
    sequence: int,
    *,
    reason: str,
    previous: str | None,
) -> LedgerEntry:
    payload = {
        "schema_version": "aqse.test-access-ledger.v2",
        "sequence": sequence,
        "event": "sealed" if sequence == 0 else "opened",
        "dataset_id": "aqse-development-064acca20fc788c6",
        "scientific_digest": (
            "064acca20fc788c6b5524cd95f56404dfcf4dd3cea942d970b75b21d76ba61d7"
        ),
        "occurred_at_utc": f"2026-09-09T00:00:0{sequence}Z",
        "actor": "AQSE 1D.1 archiver" if sequence == 0 else ACTOR,
        "reason": reason,
        "fixture_only": False,
        "previous_entry_sha256": previous,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return LedgerEntry.model_validate({**payload, "entry_sha256": digest})


def test_g5_authorization_is_deterministic_contains_no_test_values_and_is_immutable(
    tmp_path: Path,
) -> None:
    *_, authorization = _frozen_graph()
    assert not authorization.contains_test_values
    assert authorization.no_refit
    assert authorization.no_training_rerun
    assert tuple(item.sequence for item in authorization.expected_semantic_loads) == (1, 2)

    path, execution = write_g5_authorization(authorization, root=tmp_path)
    loaded, loaded_execution = load_g5_authorization(
        path,
        expected_authorization_id=authorization.authorization_id,
    )
    assert loaded == authorization
    assert loaded_execution == execution
    assert write_g5_authorization(authorization, root=tmp_path)[0] == path


def test_test_bank_is_label_independent_fixed_ordinal_and_has_no_fallback(
    tmp_path: Path,
) -> None:
    root, *_, authorization = _frozen_graph()
    observations = _observations(
        offset=4.0,
        partition=DatasetPartition.TEST,
        invalid=2,
    )
    manifest = verify_archive_opaque(root / authorization.dataset_id)
    bank = build_test_bank(observations, manifest, authorization)  # type: ignore[arg-type]

    assert bank.row_count == 24
    assert bank.eligible_count == 22
    assert bank.abstained_count == 2
    assert all(item.window_ordinal == 9 for item in bank.rows)
    assert [item.episode_id for item in bank.rows] == sorted(observations.episode_ids)
    assert all(item.abstention_reason for item in bank.rows[:2])
    assert bank.selection_independent_of_labels
    assert bank.no_fallback
    path, execution = write_test_bank(bank, root=tmp_path)
    loaded, loaded_execution = load_test_bank(path, expected_bank_id=bank.artifact_id)
    assert loaded == bank
    assert loaded_execution == execution


def test_fixture_test_access_has_exact_two_loads_and_valid_three_event_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_development_build,
) -> None:  # type: ignore[no-untyped-def]
    path, manifest, _ = write_dataset(small_development_build, root=tmp_path)
    monkeypatch.setenv("AQSE_TEST_FIXTURE_DATASET_ID", manifest.dataset_id)
    load_observations(
        path,
        DatasetPartition.TEST,
        authorization=AccessAuthorization(
            actor=ACTOR,
            reason=OBSERVATION_REASON,
            fixture_only=True,
        ),
    )
    load_labels(
        path,
        DatasetPartition.TEST,
        authorization=AccessAuthorization(
            actor=ACTOR,
            reason=LABEL_REASON,
            fixture_only=True,
        ),
    )
    ledger = read_test_ledger_opaque(path)

    assert len(ledger.entries) == 3
    assert tuple(item.sequence for item in ledger.entries) == (0, 1, 2)
    assert tuple(item.reason for item in ledger.entries[1:]) == (
        OBSERVATION_REASON,
        LABEL_REASON,
    )
    assert ledger.entries[1].previous_entry_sha256 == ledger.entries[0].entry_sha256
    assert ledger.entries[2].previous_entry_sha256 == ledger.entries[1].entry_sha256


def test_canonical_runner_declares_only_two_test_loads_and_no_generation_access() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "scripts" / "run_1d4b_held_out.py"
    ).read_text(encoding="utf-8")
    assert source.count("test_observations = load_observations(") == 1
    assert source.count("test_labels = load_labels(") == 1
    assert "load_generation_channel" not in source


def test_reconstruction_uses_frozen_rbf_snapshot_and_does_not_train_theta() -> None:
    _, _, evaluation, freeze, comparison, _, _ = _frozen_graph()
    prepared = prepare_frozen_methods(comparison, evaluation, freeze)

    np.testing.assert_allclose(
        prepared.rbf_mean,
        np.asarray(evaluation.classical_preprocessing.fitted_mean),
        atol=0.0,
    )
    np.testing.assert_allclose(
        prepared.rbf_scale,
        np.asarray(evaluation.classical_preprocessing.fitted_scale),
        atol=0.0,
    )
    assert tuple(item.selection for item in prepared.methods) == freeze.selections
    assert prepared.methods[2].selection.selected_theta == freeze.selections[2].selected_theta
    assert prepared.methods[3].selection.selected_theta == freeze.selections[3].selected_theta
    assert prepared.methods[4].selection.selected_theta == freeze.selections[4].selected_theta


def test_abstention_produces_one_common_cohort_and_fixed_kernel_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, protocol, evaluation, freeze, comparison, training_input, authorization = (
        _frozen_graph()
    )
    observations = _observations(
        offset=5.0,
        partition=DatasetPartition.TEST,
        invalid=3,
    )
    manifest = verify_archive_opaque(root / authorization.dataset_id)
    bank = build_test_bank(observations, manifest, authorization)  # type: ignore[arg-type]
    monkeypatch.setattr("app.training.held_out.input_contract_for", lambda _: None)
    held_out = assemble_held_out_input(
        observations=observations,
        labels=_labels(observations),
        bank=bank,
        encoder=_IdentityEncoder(),  # type: ignore[arg-type]
        training_input=training_input,
        comparison=comparison,
        reference_observations=(
            _observations(offset=7.0, partition=DatasetPartition.TRAIN),
            _observations(offset=9.0, partition=DatasetPartition.VALIDATION),
        ),
    )

    assert held_out.raw.shape == (21, 8)
    assert held_out.encoded.shape == (21, 8)
    assert held_out.labels.shape == (21,)
    assert len(held_out.identity.abstained_lineage_ids) == 3
    assert held_out.identity.quantum_test_train_kernel_shape == (21, 32)


def test_final_evaluation_is_deterministic_has_no_winner_and_quantum_results_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, protocol, evaluation, freeze, comparison, training_input, authorization = (
        _frozen_graph()
    )
    observations = _observations(offset=6.0, partition=DatasetPartition.TEST)
    manifest = verify_archive_opaque(root / authorization.dataset_id)
    bank = build_test_bank(observations, manifest, authorization)  # type: ignore[arg-type]
    monkeypatch.setattr("app.training.held_out.input_contract_for", lambda _: None)
    held_out = assemble_held_out_input(
        observations=observations,
        labels=_labels(observations),
        bank=bank,
        encoder=_IdentityEncoder(),  # type: ignore[arg-type]
        training_input=training_input,
        comparison=comparison,
        reference_observations=(
            _observations(offset=8.0, partition=DatasetPartition.TRAIN),
            _observations(offset=10.0, partition=DatasetPartition.VALIDATION),
        ),
    )
    prepared = prepare_frozen_methods(comparison, evaluation, freeze)
    genesis = _ledger_entry(
        0,
        reason="test partition sealed at dataset creation",
        previous=None,
    )
    observation = _ledger_entry(
        1,
        reason=OBSERVATION_REASON,
        previous=genesis.entry_sha256,
    )
    label = _ledger_entry(
        2,
        reason=LABEL_REASON,
        previous=observation.entry_sha256,
    )
    monkeypatch.setattr(
        "app.training.held_out._protected_hashes",
        lambda _: PROTECTED_HASHES,
    )
    first = evaluate_held_out(
        prepared=prepared,
        held_out=held_out,
        authorization=authorization,
        protocol=protocol,
        evaluation=evaluation,
        freeze=freeze,
        bank=bank,
        ledger_entries=(genesis, observation, label),
        final_ledger_sha256="f" * 64,
        backend_root=Path(__file__).resolve().parents[2],
    )
    second = evaluate_held_out(
        prepared=prepared,
        held_out=held_out,
        authorization=authorization,
        protocol=protocol,
        evaluation=evaluation,
        freeze=freeze,
        bank=bank,
        ledger_entries=(genesis, observation, label),
        final_ledger_sha256="f" * 64,
        backend_root=Path(__file__).resolve().parents[2],
    )

    assert first.artifact == second.artifact
    assert first.artifact.no_winner_predeclared
    assert first.artifact.test_results_not_used_for_model_selection
    assert first.artifact.generation_plan_access_count == 0
    assert all(item.eligible_count == 24 for item in first.artifact.results)
    assert all(item.confidence_intervals[0].resamples == 2000 for item in first.artifact.results)
    quantum = first.artifact.results[2:]
    assert len({item.prediction_digest for item in quantum}) == 1
    assert len({item.score_digest for item in quantum}) == 1
    assert all(item.compute_cost.test_train_kernel_coordinates == 24 * 32 for item in quantum)
    path, execution = write_final_held_out(
        first.artifact,
        method_runtimes=first.method_runtimes,
        total_wall_time_ms=first.total_wall_time_ms,
        root=tmp_path,
    )
    loaded, loaded_execution = load_final_held_out(
        path,
        expected_artifact_id=first.artifact.artifact_id,
    )
    assert loaded == first.artifact
    assert loaded_execution == execution
    with pytest.raises(FileExistsError, match="do not rerun TEST"):
        write_final_held_out(
            first.artifact,
            method_runtimes=first.method_runtimes,
            total_wall_time_ms=first.total_wall_time_ms,
            root=tmp_path,
        )


def test_published_held_out_evidence_is_verified_without_reopening_test() -> None:
    root = artifact_root()
    final_paths = sorted((root / "final-held-out-evaluations").glob("*"))
    if not final_paths:
        pytest.skip("canonical 1D.4b evidence has not been published")
    artifact, execution = load_final_held_out(
        final_paths[0],
        expected_artifact_id=final_paths[0].name,
    )
    bank, _ = load_test_bank(
        root / "test-banks" / artifact.test_bank_id,
        expected_bank_id=artifact.test_bank_id,
    )
    authorization, _ = load_g5_authorization(
        root / "g5-authorizations" / artifact.authorization_id,
        expected_authorization_id=artifact.authorization_id,
    )
    ledger = read_test_ledger_opaque(root / artifact.dataset_id)

    assert artifact.content_digest == (
        "7f338dafc2c02f0959b33caf8c3033cf7a971ba455d4c8593bbde91774404228"
    )
    assert authorization.content_digest == artifact.authorization_digest
    assert bank.content_digest == artifact.test_bank_digest
    assert ledger.file_sha256 == artifact.final_ledger_sha256
    assert len(ledger.entries) == artifact.final_ledger_event_count == 3
    assert execution.artifact_id == artifact.artifact_id
    assert artifact.semantic_test_load_count == 2
    assert artifact.generation_plan_access_count == 0
