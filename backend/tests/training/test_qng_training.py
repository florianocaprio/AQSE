from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from app.training import assembly, banks, encoding
from app.training.assembly import assemble_training_input
from app.training.banks import select_training_quantum_bank
from app.training.encoding import fit_phase_direct_encoder
from app.training.encoding_storage import (
    load_bank_artifact,
    load_encoder_artifact,
    write_bank_artifact,
    write_encoder_artifact,
)
from app.training.generation import (
    DatasetBuild,
    build_episode_plans,
    generate_episode,
)
from app.training.models import DatasetPartition, SplitAssignment
from app.training.run_storage import (
    load_training_run_artifact,
    write_training_run_artifact,
)
from app.training.runner import (
    canonical_theta0,
    cross_engine_one_step_gate,
    run_bounded_qng,
    validate_training_run_artifact,
    wrapper_equivalence,
)
from app.training.storage import load_labels, load_observations, write_dataset
from app.training.workflow import validate_training_input_arrays


@dataclass(frozen=True)
class QngFixture:
    dataset_path: Path
    encoder_path: Path
    bank_path: Path
    training_input: assembly.TrainingInput


@pytest.fixture(scope="module")
def qng_fixture(tmp_path_factory) -> QngFixture:
    patcher = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("qng-training")
    plans = build_episode_plans(
        master_seed=71_001,
        episodes_per_class=20,
        purpose="qng-test-fixture",
    )
    episodes = tuple(generate_episode(plan) for plan in plans)
    by_target = {
        target: [episode for episode in episodes if episode.label.target == target]
        for target in (-1, 1)
    }
    partition_by_id: dict[str, DatasetPartition] = {}
    for target in (-1, 1):
        selected = by_target[target]
        for episode in selected[:16]:
            partition_by_id[episode.plan.episode_id] = DatasetPartition.TRAIN
        for episode in selected[16:18]:
            partition_by_id[episode.plan.episode_id] = DatasetPartition.VALIDATION
        for episode in selected[18:20]:
            partition_by_id[episode.plan.episode_id] = DatasetPartition.TEST
    build = DatasetBuild(
        kind="development",
        master_seed=71_001,
        split_seed=71_002,
        episodes=episodes,
        assignments=tuple(
            SplitAssignment(
                episode_id=episode.plan.episode_id,
                lineage_id=episode.plan.lineage_id,
                partition=partition_by_id[episode.plan.episode_id],
            )
            for episode in episodes
        ),
    )
    dataset_path, manifest, _ = write_dataset(build, root=root)
    for module in (encoding, banks):
        patcher.setattr(module, "CANONICAL_DEVELOPMENT_DATASET_ID", manifest.dataset_id)
        patcher.setattr(
            module,
            "CANONICAL_DEVELOPMENT_DIGEST",
            manifest.scientific_digest,
        )
    patcher.setattr(
        encoding,
        "CANONICAL_FEATURE_PROFILE_FINGERPRINT",
        manifest.feature_profile_fingerprint,
    )
    train = load_observations(dataset_path, DatasetPartition.TRAIN)
    labels = load_labels(dataset_path, DatasetPartition.TRAIN)
    encoder_value = fit_phase_direct_encoder(train, manifest)
    encoder_path, _ = write_encoder_artifact(encoder_value, root=root)
    bank = select_training_quantum_bank(
        train,
        labels,
        manifest,
        encoder_artifact_id=encoder_value.artifact.artifact_id,
    )
    bank_path = write_bank_artifact(bank, root=root)
    replacements = {
        "CANONICAL_DATASET_ID": manifest.dataset_id,
        "CANONICAL_DATASET_DIGEST": manifest.scientific_digest,
        "CANONICAL_PROFILE_FINGERPRINT": manifest.feature_profile_fingerprint,
        "CANONICAL_ENCODER_ID": encoder_value.artifact.artifact_id,
        "CANONICAL_ENCODER_DIGEST": encoder_value.artifact.content_digest,
        "CANONICAL_SCALER_ID": encoder_value.artifact.scaler_id,
        "CANONICAL_TRAIN_BANK_ID": bank.artifact_id,
        "CANONICAL_TRAIN_BANK_DIGEST": bank.content_digest,
    }
    for name, value in replacements.items():
        patcher.setattr(assembly, name, value)
    training_input = assemble_training_input(
        dataset_path=dataset_path,
        encoder_path=encoder_path,
        train_bank_path=bank_path,
    )
    try:
        yield QngFixture(
            dataset_path=dataset_path,
            encoder_path=encoder_path,
            bank_path=bank_path,
            training_input=training_input,
        )
    finally:
        patcher.undo()


def test_assembler_joins_bank_and_labels_by_identity(
    qng_fixture: QngFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = qng_fixture.training_input
    original = assembly.load_labels
    monkeypatch.setattr(
        assembly,
        "load_labels",
        lambda *args, **kwargs: tuple(reversed(original(*args, **kwargs))),
    )
    reordered = assemble_training_input(
        dataset_path=qng_fixture.dataset_path,
        encoder_path=qng_fixture.encoder_path,
        train_bank_path=qng_fixture.bank_path,
    )

    assert expected.X_train.shape == (32, 8)
    assert expected.y_train.shape == (32,)
    assert np.count_nonzero(expected.y_train == -1) == 16
    assert np.count_nonzero(expected.y_train == 1) == 16
    assert len(set(expected.ordered_lineage_ids)) == 32
    assert expected.ordered_window_ids == tuple(
        row.window_id for row in expected.identity.ordered_rows
    )
    np.testing.assert_array_equal(reordered.X_train, expected.X_train)
    np.testing.assert_array_equal(reordered.y_train, expected.y_train)
    assert reordered.identity == expected.identity
    assert not expected.X_train.flags.writeable
    assert not expected.y_train.flags.writeable
    assert set(vars(expected)).isdisjoint(
        {"plans", "generation_seed", "signal_seed", "white_noise_std_nt"}
    )


def test_assembler_loads_train_labels_only(
    qng_fixture: QngFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_partitions: list[DatasetPartition] = []
    original = assembly.load_labels

    def recording_load_labels(path: Path, partition: DatasetPartition):
        requested_partitions.append(partition)
        return original(path, partition)

    monkeypatch.setattr(assembly, "load_labels", recording_load_labels)
    assemble_training_input(
        dataset_path=qng_fixture.dataset_path,
        encoder_path=qng_fixture.encoder_path,
        train_bank_path=qng_fixture.bank_path,
    )

    assert requested_partitions == [DatasetPartition.TRAIN]


def test_assembler_row_order_matches_frozen_bank(qng_fixture: QngFixture) -> None:
    training_input = qng_fixture.training_input
    bank = load_bank_artifact(
        qng_fixture.bank_path,
        expected_artifact_id=assembly.CANONICAL_TRAIN_BANK_ID,
        expected_dataset_id=assembly.CANONICAL_DATASET_ID,
        expected_dataset_digest=assembly.CANONICAL_DATASET_DIGEST,
        expected_encoder_artifact_id=assembly.CANONICAL_ENCODER_ID,
    )
    encoder_value = load_encoder_artifact(
        qng_fixture.encoder_path,
        expected_artifact_id=assembly.CANONICAL_ENCODER_ID,
        expected_dataset_id=assembly.CANONICAL_DATASET_ID,
        expected_dataset_digest=assembly.CANONICAL_DATASET_DIGEST,
        expected_profile_fingerprint=assembly.CANONICAL_PROFILE_FINGERPRINT,
    )
    observations = load_observations(qng_fixture.dataset_path, DatasetPartition.TRAIN)
    index = {episode_id: i for i, episode_id in enumerate(observations.episode_ids)}
    raw = np.stack(
        [
            observations.features[index[row.episode_id], row.window_ordinal]
            for row in bank.rows
        ]
    )
    expected = encoder_value.transform(
        raw,
        context=encoding.input_contract_for(encoder_value.artifact),
    )
    np.testing.assert_array_equal(training_input.X_train, expected)


def test_theta_initialization_and_wrapper_equivalence_are_deterministic(
    qng_fixture: QngFixture,
) -> None:
    theta_first = canonical_theta0()
    np.random.default_rng(999).normal(size=100)
    theta_second = canonical_theta0()
    np.testing.assert_array_equal(theta_first, theta_second)
    assert theta_first.tobytes() == theta_second.tobytes()
    assert theta_first.shape == (16,)
    assert np.all(theta_first >= -0.8) and np.all(theta_first <= 0.8)
    indices = np.asarray([0, 1, 16, 17, 2, 18])
    result = wrapper_equivalence(
        qng_fixture.training_input.X_train[indices],
        qng_fixture.training_input.y_train[indices],
        theta_first,
        steps=3,
    )
    assert result["accepted_steps"] == 3
    assert result["final_theta_max_abs_delta"] <= 1.0e-12
    assert result["history_max_abs_delta"] <= 1.0e-12


def test_cross_engine_gate_agrees_on_representative_fixture_rows(
    qng_fixture: QngFixture,
) -> None:
    indices = np.asarray([0, 1, 16, 17, 2, 18])
    result = cross_engine_one_step_gate(
        qng_fixture.training_input.X_train[indices],
        qng_fixture.training_input.y_train[indices],
        canonical_theta0(),
    )

    assert result["kernel_max_abs_delta"] <= 1.0e-12
    assert result["initial_loss_abs_delta"] <= 1.0e-10
    assert result["gradient_max_abs_delta"] <= 1.0e-10
    assert result["metric_max_abs_delta"] <= 1.0e-10
    assert result["candidate_theta_max_abs_delta"] <= 1.0e-10


def test_runner_records_real_monotonic_protected_history(qng_fixture: QngFixture) -> None:
    artifact = run_bounded_qng(qng_fixture.training_input)

    assert artifact.accepted_update_count == len(artifact.steps)
    assert artifact.accepted_update_count <= 10
    assert artifact.stop_reason in {"MAX_UPDATES_REACHED", "NO_ACCEPTED_UPDATE"}
    assert [step.step_index for step in artifact.steps] == list(
        range(artifact.accepted_update_count)
    )
    assert all(step.loss_after <= step.loss_before for step in artifact.steps)
    assert artifact.final_theta == (
        artifact.steps[-1].theta_after if artifact.steps else artifact.theta0
    )
    assert artifact.counters.differential_calls == artifact.accepted_update_count
    validate_training_run_artifact(artifact)


def test_runner_does_not_fabricate_history_and_honors_cancellation(
    qng_fixture: QngFixture,
) -> None:
    def no_update(engine, X, y, theta, **kwargs):
        return np.asarray(theta).copy(), []

    stopped = run_bounded_qng(qng_fixture.training_input, optimizer=no_update)
    assert stopped.accepted_update_count == 0
    assert stopped.steps == ()
    assert stopped.stop_reason == "NO_ACCEPTED_UPDATE"
    cancellation = Event()
    cancellation.set()
    cancelled = run_bounded_qng(qng_fixture.training_input, cancellation=cancellation)
    assert cancelled.accepted_update_count == 0
    assert cancelled.stop_reason == "CANCELLED"


def test_workflow_rejects_arrays_that_differ_from_frozen_identity(
    qng_fixture: QngFixture,
) -> None:
    changed_labels = qng_fixture.training_input.y_train.copy()
    changed_labels[0] *= -1
    changed_labels.setflags(write=False)

    with pytest.raises(ValueError, match="labels differ"):
        validate_training_input_arrays(
            replace(qng_fixture.training_input, y_train=changed_labels)
        )


def test_training_run_storage_is_immutable_and_rejects_corruption(
    tmp_path: Path,
    qng_fixture: QngFixture,
) -> None:
    artifact = run_bounded_qng(qng_fixture.training_input)
    path, _ = write_training_run_artifact(artifact, root=tmp_path)
    loaded = load_training_run_artifact(
        path,
        expected_run_id=artifact.run_id,
        expected_training_input_fingerprint=artifact.training_input.fingerprint_id,
    )
    payload = (path / "run.json").read_text(encoding="utf-8")

    assert loaded == artifact
    assert all(word not in payload.lower() for word in ("selected", "promoted", "deployed"))
    assert "test_bank" not in payload.lower()
    with pytest.raises(FileExistsError, match="already exists"):
        write_training_run_artifact(artifact, root=tmp_path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_training_run_artifact(
            path,
            expected_run_id=artifact.run_id,
            expected_training_input_fingerprint="aqse-training-input-0000000000000000",
        )
    unexpected = path / "unexpected"
    unexpected.mkdir()
    with pytest.raises(ValueError, match="unexpected entries"):
        load_training_run_artifact(
            path,
            expected_run_id=artifact.run_id,
            expected_training_input_fingerprint=artifact.training_input.fingerprint_id,
        )
    unexpected.rmdir()
    target = path / "run.json"
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="size or path"):
        load_training_run_artifact(
            path,
            expected_run_id=artifact.run_id,
            expected_training_input_fingerprint=artifact.training_input.fingerprint_id,
        )


def test_training_run_rejects_incompatible_embedded_identity(
    qng_fixture: QngFixture,
) -> None:
    artifact = run_bounded_qng(qng_fixture.training_input)
    corrupted_input = artifact.training_input.model_copy(
        update={"source_dataset_id": "aqse-development-wrong"}
    )
    with pytest.raises(ValueError, match="content digest is invalid"):
        validate_training_run_artifact(
            artifact.model_copy(update={"training_input": corrupted_input})
        )


def test_training_run_rejects_changed_protected_source_identity(
    qng_fixture: QngFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = run_bounded_qng(qng_fixture.training_input)
    monkeypatch.setattr(
        "app.training.runner.TQK8_SOURCE_SHA256",
        "0" * 64,
    )

    with pytest.raises(ValueError, match="protected TQK8 source identity"):
        validate_training_run_artifact(artifact)
