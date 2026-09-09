from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.training import banks
from app.training.banks import (
    CENTRAL_WINDOW_ORDINAL,
    TRAIN_SELECTION_SEED,
    select_training_quantum_bank,
    select_validation_quantum_bank,
)
from app.training.encoding_storage import load_bank_artifact, write_bank_artifact
from app.training.models import DatasetPartition, EpisodeLabel
from app.training.storage import ObservablePartition, load_observations, write_dataset

ENCODER_ID = "aqse-encoder-0123456789abcdef"


@pytest.fixture(scope="module")
def bank_source(tmp_path_factory, small_development_build):
    root = tmp_path_factory.mktemp("bank-dataset")
    path, manifest, _ = write_dataset(small_development_build, root=root)
    train = load_observations(path, DatasetPartition.TRAIN)
    return manifest, train


def _approve_fixture_identity(manifest, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(banks, "CANONICAL_DEVELOPMENT_DATASET_ID", manifest.dataset_id)
    monkeypatch.setattr(
        banks,
        "CANONICAL_DEVELOPMENT_DIGEST",
        manifest.scientific_digest,
    )


def _synthetic_partition(
    base: ObservablePartition,
    partition: DatasetPartition,
    count: int,
) -> ObservablePartition:
    episode_ids = tuple(f"episode-{index + 1000:016x}" for index in range(count))
    lineage_ids = tuple(f"lineage-{index + 5000:016x}" for index in range(count))
    windows = tuple(
        tuple(
            record.model_copy(
                update={
                    "acquisition_id": episode_ids[index],
                    "window_id": f"window-{index:04d}-{ordinal:02d}",
                }
            )
            for ordinal, record in enumerate(base.windows[index % len(base.windows)])
        )
        for index in range(count)
    )
    template_features = np.asarray(base.features[0], dtype=np.float64)
    features = np.stack(
        [template_features + (index * 1.0e-6) for index in range(count)]
    )
    valid_mask = np.ones((count, features.shape[1]), dtype=np.bool_)
    return replace(
        base,
        partition=partition,
        episode_ids=episode_ids,
        lineage_ids=lineage_ids,
        features=features,
        valid_mask=valid_mask,
        windows=windows,
    )


def _training_labels(observations: ObservablePartition) -> tuple[EpisodeLabel, ...]:
    half = len(observations.episode_ids) // 2
    return tuple(
        EpisodeLabel(
            episode_id=episode_id,
            lineage_id=observations.lineage_ids[index],
            target=-1 if index < half else 1,
        )
        for index, episode_id in enumerate(observations.episode_ids)
    )


def test_training_bank_is_fixed_balanced_and_feature_independent(
    bank_source,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, base = bank_source
    _approve_fixture_identity(manifest, monkeypatch)
    train = _synthetic_partition(base, DatasetPartition.TRAIN, 72)
    labels = _training_labels(train)
    first = select_training_quantum_bank(
        train,
        labels,
        manifest,
        encoder_artifact_id=ENCODER_ID,
    )
    repeated = select_training_quantum_bank(
        train,
        labels,
        manifest,
        encoder_artifact_id=ENCODER_ID,
    )
    perturbed = replace(train, features=np.full_like(train.features, 999_999.0))
    independent = select_training_quantum_bank(
        perturbed,
        labels,
        manifest,
        encoder_artifact_id=ENCODER_ID,
    )
    labels_by_id = {item.episode_id: item.target for item in labels}

    assert first == repeated == independent
    assert first.row_count == 32
    assert first.lineage_count == 32
    assert first.class_balance == {"-1": 16, "1": 16}
    assert first.selection_seed == TRAIN_SELECTION_SEED
    assert len({item.episode_id for item in first.rows}) == 32
    assert len({item.lineage_id for item in first.rows}) == 32
    assert {item.partition for item in first.rows} == {DatasetPartition.TRAIN}
    assert {item.window_ordinal for item in first.rows} == {CENTRAL_WINDOW_ORDINAL}
    assert sum(labels_by_id[item.episode_id] == -1 for item in first.rows) == 16
    assert sum(labels_by_id[item.episode_id] == 1 for item in first.rows) == 16


def test_bank_selection_fails_if_fixed_window_is_not_eligible(
    bank_source,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, base = bank_source
    _approve_fixture_identity(manifest, monkeypatch)
    train = _synthetic_partition(base, DatasetPartition.TRAIN, 72)
    labels = _training_labels(train)
    valid_mask = train.valid_mask.copy()
    valid_mask[:, CENTRAL_WINDOW_ORDINAL] = False

    with pytest.raises(ValueError, match="not quantum-eligible"):
        select_training_quantum_bank(
            replace(train, valid_mask=valid_mask),
            labels,
            manifest,
            encoder_artifact_id=ENCODER_ID,
        )


def test_validation_bank_includes_all_lineages_without_labels(
    bank_source,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, base = bank_source
    _approve_fixture_identity(manifest, monkeypatch)
    validation = _synthetic_partition(base, DatasetPartition.VALIDATION, 24)
    bank = select_validation_quantum_bank(
        validation,
        manifest,
        encoder_artifact_id=ENCODER_ID,
    )

    assert bank.row_count == 24
    assert bank.lineage_count == 24
    assert bank.class_balance is None
    assert bank.selection_seed is None
    assert {item.episode_id for item in bank.rows} == set(validation.episode_ids)
    assert {item.partition for item in bank.rows} == {DatasetPartition.VALIDATION}
    assert {item.window_ordinal for item in bank.rows} == {CENTRAL_WINDOW_ORDINAL}


def test_no_test_bank_can_be_selected(
    bank_source,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, base = bank_source
    _approve_fixture_identity(manifest, monkeypatch)
    test_observations = _synthetic_partition(base, DatasetPartition.TEST, 24)

    with pytest.raises(ValueError, match="validation observations"):
        select_validation_quantum_bank(
            test_observations,
            manifest,
            encoder_artifact_id=ENCODER_ID,
        )


def test_bank_artifact_is_immutable_and_corruption_is_rejected(
    tmp_path: Path,
    bank_source,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, base = bank_source
    _approve_fixture_identity(manifest, monkeypatch)
    validation = _synthetic_partition(base, DatasetPartition.VALIDATION, 24)
    bank = select_validation_quantum_bank(
        validation,
        manifest,
        encoder_artifact_id=ENCODER_ID,
    )
    path = write_bank_artifact(bank, root=tmp_path)
    loaded = load_bank_artifact(
        path,
        expected_artifact_id=bank.artifact_id,
        expected_dataset_id=manifest.dataset_id,
        expected_dataset_digest=manifest.scientific_digest,
        expected_encoder_artifact_id=ENCODER_ID,
    )

    assert loaded == bank
    assert "target" not in (path / "bank.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        write_bank_artifact(bank, root=tmp_path)

    target = path / "bank.json"
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="size or path"):
        load_bank_artifact(
            path,
            expected_artifact_id=bank.artifact_id,
            expected_dataset_id=manifest.dataset_id,
            expected_dataset_digest=manifest.scientific_digest,
            expected_encoder_artifact_id=ENCODER_ID,
        )
