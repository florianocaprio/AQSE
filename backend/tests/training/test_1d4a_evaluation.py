from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.quantum.user_pipeline.tqk8 import StateEngine
from app.training.evaluation import (
    _circular_features,
    _evaluate_rbf,
    _scored_candidate,
    _select,
    _snr_threshold,
    _train_gradient_checkpoints,
    build_model_selection_freeze,
)
from app.training.evaluation_assembly import assemble_comparison_input
from app.training.evaluation_models import EvaluationProtocol
from app.training.evaluation_protocol import (
    METHODS,
    build_evaluation_protocol,
    validate_evaluation_protocol,
)
from app.training.evaluation_storage import (
    load_comparative_evaluation,
    load_evaluation_protocol,
    load_model_selection_freeze,
    write_evaluation_protocol,
)
from app.training.held_out import LABEL_REASON, OBSERVATION_REASON
from app.training.held_out_storage import read_test_ledger_opaque
from app.training.storage import artifact_root, verify_archive_opaque


def test_protocol_is_deterministic_finite_and_keeps_test_sealed() -> None:
    first = build_evaluation_protocol()
    second = build_evaluation_protocol()

    assert first == second
    assert first.seed_schedule == (1_001_005, 1_001_006, 1_001_007, 1_001_008, 1_001_009)
    assert first.methods == METHODS
    assert len(first.rbf_c_grid) * len(first.rbf_gamma_grid) == 9
    assert len(first.quantum_svc_c_grid) == 3
    assert first.maximum_updates == 10
    assert first.validation_role == "selection-only"
    assert first.test_state_required == "sealed"
    assert not first.create_test_bank
    validate_evaluation_protocol(first)

    with pytest.raises(ValidationError, match="seed schedule"):
        EvaluationProtocol.model_validate(
            first.model_copy(update={"seed_schedule": (1, 2, 3, 4, 5)}).model_dump(
                mode="json"
            )
        )


def test_protocol_storage_is_immutable_and_loads_back(tmp_path: Path) -> None:
    protocol = build_evaluation_protocol()
    path, execution = write_evaluation_protocol(protocol, root=tmp_path)
    loaded, loaded_execution = load_evaluation_protocol(
        path,
        expected_protocol_id=protocol.protocol_id,
    )

    assert loaded == protocol
    assert loaded_execution == execution
    assert write_evaluation_protocol(protocol, root=tmp_path)[0] == path
    tampered = protocol.model_copy(update={"content_digest": "0" * 64})
    with pytest.raises(ValueError, match="protocol digest"):
        write_evaluation_protocol(tampered, root=tmp_path)


def test_snr_threshold_uses_train_only_and_has_a_deterministic_tie_break() -> None:
    train_snr = np.asarray([9.0, 8.0, 3.0, 2.0], dtype=np.float64)
    train_labels = np.asarray([-1, -1, 1, 1], dtype=np.int8)
    threshold = _snr_threshold(train_snr, train_labels)

    assert threshold == 5.5
    assert _snr_threshold(train_snr, train_labels) == threshold


def test_circular_classical_features_are_periodic_and_do_not_add_information() -> None:
    raw = np.zeros((2, 8), dtype=np.float64)
    raw[0, 1] = -np.pi + 0.01
    raw[1, 1] = np.pi + 0.01
    transformed = _circular_features(raw)

    assert transformed.shape == (2, 9)
    np.testing.assert_allclose(transformed[0, 1:3], transformed[1, 1:3], atol=1.0e-14)
    np.testing.assert_array_equal(transformed[:, 3:], raw[:, 2:])


def test_rbf_predictions_are_independent_from_validation_labels() -> None:
    rng = np.random.default_rng(404)
    train_raw = rng.normal(size=(32, 8))
    validation_raw = rng.normal(size=(24, 8))
    train_labels = np.asarray([-1] * 16 + [1] * 16, dtype=np.int8)
    validation_labels = np.asarray([-1] * 12 + [1] * 12, dtype=np.int8)
    protocol = build_evaluation_protocol()

    class Comparison:
        X_train_raw = train_raw
        X_validation_raw = validation_raw
        y_train = train_labels
        y_validation = validation_labels

    circular_train = _circular_features(train_raw)
    circular_validation = _circular_features(validation_raw)
    mean = circular_train.mean(axis=0)
    scale = circular_train.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    first = _evaluate_rbf(
        Comparison(),  # type: ignore[arg-type]
        protocol,
        (circular_train - mean) / scale,
        (circular_validation - mean) / scale,
    )
    Comparison.y_validation = validation_labels[::-1]
    second = _evaluate_rbf(
        Comparison(),  # type: ignore[arg-type]
        protocol,
        (circular_train - mean) / scale,
        (circular_validation - mean) / scale,
    )

    assert [item.artifact.validation_prediction_digest for item in first] == [
        item.artifact.validation_prediction_digest for item in second
    ]
    assert [item.artifact.validation_score_digest for item in first] == [
        item.artifact.validation_score_digest for item in second
    ]


def test_selection_rule_prefers_metric_then_lower_complexity() -> None:
    labels = np.asarray([-1, -1, 1, 1], dtype=np.int8)
    predictions = labels.copy()
    scores = labels.astype(np.float64)
    protocol = build_evaluation_protocol()
    late = _scored_candidate(
        method="qng_tqk",
        y_train=labels,
        train_predictions=predictions,
        train_scores=scores,
        y_validation=labels,
        validation_predictions=predictions,
        validation_scores=scores,
        seed=1_001_009,
        checkpoint_index=4,
        svc_c=10.0,
        theta=np.ones(16),
        train_alignment_loss=0.2,
        validation_alignment_loss=0.2,
    )
    early = _scored_candidate(
        method="qng_tqk",
        y_train=labels,
        train_predictions=predictions,
        train_scores=scores,
        y_validation=labels,
        validation_predictions=predictions,
        validation_scores=scores,
        seed=1_001_005,
        checkpoint_index=2,
        svc_c=0.1,
        theta=np.zeros(16),
        train_alignment_loss=0.3,
        validation_alignment_loss=0.3,
    )

    selected = _select([late, early], labels, protocol)
    assert selected.selected_candidate_id == early.artifact.candidate_id
    assert selected.selected_checkpoint_index == 2
    assert selected.confidence_intervals[0].resamples == 2000


def test_ordinary_gradient_checkpoints_are_deterministic_on_separate_fixture() -> None:
    rng = np.random.default_rng(505)
    X = rng.normal(size=(4, 8))
    y = np.asarray([-1, -1, 1, 1], dtype=np.int8)
    protocol = build_evaluation_protocol()
    first = _train_gradient_checkpoints(X, y, 1_001_005, protocol)
    second = _train_gradient_checkpoints(X, y, 1_001_005, protocol)

    assert len(first) == 11
    assert len(second) == 11
    for left, right in zip(first, second, strict=True):
        assert left.seed == right.seed
        assert left.checkpoint_index == right.checkpoint_index
        assert left.train_alignment_loss == right.train_alignment_loss
        np.testing.assert_array_equal(left.theta, right.theta)


def test_quantum_cross_kernel_is_query_order_invariant() -> None:
    rng = np.random.default_rng(606)
    train = rng.normal(size=(4, 8))
    query = rng.normal(size=(3, 8))
    theta = rng.uniform(-0.8, 0.8, size=16)
    engine = StateEngine("numpy")
    expected = engine.gram(query, theta, train)
    order = np.asarray([2, 0, 1])
    reordered = engine.gram(query[order], theta, train)

    np.testing.assert_allclose(reordered, expected[order], atol=1.0e-14)


def test_published_1d4a_evidence_preserves_its_pre_test_freeze() -> None:
    root = artifact_root()
    protocol = build_evaluation_protocol()
    protocol_path = root / "evaluation-protocols" / protocol.protocol_id
    evaluations = sorted((root / "comparative-evaluations").glob("*"))
    freezes = sorted((root / "model-selection-freezes").glob("*"))
    if not protocol_path.is_dir() or not evaluations or not freezes:
        pytest.skip("canonical 1D.4a evidence has not been published")
    loaded_protocol, protocol_execution = load_evaluation_protocol(
        protocol_path,
        expected_protocol_id=protocol.protocol_id,
    )
    evaluation, evaluation_execution = load_comparative_evaluation(
        evaluations[-1],
        expected_evaluation_id=evaluations[-1].name,
    )
    freeze, freeze_execution = load_model_selection_freeze(
        freezes[-1],
        expected_freeze_id=freezes[-1].name,
    )
    assert loaded_protocol == protocol
    assert tuple(item.method for item in evaluation.selections) == METHODS
    assert build_model_selection_freeze(protocol, evaluation) == freeze
    assert protocol_execution.created_at_utc <= evaluation_execution.created_at_utc
    assert evaluation_execution.created_at_utc <= freeze_execution.created_at_utc
    dataset_path = root / "aqse-development-064acca20fc788c6"
    manifest = verify_archive_opaque(dataset_path)
    ledger = read_test_ledger_opaque(dataset_path)
    assert manifest.test_state == "sealed"
    assert evaluation.test_state_after == "sealed"
    assert freeze.test_state == "sealed"
    assert freeze.test_ledger_sha256 == (
        "210077e41754c47eebe572660f07b7cdaf8653ade2945fccd368e04d64a432c6"
    )
    if len(ledger.entries) == 1:
        assert ledger.file_sha256 == freeze.test_ledger_sha256
        assert not (root / "test-banks").exists()
    else:
        assert len(ledger.entries) == 3
        assert tuple(item.reason for item in ledger.entries[1:]) == (
            OBSERVATION_REASON,
            LABEL_REASON,
        )
        assert len(tuple((root / "test-banks").glob("aqse-test-bank-*"))) == 1
    comparison = assemble_comparison_input(
        dataset_path=root / protocol.dataset_id,
        encoder_path=root / "encoders" / protocol.encoder_id,
        train_bank_path=root / "banks" / protocol.train_bank_id,
        validation_bank_path=root / "banks" / protocol.validation_bank_id,
    )
    assert comparison.identity == evaluation.comparison_input
