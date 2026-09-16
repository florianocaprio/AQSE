from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.paper_study.datasets import positive_control_dataset, split_binary_dataset
from app.paper_study.reporting import build_aggregate_report, write_aggregate_report
from app.paper_study.study import permutation_control, run_replicated_study


def _fixture_factory(dataset_seed: int, split_seed: int):
    rng = np.random.default_rng(dataset_seed)
    X = rng.normal(size=(48, 8))
    score = X[:, 0] + 0.7 * X[:, 1] * X[:, 2] - 0.4 * X[:, 3]
    order = np.argsort(score)
    y = np.empty(48, dtype=np.int8)
    y[order[:24]] = -1
    y[order[24:]] = 1
    return split_binary_dataset(X, y, split_seed=split_seed)


def _positive_factory(dataset_seed: int, split_seed: int):
    X, y = positive_control_dataset(80, dataset_seed)
    return split_binary_dataset(X, y, split_seed=split_seed)


def test_two_replica_end_to_end_workflow_closes_every_test_ledger(tmp_path: Path) -> None:
    result = run_replicated_study(
        n_replicas=2,
        base_seed=8801,
        output_dir=tmp_path / "study",
        dataset_factory=_fixture_factory,
        qng_steps=1,
        bootstrap_resamples=200,
    )
    assert result.replica_count == 2
    assert len(result.replicas) == 2
    assert set(result.aggregate.delta_by_baseline) == {
        "rbf_svc",
        "mlp_11_parameter",
        "rff_256",
        "gradient_boosting",
    }
    for replica in result.replicas:
        events = [
            json.loads(line)["event"]
            for line in Path(replica.test_ledger_path).read_text(encoding="utf-8").splitlines()
        ]
        assert events == [
            "sealed",
            "observations_opened",
            "labels_opened",
            "evaluation_published",
        ]


def test_positive_control_separates_better_than_rbf_baseline(tmp_path: Path) -> None:
    result = run_replicated_study(
        n_replicas=2,
        base_seed=9901,
        output_dir=tmp_path / "positive",
        dataset_factory=_positive_factory,
        qng_steps=1,
        bootstrap_resamples=200,
    )
    assert result.aggregate.delta_by_baseline["rbf_svc"].delta.point > 0.0


def test_permutation_control_reports_chance_compatibility(tmp_path: Path) -> None:
    rng = np.random.default_rng(440)
    X = rng.normal(size=(200, 8))
    y = np.asarray([-1, 1] * 100, dtype=np.int8)
    rng.shuffle(y)
    result = permutation_control(
        X,
        y,
        n_replicas=12,
        base_seed=4401,
        output_dir=tmp_path / "permutation",
        qng_steps=0,
        bootstrap_resamples=200,
    )
    assert all(result.compatible_with_chance.values())


def test_aggregate_report_preserves_interval_objects_and_writes_json(tmp_path: Path) -> None:
    study = run_replicated_study(
        n_replicas=2,
        base_seed=7701,
        output_dir=tmp_path / "report-study",
        dataset_factory=_fixture_factory,
        qng_steps=0,
        bootstrap_resamples=200,
    )
    report = build_aggregate_report(study)
    interval = report["primary_study"]["aggregate"]["quantum_balanced_accuracy"]
    assert set(interval) == {
        "point",
        "lower",
        "upper",
        "confidence_level",
        "bootstrap_resamples",
    }
    path = write_aggregate_report(tmp_path / "aggregate.json", study)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == report
