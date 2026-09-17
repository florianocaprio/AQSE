from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import qiskit
import scipy
import sklearn

from app.paper_study.execution import (
    METHOD_ORDER,
    _kernel_aggregate_payload,
    _method_statistics,
)
from app.paper_study.models import (
    ClassificationMetricPair,
    IntervalEstimate,
    KernelDiagnosticsResult,
    PermutationControlResult,
    QuantumSelection,
    ReplicaResult,
    ReplicaSeeds,
    ReplicatedStudyResult,
    SplitSizes,
)
from app.paper_study.recovery import (
    EXPECTED_PROTOCOL_SHA256,
    EXPERIMENT_SOURCE_COMMIT,
    recover_completed_study,
)
from app.paper_study.statistics import aggregate_replicas, bootstrap_mean_interval
from app.training.canonical import canonical_json_bytes


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _diagnostics() -> KernelDiagnosticsResult:
    return KernelDiagnosticsResult(
        eigenvalues=(0.5, 1.5),
        trace=2.0,
        maximum_eigenvalue_trace_ratio=0.75,
        effective_rank=1.75,
        minimum_eigenvalue=0.5,
        maximum_eigenvalue=1.5,
        diagonal_minimum=1.0,
        diagonal_maximum=1.0,
        diagonal_mean=1.0,
        symmetry_max_abs_error=0.0,
        off_diagonal_minimum=0.5,
        off_diagonal_maximum=0.5,
        off_diagonal_mean=0.5,
        off_diagonal_standard_deviation=0.0,
        positive_condition_number=3.0,
    )


def _replica(index: int, base_seed: int) -> ReplicaResult:
    quantum = 0.55 + (index % 5) * 0.01
    baselines = {
        "rbf_svc": quantum - 0.01,
        "mlp_11_parameter": quantum - 0.02,
        "rff_256": quantum - 0.03,
        "gradient_boosting": quantum - 0.04,
    }
    metrics = {
        "quantum": ClassificationMetricPair(
            balanced_accuracy=quantum, macro_f1=quantum - 0.01
        ),
        **{
            name: ClassificationMetricPair(
                balanced_accuracy=value, macro_f1=value - 0.01
            )
            for name, value in baselines.items()
        },
    }
    diagnostics = _diagnostics()
    return ReplicaResult(
        replica_index=index,
        base_seed=base_seed,
        seeds=ReplicaSeeds(
            replica=10_000 + index,
            dataset=20_000 + index,
            split=30_000 + index,
            qng=40_000 + index,
            classical=50_000 + index,
            permutation=60_000 + index,
        ),
        split_sizes=SplitSizes(train=12, validation=4, test=4),
        split_identities={
            "train": tuple(f"train-{index}-{item}" for item in range(12)),
            "validation": tuple(f"validation-{index}-{item}" for item in range(4)),
            "test": tuple(f"test-{index}-{item}" for item in range(4)),
        },
        class_balance={
            "train": {"-1": 6, "1": 6},
            "validation": {"-1": 2, "1": 2},
            "test": {"-1": 2, "1": 2},
        },
        dataset_metadata={"fixture": True},
        quantum_balanced_accuracy=quantum,
        baseline_balanced_accuracy=baselines,
        delta_by_baseline={name: quantum - value for name, value in baselines.items()},
        quantum_selection=QuantumSelection(
            checkpoint_index=0,
            accepted_qng_updates=0,
            svc_c=1.0,
            theta=(0.0,) * 16,
            validation_balanced_accuracy=0.5,
            validation_macro_f1=0.5,
        ),
        kernel_diagnostics_by_checkpoint=(diagnostics,),
        selected_kernel_diagnostics=diagnostics,
        test_kernel_statistics={
            "minimum": 0.0,
            "maximum": 1.0,
            "mean": 0.5,
            "standard_deviation": 0.2,
        },
        train_metrics=metrics,
        validation_metrics=metrics,
        test_metrics=metrics,
        runtime_seconds={
            "dataset_generation": 0.1,
            "training_and_selection": 0.2,
            "test_evaluation": 0.1,
            "total": 0.4,
        },
        source_hashes={"fixture.py": "0" * 64},
        abstention_count=0,
        failure=None,
        test_ledger_path=f"replica-{index:03d}/test-access-ledger.jsonl",
    )


def _event(sequence: int, event: str, previous: str | None, details: dict) -> dict:
    scientific = {
        "schema_version": "aqse.paper-replica-test-ledger.v1",
        "sequence": sequence,
        "event": event,
        "previous_entry_sha256": previous,
        "details": details,
    }
    return {**scientific, "entry_sha256": _digest(canonical_json_bytes(scientific))}


def _write_replica(directory: Path, replica: ReplicaResult, *, permutation: bool) -> None:
    directory.mkdir(parents=True)
    protocol = {
        "schema_version": "aqse.paper-replica-protocol.v1",
        "replica_index": replica.replica_index,
        "seeds": replica.seeds.model_dump(mode="json"),
        "split_sizes": replica.split_sizes.model_dump(mode="json"),
        "qng_steps": 10,
        "training_label_permutation_control": permutation,
        "test_access_policy": "fixture",
    }
    protocol_bytes = canonical_json_bytes(protocol) + b"\n"
    observation_bytes = b"fixture-observations"
    label_bytes = b"fixture-labels"
    result_bytes = canonical_json_bytes(replica.model_dump(mode="json")) + b"\n"
    (directory / "protocol.json").write_bytes(protocol_bytes)
    (directory / "test-observations.npy").write_bytes(observation_bytes)
    (directory / "test-labels.npy").write_bytes(label_bytes)
    (directory / "model-selection-freeze.json").write_text("{}\n", encoding="utf-8")
    (directory / "result.json").write_bytes(result_bytes)
    entries = []
    for event, details in (
        (
            "sealed",
            {
                "test_observation_sha256": _digest(observation_bytes),
                "test_label_sha256": _digest(label_bytes),
                "protocol_sha256": _digest(protocol_bytes),
            },
        ),
        ("observations_opened", {}),
        ("labels_opened", {}),
        ("evaluation_published", {"result_sha256": _digest(result_bytes)}),
    ):
        previous = None if not entries else entries[-1]["entry_sha256"]
        entries.append(_event(len(entries), event, previous, details))
    (directory / "test-access-ledger.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in entries),
        encoding="utf-8",
    )


def _study(root: Path, name: str, count: int, seed: int, *, permutation: bool):
    replicas = tuple(_replica(index, seed) for index in range(count))
    study = ReplicatedStudyResult(
        base_seed=seed,
        replica_count=count,
        qng_steps=10,
        replicas=replicas,
        aggregate=aggregate_replicas(
            replicas, bootstrap_seed=seed + 1, bootstrap_resamples=100
        ),
    )
    group = root / name
    group.mkdir()
    for replica in replicas:
        _write_replica(
            group / f"replica-{replica.replica_index:03d}",
            replica,
            permutation=permutation,
        )
    (group / "aggregate.json").write_text(study.model_dump_json(indent=2) + "\n")
    return study


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    protocol = {
        "source_commit": EXPERIMENT_SOURCE_COMMIT,
        "protocol_document_sha256": EXPECTED_PROTOCOL_SHA256,
    }
    environment = {
        "source_commit": EXPERIMENT_SOURCE_COMMIT,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "qiskit": qiskit.__version__,
    }
    for name, value in (
        ("protocol_snapshot.json", protocol),
        ("environment.json", environment),
    ):
        (source / name).write_text(json.dumps(value) + "\n")
    negative_study = _study(
        source, "negative-control-replicas", 12, 3_002_001, permutation=True
    )
    positive = _study(
        source, "positive-control-replicas", 12, 3_003_001, permutation=False
    )
    main = _study(source, "main-replicas", 30, 3_001_001, permutation=False)
    estimates = {
        name: bootstrap_mean_interval(
            [replica.test_metrics[name].balanced_accuracy for replica in negative_study.replicas],
            seed=700 + index,
            resamples=100,
        )
        for index, name in enumerate(METHOD_ORDER)
    }
    negative = PermutationControlResult(
        replica_count=12,
        balanced_accuracy=estimates,
        compatible_with_chance={name: True for name in METHOD_ORDER},
        study=negative_study,
    )
    (source / "negative_control.json").write_text(negative.model_dump_json(indent=2) + "\n")
    (source / "positive_control.json").write_text(positive.model_dump_json(indent=2) + "\n")
    (source / "replicas.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in main.replicas], sort_keys=True)
        + "\n"
    )
    for name in ("replicas.csv", "kernel_diagnostics.csv"):
        (source / name).write_text("fixture\n")
    (source / "kernel_diagnostics.json").write_text("[]\n")
    return source


def test_original_nested_interval_failure_and_fixed_payload_preserve_numbers() -> None:
    interval = IntervalEstimate(
        point=0.5, lower=0.4, upper=0.6, bootstrap_resamples=2000
    )
    with pytest.raises(TypeError, match="IntervalEstimate"):
        json.dumps({"kernel_aggregate": {"trace": interval}}, allow_nan=False)
    study = ReplicatedStudyResult(
        base_seed=1,
        replica_count=2,
        qng_steps=0,
        replicas=(_replica(0, 1), _replica(1, 1)),
        aggregate=aggregate_replicas(
            (_replica(0, 1), _replica(1, 1)),
            bootstrap_seed=2,
            bootstrap_resamples=100,
        ),
    )
    payload = _kernel_aggregate_payload(study)
    loaded = json.loads(json.dumps(payload, allow_nan=False))
    assert isinstance(loaded["trace"]["point"], float)
    assert set(loaded["trace"]) == {
        "point",
        "lower",
        "upper",
        "confidence_level",
        "bootstrap_resamples",
    }


def test_statistics_use_frozen_method_order_after_sorted_json_round_trip() -> None:
    replicas = (_replica(0, 3_001_001), _replica(1, 3_001_001))
    study = ReplicatedStudyResult(
        base_seed=3_001_001,
        replica_count=2,
        qng_steps=10,
        replicas=replicas,
        aggregate=aggregate_replicas(replicas, bootstrap_seed=7, bootstrap_resamples=100),
    )
    loaded = ReplicatedStudyResult.model_validate(
        json.loads(json.dumps(study.model_dump(mode="json"), sort_keys=True))
    )
    assert tuple(_method_statistics(loaded)) == METHOD_ORDER
    assert _method_statistics(loaded) == _method_statistics(study)


def test_report_only_recovery_is_read_only_idempotent_and_blocks_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    import app.paper_study.study as scientific

    def forbidden(*args, **kwargs):
        raise AssertionError("scientific execution is forbidden during recovery")

    for name in (
        "run_replicated_study",
        "permutation_control",
        "_run_replicas",
        "_run_replica",
    ):
        monkeypatch.setattr(scientific, name, forbidden)
    monkeypatch.setattr("app.paper_study.storage.ReplicaStore.open_test_observations", forbidden)
    monkeypatch.setattr("app.paper_study.storage.ReplicaStore.open_test_labels", forbidden)
    monkeypatch.setattr("app.paper_study.storage.ReplicaStore.publish_evaluation", forbidden)
    for path in sorted(source.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    output = tmp_path / "recovered"
    first = recover_completed_study(
        source=source,
        output=output,
        reporting_source_commit="1" * 40,
    )
    output_before = {path: path.stat().st_mtime_ns for path in output.rglob("*")}
    second = recover_completed_study(
        source=source,
        output=output,
        reporting_source_commit="1" * 40,
    )
    assert first["reused"] is False
    assert second["reused"] is True
    assert output_before == {path: path.stat().st_mtime_ns for path in output.rglob("*")}
    assert before == {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    report = json.loads((output / "final_report.json").read_text())
    assert report["completed_replicas"] == {
        "main": 30,
        "negative_control": 12,
        "positive_control": 12,
    }
    assert report["experiment_source_commit"] == EXPERIMENT_SOURCE_COMMIT
    assert report["reporting_source_commit"] == "1" * 40


@pytest.mark.parametrize("damage", ("missing_replica", "corrupt_result", "altered_ledger"))
def test_recovery_rejects_incomplete_or_corrupt_evidence(
    tmp_path: Path, damage: str
) -> None:
    source = _source(tmp_path)
    replica = source / "main-replicas" / "replica-029"
    if damage == "missing_replica":
        os.rename(replica, source / "main-replicas" / "replica-missing")
    elif damage == "corrupt_result":
        (replica / "result.json").write_text("{}\n")
    else:
        ledger = replica / "test-access-ledger.jsonl"
        ledger.write_text(ledger.read_text().replace("labels_opened", "labels_tampered"))
    with pytest.raises((ValueError, FileExistsError)):
        recover_completed_study(
            source=source,
            output=tmp_path / "output",
            reporting_source_commit="2" * 40,
        )
