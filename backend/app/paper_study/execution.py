from __future__ import annotations

import csv
import json
import platform
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import qiskit
import scipy
import sklearn

from app.training.canonical import file_sha256

from .datasets import positive_control_dataset, split_binary_dataset
from .models import ReplicatedStudyResult
from .reporting import build_aggregate_report
from .statistics import bootstrap_mean_interval, paired_wilcoxon
from .study import permutation_control, run_replicated_study

MAIN_REPLICAS = 30
MAIN_BASE_SEED = 3_001_001
NEGATIVE_REPLICAS = 12
NEGATIVE_BASE_SEED = 3_002_001
POSITIVE_REPLICAS = 12
POSITIVE_BASE_SEED = 3_003_001
CONTROL_DATASET_SIZE = 200
CONTROL_DATASET_SEED = 3_004_001
BOOTSTRAP_RESAMPLES = 2_000


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _positive_factory(dataset_seed: int, split_seed: int):
    X, y = positive_control_dataset(CONTROL_DATASET_SIZE, dataset_seed)
    return split_binary_dataset(X, y, split_seed=split_seed)


def _interval(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    return bootstrap_mean_interval(
        values, seed=seed, resamples=BOOTSTRAP_RESAMPLES
    ).model_dump(mode="json")


def _method_statistics(study: ReplicatedStudyResult) -> dict[str, Any]:
    methods = tuple(study.replicas[0].test_metrics)
    result: dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        result[method] = {}
        for metric_index, metric in enumerate(("balanced_accuracy", "macro_f1")):
            values = [
                float(getattr(replica.test_metrics[method], metric))
                for replica in study.replicas
            ]
            result[method][metric] = {
                "mean_interval": _interval(
                    values,
                    seed=MAIN_BASE_SEED + 10_000 + method_index * 10 + metric_index,
                ),
                "median": float(np.median(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
    return result


def _paired_statistics(study: ReplicatedStudyResult) -> dict[str, Any]:
    quantum = {
        metric: [float(getattr(item.test_metrics["quantum"], metric)) for item in study.replicas]
        for metric in ("balanced_accuracy", "macro_f1")
    }
    result: dict[str, Any] = {}
    baselines = tuple(name for name in study.replicas[0].test_metrics if name != "quantum")
    for baseline_index, baseline in enumerate(baselines):
        result[baseline] = {}
        for metric_index, metric in enumerate(("balanced_accuracy", "macro_f1")):
            classical = [
                float(getattr(item.test_metrics[baseline], metric)) for item in study.replicas
            ]
            deltas = [
                left - right
                for left, right in zip(quantum[metric], classical, strict=True)
            ]
            result[baseline][metric] = {
                "delta_mean_interval": _interval(
                    deltas,
                    seed=MAIN_BASE_SEED + 20_000 + baseline_index * 10 + metric_index,
                ),
                "delta_median": float(np.median(deltas)),
                "delta_standard_deviation": float(np.std(deltas, ddof=1)),
                "delta_minimum": float(np.min(deltas)),
                "delta_maximum": float(np.max(deltas)),
                "proportion_delta_greater_than_zero": float(np.mean(np.asarray(deltas) > 0.0)),
                "wilcoxon": paired_wilcoxon(
                    quantum[metric], classical
                ).model_dump(mode="json"),
            }
    return result


def _replica_rows(study: ReplicatedStudyResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replica in study.replicas:
        for method, metrics in replica.test_metrics.items():
            rows.append(
                {
                    "replica_index": replica.replica_index,
                    "method": method,
                    "balanced_accuracy": metrics.balanced_accuracy,
                    "macro_f1": metrics.macro_f1,
                    "replica_seed": replica.seeds.replica,
                    "dataset_seed": replica.seeds.dataset,
                    "split_seed": replica.seeds.split,
                    "qng_seed": replica.seeds.qng,
                    "classical_seed": replica.seeds.classical,
                    "train_size": replica.split_sizes.train,
                    "validation_size": replica.split_sizes.validation,
                    "test_size": replica.split_sizes.test,
                    "runtime_seconds": replica.runtime_seconds["total"],
                    "failure": replica.failure or "",
                }
            )
    return rows


def _kernel_rows(study: ReplicatedStudyResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replica in study.replicas:
        for checkpoint, diagnostics in enumerate(replica.kernel_diagnostics_by_checkpoint):
            rows.append(
                {
                    "replica_index": replica.replica_index,
                    "checkpoint_index": checkpoint,
                    "selected": checkpoint == replica.quantum_selection.checkpoint_index,
                    "effective_rank": diagnostics.effective_rank,
                    "spectral_concentration": diagnostics.maximum_eigenvalue_trace_ratio,
                    "minimum_eigenvalue": diagnostics.minimum_eigenvalue,
                    "maximum_eigenvalue": diagnostics.maximum_eigenvalue,
                    "condition_number": diagnostics.positive_condition_number,
                    "diagonal_minimum": diagnostics.diagonal_minimum,
                    "diagonal_maximum": diagnostics.diagonal_maximum,
                    "symmetry_error": diagnostics.symmetry_max_abs_error,
                    "off_diagonal_mean": diagnostics.off_diagonal_mean,
                    "off_diagonal_standard_deviation": diagnostics.off_diagonal_standard_deviation,
                }
            )
    return rows


def _manifest(root: Path, names: Sequence[str]) -> dict[str, Any]:
    files = []
    for name in names:
        path = root / name
        files.append(
            {"relative_path": name, "byte_count": path.stat().st_size, "sha256": file_sha256(path)}
        )
    return {"schema_version": "aqse.paper-result-manifest.v1", "files": files}


def execute_preregistered_study(
    *,
    artifact_root: Path,
    source_commit: str,
    protocol_document_sha256: str,
) -> Path:
    """Execute the frozen controls and N=30 study without adaptive decisions.

    This offline simulator run can support only the pre-registered statistical
    conclusion.  It does not demonstrate field validity or hardware advantage.
    """

    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    if len(protocol_document_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in protocol_document_sha256
    ):
        raise ValueError("protocol_document_sha256 must be a lowercase SHA-256")
    execution_id = f"execution-{source_commit[:12]}-{MAIN_BASE_SEED}"
    root = artifact_root / "paper-replicated-study-v1" / execution_id
    if root.exists():
        raise FileExistsError(f"scientific execution already exists: {root}")
    root.mkdir(parents=True)
    started = perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    protocol = {
        "schema_version": "aqse.paper-execution-protocol.v1",
        "source_commit": source_commit,
        "timestamp_utc": timestamp,
        "main": {"replicas": MAIN_REPLICAS, "base_seed": MAIN_BASE_SEED},
        "negative_control": {
            "replicas": NEGATIVE_REPLICAS,
            "base_seed": NEGATIVE_BASE_SEED,
            "dataset_seed": CONTROL_DATASET_SEED,
            "dataset_size": CONTROL_DATASET_SIZE,
        },
        "positive_control": {
            "replicas": POSITIVE_REPLICAS,
            "base_seed": POSITIVE_BASE_SEED,
            "dataset_size_per_replica": CONTROL_DATASET_SIZE,
        },
        "qng_steps": 10,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "primary_advantage_criterion": (
            "RBF-SVC paired TEST balanced-accuracy delta bootstrap lower bound > 0 "
            "and two-sided Wilcoxon p < 0.05"
        ),
        "protocol_document_sha256": protocol_document_sha256,
    }
    _write_json(root / "protocol_snapshot.json", protocol)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "qiskit": qiskit.__version__,
        "source_commit": source_commit,
    }
    _write_json(root / "environment.json", environment)

    control_X, control_y = positive_control_dataset(
        CONTROL_DATASET_SIZE, CONTROL_DATASET_SEED
    )
    negative = permutation_control(
        control_X,
        control_y,
        n_replicas=NEGATIVE_REPLICAS,
        base_seed=NEGATIVE_BASE_SEED,
        output_dir=root / "negative-control-replicas",
        qng_steps=10,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    _write_json(root / "negative_control.json", negative.model_dump(mode="json"))

    positive = run_replicated_study(
        n_replicas=POSITIVE_REPLICAS,
        base_seed=POSITIVE_BASE_SEED,
        output_dir=root / "positive-control-replicas",
        dataset_factory=_positive_factory,
        qng_steps=10,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    _write_json(root / "positive_control.json", positive.model_dump(mode="json"))

    main = run_replicated_study(
        n_replicas=MAIN_REPLICAS,
        base_seed=MAIN_BASE_SEED,
        output_dir=root / "main-replicas",
        qng_steps=10,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    if len(main.replicas) != MAIN_REPLICAS:
        raise RuntimeError("main study did not complete exactly 30 replicas")

    _write_json(root / "replicas.json", [item.model_dump(mode="json") for item in main.replicas])
    _write_csv(root / "replicas.csv", _replica_rows(main))
    kernel_rows = _kernel_rows(main)
    _write_json(root / "kernel_diagnostics.json", kernel_rows)
    _write_csv(root / "kernel_diagnostics.csv", kernel_rows)
    aggregate = {
        "method_statistics": _method_statistics(main),
        "paired_statistics": _paired_statistics(main),
        "kernel_aggregate": main.aggregate.selected_kernel_diagnostics,
    }
    _write_json(root / "aggregate_statistics.json", aggregate)
    aggregate_rows = []
    for method, metrics in aggregate["method_statistics"].items():
        for metric, values in metrics.items():
            aggregate_rows.append({"method": method, "metric": metric, **values["mean_interval"]})
    _write_csv(root / "aggregate_statistics.csv", aggregate_rows)

    primary = aggregate["paired_statistics"]["rbf_svc"]["balanced_accuracy"]
    supports = (
        primary["delta_mean_interval"]["lower"] > 0.0
        and primary["wilcoxon"]["p_value"] < 0.05
    )
    conclusion = (
        "evidence supports the preregistered quantum-advantage criterion"
        if supports
        else "evidence does not support the preregistered quantum-advantage criterion"
    )
    final_report = {
        "schema_version": "aqse.paper-final-report.v1",
        "source_commit": source_commit,
        "protocol": protocol,
        "environment": environment,
        "controls": build_aggregate_report(main, permutation=negative, positive_control=positive),
        "main_aggregate": aggregate,
        "completed_main_replicas": len(main.replicas),
        "all_main_replica_seeds": [
            item.seeds.model_dump(mode="json") for item in main.replicas
        ],
        "failed_replicas": [item.replica_index for item in main.replicas if item.failure],
        "runtime_seconds": perf_counter() - started,
        "conclusion": conclusion,
        "limitations": [
            "exact-state simulator only; no QPU hardware evidence",
            "synthetic sensor domain only; no field-deployment validity",
            "positive control is mechanism-aligned and is not real-sensor evidence",
            "parameter-count and feature-space matching do not equalize hypothesis classes",
        ],
    }
    _write_json(root / "final_report.json", final_report)
    markdown = [
        "# AQSE preregistered replicated study — final report",
        "",
        f"- Source commit: `{source_commit}`",
        f"- Completed main replicas: `{len(main.replicas)}`",
        f"- Runtime seconds: `{final_report['runtime_seconds']}`",
        f"- Conclusion: **{conclusion}**",
        "",
        "## Primary paired comparison",
        "",
        "```json",
        json.dumps(primary, indent=2, sort_keys=True, allow_nan=False),
        "```",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in final_report["limitations"]],
        "",
    ]
    (root / "final_report.md").write_text("\n".join(markdown), encoding="utf-8")
    manifest_names = (
        "protocol_snapshot.json",
        "environment.json",
        "replicas.csv",
        "replicas.json",
        "kernel_diagnostics.csv",
        "kernel_diagnostics.json",
        "negative_control.json",
        "positive_control.json",
        "aggregate_statistics.json",
        "aggregate_statistics.csv",
        "final_report.json",
        "final_report.md",
    )
    _write_json(root / "manifest.json", _manifest(root, manifest_names))
    return root
