from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import qiskit
import scipy
import sklearn

from app.training.canonical import file_sha256

from .execution import (
    MAIN_BASE_SEED,
    MAIN_REPLICAS,
    METHOD_ORDER,
    NEGATIVE_BASE_SEED,
    NEGATIVE_REPLICAS,
    POSITIVE_BASE_SEED,
    POSITIVE_REPLICAS,
    _kernel_aggregate_payload,
    _method_statistics,
    _paired_statistics,
)
from .models import PermutationControlResult, ReplicaResult, ReplicatedStudyResult
from .reporting import build_aggregate_report
from .storage import ReplicaStore

EXPERIMENT_SOURCE_COMMIT = "091acf2e98b2b88720730eea276e253f0c96f5a6"
EXPECTED_PROTOCOL_SHA256 = (
    "db336a0c01880da07e44687a0d6ed937cd1eee4446a6d109b96c3b3f703768c4"
)
GROUPS = (
    ("negative-control-replicas", NEGATIVE_REPLICAS, NEGATIVE_BASE_SEED, True),
    ("positive-control-replicas", POSITIVE_REPLICAS, POSITIVE_BASE_SEED, False),
    ("main-replicas", MAIN_REPLICAS, MAIN_BASE_SEED, False),
)
COPIED_ROOT_FILES = (
    "protocol_snapshot.json",
    "environment.json",
    "replicas.csv",
    "replicas.json",
    "kernel_diagnostics.csv",
    "kernel_diagnostics.json",
    "negative_control.json",
    "positive_control.json",
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_inventory(root: Path) -> dict[str, Any]:
    """Hash stored bytes without deserializing TEST arrays.

    The inventory proves byte identity at two points in recovery.  It does not
    re-evaluate predictions, labels, kernels, or any scientific result.
    """

    if not root.is_dir():
        raise FileNotFoundError(f"recovery source directory does not exist: {root}")
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "byte_count": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema_version": "aqse.paper-source-inventory.v1",
        "file_count": len(files),
        "files": files,
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON evidence: {path}") from error


def _verify_environment(environment: Mapping[str, Any]) -> None:
    expected = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "qiskit": qiskit.__version__,
    }
    mismatches = {
        name: {"execution": environment.get(name), "recovery": value}
        for name, value in expected.items()
        if environment.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"scientific package version mismatch: {mismatches}")


def _verify_replica(
    directory: Path,
    expected: ReplicaResult,
    *,
    expected_index: int,
    base_seed: int,
    permutation: bool,
) -> None:
    required = {
        "model-selection-freeze.json",
        "protocol.json",
        "result.json",
        "test-access-ledger.jsonl",
        "test-labels.npy",
        "test-observations.npy",
    }
    names = {path.name for path in directory.iterdir() if path.is_file()}
    if not required.issubset(names):
        raise ValueError(f"replica {directory.name} is missing required evidence")
    result_path = directory / "result.json"
    stored = ReplicaResult.model_validate(_load_json(result_path))
    if stored != expected:
        raise ValueError(f"replica {directory.name} contradicts its aggregate")
    if stored.replica_index != expected_index or stored.base_seed != base_seed:
        raise ValueError(f"replica {directory.name} identity is inconsistent")
    if stored.failure is not None:
        raise ValueError(f"replica {directory.name} contains a hidden failure")

    protocol_path = directory / "protocol.json"
    protocol = _load_json(protocol_path)
    if protocol.get("replica_index") != expected_index:
        raise ValueError(f"replica {directory.name} protocol index is inconsistent")
    if protocol.get("seeds") != stored.seeds.model_dump(mode="json"):
        raise ValueError(f"replica {directory.name} seed evidence is inconsistent")
    if protocol.get("split_sizes") != stored.split_sizes.model_dump(mode="json"):
        raise ValueError(f"replica {directory.name} split evidence is inconsistent")
    if protocol.get("training_label_permutation_control") is not permutation:
        raise ValueError(f"replica {directory.name} family is inconsistent")

    entries = ReplicaStore(directory).verify_closed()
    if [entry["sequence"] for entry in entries] != [0, 1, 2, 3]:
        raise ValueError(f"replica {directory.name} ledger sequence is inconsistent")
    sealed = entries[0]["details"]
    byte_hashes = {
        "test_observation_sha256": file_sha256(directory / "test-observations.npy"),
        "test_label_sha256": file_sha256(directory / "test-labels.npy"),
        "protocol_sha256": file_sha256(protocol_path),
    }
    if sealed != byte_hashes:
        raise ValueError(f"replica {directory.name} sealed payload hash is inconsistent")
    if entries[-1]["details"].get("result_sha256") != file_sha256(result_path):
        raise ValueError(f"replica {directory.name} result hash is inconsistent")


def validate_recovery_inputs(source: Path) -> dict[str, Any]:
    """Validate persisted evidence without loading TEST arrays or invoking science.

    Successful validation demonstrates internal artifact consistency only.  It
    does not repeat or independently validate model predictions.
    """

    protocol = _load_json(source / "protocol_snapshot.json")
    environment = _load_json(source / "environment.json")
    if protocol.get("source_commit") != EXPERIMENT_SOURCE_COMMIT:
        raise ValueError("experiment source commit does not match the frozen baseline")
    if protocol.get("protocol_document_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("protocol document hash does not match the preregistration")
    if environment.get("source_commit") != EXPERIMENT_SOURCE_COMMIT:
        raise ValueError("environment source identity is inconsistent")
    _verify_environment(environment)

    studies: dict[str, ReplicatedStudyResult] = {}
    common_source_hashes: dict[str, str] | None = None
    for group_name, count, base_seed, permutation in GROUPS:
        group = source / group_name
        expected_names = {f"replica-{index:03d}" for index in range(count)}
        actual_names = {path.name for path in group.glob("replica-*") if path.is_dir()}
        if actual_names != expected_names:
            raise ValueError(f"{group_name} replica indices are missing or duplicated")
        study = ReplicatedStudyResult.model_validate(_load_json(group / "aggregate.json"))
        if study.replica_count != count or study.base_seed != base_seed:
            raise ValueError(f"{group_name} aggregate identity is inconsistent")
        if study.qng_steps != 10:
            raise ValueError(f"{group_name} QNG budget differs from the frozen protocol")
        for index, replica in enumerate(study.replicas):
            _verify_replica(
                group / f"replica-{index:03d}",
                replica,
                expected_index=index,
                base_seed=base_seed,
                permutation=permutation,
            )
            if common_source_hashes is None:
                common_source_hashes = replica.source_hashes
            elif replica.source_hashes != common_source_hashes:
                raise ValueError("replica source hashes are not identical across the execution")
        studies[group_name] = study

    negative = PermutationControlResult.model_validate(
        _load_json(source / "negative_control.json")
    )
    if negative.study != studies["negative-control-replicas"]:
        raise ValueError("negative-control summary contradicts its stored aggregate")
    positive = ReplicatedStudyResult.model_validate(_load_json(source / "positive_control.json"))
    if positive != studies["positive-control-replicas"]:
        raise ValueError("positive-control summary contradicts its stored aggregate")
    main = studies["main-replicas"]
    replicas = tuple(ReplicaResult.model_validate(item) for item in _load_json(source / "replicas.json"))
    if replicas != main.replicas:
        raise ValueError("main replica summary contradicts its stored aggregate")
    return {
        "protocol": protocol,
        "environment": environment,
        "negative": negative,
        "positive": positive,
        "main": main,
        "source_hashes": common_source_hashes or {},
    }


def _aggregate_payload(main: ReplicatedStudyResult) -> dict[str, Any]:
    return {
        "method_statistics": _method_statistics(main),
        "paired_statistics": _paired_statistics(main),
        "kernel_aggregate": _kernel_aggregate_payload(main),
    }


def _manifest(root: Path, source_inventory: Mapping[str, Any]) -> dict[str, Any]:
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "completion.json"}
    )
    return {
        "schema_version": "aqse.paper-report-recovery-manifest.v1",
        "source_inventory_sha256": _sha256_bytes(_json_bytes(source_inventory)),
        "source_files": source_inventory["files"],
        "output_files": [
            {
                "relative_path": name,
                "byte_count": (root / name).stat().st_size,
                "sha256": file_sha256(root / name),
            }
            for name in names
        ],
    }


def verify_recovery_package(output: Path) -> dict[str, Any]:
    """Verify a completed report package; this never reads scientific TEST arrays."""

    completion_path = output / "completion.json"
    manifest_path = output / "manifest.json"
    if not completion_path.is_file() or not manifest_path.is_file():
        raise FileExistsError(f"incomplete recovery output already exists: {output}")
    completion = _load_json(completion_path)
    manifest = _load_json(manifest_path)
    if completion.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("recovery completion marker does not match the manifest")
    for item in manifest.get("output_files", []):
        path = output / item["relative_path"]
        if not path.is_file() or path.stat().st_size != item["byte_count"]:
            raise ValueError(f"recovery output is missing or truncated: {path}")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"recovery output hash mismatch: {path}")
    for name in (
        "aggregate_statistics.json",
        "final_report.json",
        "recovery_record.json",
        "source_inventory.json",
    ):
        _load_json(output / name)
    return {"reused": True, "manifest_sha256": file_sha256(manifest_path)}


def recover_completed_study(
    *,
    source: Path,
    output: Path,
    reporting_source_commit: str,
) -> dict[str, Any]:
    """Finalize persisted results only; this cannot run training or reopen TEST.

    The function validates and packages already-published scores.  It does not
    generate datasets, fit models, predict, or establish field/QPU advantage.
    """

    if len(reporting_source_commit) != 40 or any(
        char not in "0123456789abcdef" for char in reporting_source_commit
    ):
        raise ValueError("reporting_source_commit must be a full lowercase Git SHA")
    source = source.resolve()
    output = output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output must be separate sibling trees")
    if output.exists():
        result = verify_recovery_package(output)
        record = _load_json(output / "recovery_record.json")
        if record.get("experiment_source_commit") != EXPERIMENT_SOURCE_COMMIT:
            raise ValueError("existing recovery package has the wrong experiment identity")
        if record.get("reporting_source_commit") != reporting_source_commit:
            raise ValueError("existing recovery package has the wrong reporting identity")
        return {**result, "output": str(output)}

    started = perf_counter()
    inventory_before = build_inventory(source)
    validated = validate_recovery_inputs(source)
    main: ReplicatedStudyResult = validated["main"]
    negative: PermutationControlResult = validated["negative"]
    positive: ReplicatedStudyResult = validated["positive"]
    aggregate = _aggregate_payload(main)
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
    runtime_sums = {
        name: float(sum(replica.runtime_seconds["total"] for replica in study.replicas))
        for name, study in (
            ("main", main),
            ("negative_control", negative.study),
            ("positive_control", positive),
        )
    }
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        for name in COPIED_ROOT_FILES:
            shutil.copyfile(source / name, staging / name)
        shutil.copyfile(source / "main-replicas" / "aggregate.json", staging / "main_study.json")
        _write_json(staging / "source_inventory.json", inventory_before)
        _write_json(staging / "aggregate_statistics.json", aggregate)
        aggregate_rows = [
            {"method": method, "metric": metric, **values["mean_interval"]}
            for method in METHOD_ORDER
            for metric, values in aggregate["method_statistics"][method].items()
        ]
        _write_csv(staging / "aggregate_statistics.csv", aggregate_rows)
        final_report = {
            "schema_version": "aqse.paper-final-report.recovery.v1",
            "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
            "reporting_source_commit": reporting_source_commit,
            "protocol": validated["protocol"],
            "execution_environment": validated["environment"],
            "recovery_environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "qiskit": qiskit.__version__,
            },
            "controls": build_aggregate_report(
                main, permutation=negative, positive_control=positive
            ),
            "main_aggregate": aggregate,
            "completed_replicas": {"main": 30, "negative_control": 12, "positive_control": 12},
            "failed_replicas": {
                "main": [item.replica_index for item in main.replicas if item.failure],
                "negative_control": [
                    item.replica_index for item in negative.study.replicas if item.failure
                ],
                "positive_control": [
                    item.replica_index for item in positive.replicas if item.failure
                ],
            },
            "saved_replica_runtime_seconds_sum": runtime_sums,
            "original_execution_wall_time_seconds": None,
            "original_execution_wall_time_note": (
                "unavailable because the original process failed during final serialization"
            ),
            "conclusion": conclusion,
            "limitations": [
                "exact-state simulator only; no QPU hardware evidence",
                "synthetic sensor domain only; no field-deployment validity",
                "positive control is mechanism-aligned and is not real-sensor evidence",
                "parameter-count and feature-space matching do not equalize hypothesis classes",
                "report recovery packages persisted scores and does not repeat predictions",
            ],
        }
        _write_json(staging / "final_report.json", final_report)
        markdown = [
            "# AQSE preregistered replicated study — recovered final report",
            "",
            f"- Experiment source commit: `{EXPERIMENT_SOURCE_COMMIT}`",
            f"- Reporting source commit: `{reporting_source_commit}`",
            "- Completed replicas: `30 main + 12 negative + 12 positive`",
            f"- Conclusion: **{conclusion}**",
            "- Original execution wall time: unavailable; saved per-replica sums are in JSON",
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
        (staging / "final_report.md").write_text("\n".join(markdown), encoding="utf-8")
        recovery_record = {
            "schema_version": "aqse.paper-report-recovery-record.v1",
            "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
            "reporting_source_commit": reporting_source_commit,
            "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
            "recovery_seconds": perf_counter() - started,
            "source_directory": str(source),
            "output_directory": str(output),
            "input_file_count": inventory_before["file_count"],
            "replicas_recovered": {"main": 30, "negative_control": 12, "positive_control": 12},
            "scientific_operations": {
                "new_replicas": 0,
                "dataset_generation": 0,
                "training": 0,
                "predictions": 0,
                "test_observation_opens": 0,
                "test_label_opens": 0,
            },
            "copied_byte_identical": list(COPIED_ROOT_FILES),
            "produced_first_time": [
                "aggregate_statistics.json",
                "aggregate_statistics.csv",
                "final_report.json",
                "final_report.md",
                "manifest.json",
                "recovery_record.json",
                "source_inventory.json",
            ],
        }
        _write_json(staging / "recovery_record.json", recovery_record)

        inventory_after = build_inventory(source)
        if inventory_after != inventory_before:
            raise RuntimeError("source evidence changed during report recovery")
        manifest = _manifest(staging, inventory_before)
        _write_json(staging / "manifest.json", manifest)
        loaded_manifest = _load_json(staging / "manifest.json")
        if loaded_manifest != manifest:
            raise RuntimeError("manifest load-back verification failed")
        for item in manifest["output_files"]:
            if file_sha256(staging / item["relative_path"]) != item["sha256"]:
                raise RuntimeError("staged output changed during manifest verification")
        _write_json(
            staging / "completion.json",
            {
                "schema_version": "aqse.paper-report-recovery-complete.v1",
                "manifest_sha256": file_sha256(staging / "manifest.json"),
            },
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    verified = verify_recovery_package(output)
    return {**verified, "reused": False, "output": str(output)}
