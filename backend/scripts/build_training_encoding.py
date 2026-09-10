from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from math import pi
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.quantum.user_pipeline.tqk8 import StateEngine  # noqa: E402
from app.training.banks import (  # noqa: E402
    select_training_quantum_bank,
    select_validation_quantum_bank,
)
from app.training.canonical import canonical_json_bytes  # noqa: E402
from app.training.encoding import (  # noqa: E402
    CANONICAL_DEVELOPMENT_DATASET_ID,
    CANONICAL_DEVELOPMENT_DIGEST,
    CANONICAL_FEATURE_PROFILE_FINGERPRINT,
    fit_phase_direct_encoder,
    input_contract_for,
)
from app.training.encoding_storage import (  # noqa: E402
    write_bank_artifact,
    write_encoder_artifact,
)
from app.training.models import (  # noqa: E402
    DatasetManifest,
    DatasetPartition,
    LegacyDatasetManifest,
    TestAccessLedgerEntry,
)
from app.training.storage import (  # noqa: E402
    LEDGER_NAME,
    MAX_JSON_BYTES,
    ObservablePartition,
    artifact_root,
    load_labels,
    load_observations,
    verify_archive_opaque,
)

T = TypeVar("T")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sealed_snapshot(dataset_path: Path) -> tuple[DatasetManifest, str]:
    if os.environ.get("AQSE_ALLOW_TEST_OPEN") == "1":
        raise PermissionError("AQSE_ALLOW_TEST_OPEN must remain disabled during 1D.2")
    manifest = verify_archive_opaque(dataset_path)
    if isinstance(manifest, LegacyDatasetManifest):
        raise ValueError("Milestone 1D.2 requires the canonical archive-v2 dataset")
    if (
        manifest.dataset_id != CANONICAL_DEVELOPMENT_DATASET_ID
        or manifest.scientific_digest != CANONICAL_DEVELOPMENT_DIGEST
        or manifest.feature_profile_fingerprint
        != CANONICAL_FEATURE_PROFILE_FINGERPRINT
    ):
        raise ValueError("dataset path does not identify the approved canonical development v2")
    if manifest.test_state != "sealed":
        raise ValueError("canonical TEST state is not sealed")
    ledger_path = dataset_path / LEDGER_NAME
    if (
        ledger_path.is_symlink()
        or not ledger_path.is_file()
        or ledger_path.stat().st_size > MAX_JSON_BYTES
    ):
        raise ValueError("canonical TEST ledger is missing, unsafe or oversized")
    ledger_bytes = ledger_path.read_bytes()
    lines = [line for line in ledger_bytes.splitlines() if line]
    if len(lines) != 1:
        raise ValueError("canonical TEST ledger must contain exactly one genesis event")
    entry = TestAccessLedgerEntry.model_validate_json(lines[0])
    expected_digest = _sha256_bytes(
        canonical_json_bytes(entry.model_dump(mode="json", exclude={"entry_sha256"}))
    )
    if (
        entry.sequence != 0
        or entry.event != "sealed"
        or entry.dataset_id != manifest.dataset_id
        or entry.scientific_digest != manifest.scientific_digest
        or entry.previous_entry_sha256 is not None
        or entry.fixture_only
        or entry.entry_sha256 != expected_digest
    ):
        raise ValueError("canonical TEST ledger genesis event is invalid")
    return manifest, _sha256_bytes(ledger_bytes)


def _sealed_operation(
    dataset_path: Path,
    initial_snapshot: tuple[DatasetManifest, str],
    operation: Callable[[], T],
) -> T:
    before = _sealed_snapshot(dataset_path)
    if before != initial_snapshot:
        raise RuntimeError("canonical dataset or TEST ledger changed before semantic access")
    result = operation()
    after = _sealed_snapshot(dataset_path)
    if after != before:
        raise RuntimeError("canonical dataset or TEST ledger changed during semantic access")
    return result


def _bank_matrix(
    observations: ObservablePartition,
    rows: tuple[Any, ...],
) -> np.ndarray:
    episode_index = {
        episode_id: index for index, episode_id in enumerate(observations.episode_ids)
    }
    return np.stack(
        [
            observations.features[episode_index[row.episode_id], row.window_ordinal]
            for row in rows
        ]
    )


def _numerical_diagnostics(encoder, raw_reference: np.ndarray) -> dict[str, float]:
    context = input_contract_for(encoder.artifact)
    base = np.asarray(raw_reference, dtype=np.float64).copy()
    periodic = np.stack([base, base])
    periodic[0, 1] = 0.37
    periodic[1, 1] = 0.37 + 2.0 * pi
    epsilon = 1.0e-8
    seam = np.stack([base, base])
    seam[0, 1] = -pi + epsilon
    seam[1, 1] = pi - epsilon
    encoded_periodic = encoder.transform(periodic, context=context)
    encoded_seam = encoder.transform(seam, context=context)
    diagnostic_inputs = np.vstack([encoded_periodic, encoded_seam])
    theta = np.linspace(-0.45, 0.55, 16, dtype=np.float64)

    numpy_engine = StateEngine("numpy")
    qiskit_engine = StateEngine("qiskit")
    numpy_periodic = numpy_engine.states(encoded_periodic, theta)
    qiskit_periodic = qiskit_engine.states(encoded_periodic, theta)
    numpy_density = [np.outer(state, state.conj()) for state in numpy_periodic]
    qiskit_density = [np.outer(state, state.conj()) for state in qiskit_periodic]
    numpy_kernel = numpy_engine.gram(diagnostic_inputs, theta)
    qiskit_kernel = qiskit_engine.gram(diagnostic_inputs, theta)
    return {
        "density_periodicity_max_abs_numpy": float(
            np.max(np.abs(numpy_density[0] - numpy_density[1]))
        ),
        "density_periodicity_max_abs_qiskit": float(
            np.max(np.abs(qiskit_density[0] - qiskit_density[1]))
        ),
        "seam_fidelity_numpy": float(numpy_kernel[2, 3]),
        "seam_fidelity_qiskit": float(qiskit_kernel[2, 3]),
        "kernel_cross_engine_max_abs": float(
            np.max(np.abs(numpy_kernel - qiskit_kernel))
        ),
        "declared_absolute_tolerance": 1.0e-12,
        "seam_epsilon_rad": epsilon,
    }


def _range_report(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "minimum": np.min(values, axis=0).tolist(),
        "maximum": np.max(values, axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build immutable AQSE 1D.2 phase-direct encoder and quantum-bank artifacts."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Explicit path to aqse-development-064acca20fc788c6.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Override AQSE_ARTIFACT_ROOT for encoder/bank publication.",
    )
    arguments = parser.parse_args()
    dataset_path = arguments.dataset_path.resolve()
    if dataset_path.name != CANONICAL_DEVELOPMENT_DATASET_ID:
        raise ValueError("--dataset-path must name the explicit canonical development dataset")
    output_root = (
        arguments.artifact_root.resolve() if arguments.artifact_root else artifact_root()
    )
    if output_root == dataset_path or dataset_path in output_root.parents:
        raise ValueError("encoder/bank artifact root cannot be inside the canonical dataset")

    initial_snapshot = _sealed_snapshot(dataset_path)
    manifest = initial_snapshot[0]
    train = _sealed_operation(
        dataset_path,
        initial_snapshot,
        lambda: load_observations(dataset_path, DatasetPartition.TRAIN),
    )
    train_labels = _sealed_operation(
        dataset_path,
        initial_snapshot,
        lambda: load_labels(dataset_path, DatasetPartition.TRAIN),
    )
    validation = _sealed_operation(
        dataset_path,
        initial_snapshot,
        lambda: load_observations(dataset_path, DatasetPartition.VALIDATION),
    )

    encoder = fit_phase_direct_encoder(train, manifest)
    context = input_contract_for(encoder.artifact)
    train_raw = np.asarray(train.features[train.valid_mask], dtype=np.float64)
    validation_raw = np.asarray(
        validation.features[validation.valid_mask], dtype=np.float64
    )
    train_encoded = encoder.transform(train_raw, context=context)
    validation_encoded = encoder.transform(validation_raw, context=context)
    if (
        len(train.episode_ids) != 72
        or len(train_raw) != 1_368
        or len(validation.episode_ids) != 24
        or len(validation_raw) != 456
    ):
        raise ValueError("canonical 1D.2 fitting/validation population counts are unexpected")

    train_bank = select_training_quantum_bank(
        train,
        train_labels,
        manifest,
        encoder_artifact_id=encoder.artifact.artifact_id,
    )
    validation_bank = select_validation_quantum_bank(
        validation,
        manifest,
        encoder_artifact_id=encoder.artifact.artifact_id,
    )
    train_bank_raw = _bank_matrix(train, train_bank.rows)
    diagnostics = _numerical_diagnostics(encoder, train_bank_raw[0])
    if (
        diagnostics["density_periodicity_max_abs_numpy"] > 1.0e-12
        or diagnostics["density_periodicity_max_abs_qiskit"] > 1.0e-12
        or 1.0 - diagnostics["seam_fidelity_numpy"] > 1.0e-12
        or 1.0 - diagnostics["seam_fidelity_qiskit"] > 1.0e-12
        or diagnostics["kernel_cross_engine_max_abs"] > 1.0e-12
    ):
        raise ValueError("fixed-theta exact-state encoding diagnostic exceeded tolerance")

    expected_targets = (
        output_root / "encoders" / encoder.artifact.artifact_id,
        output_root / "banks" / train_bank.artifact_id,
        output_root / "banks" / validation_bank.artifact_id,
    )
    existing = [str(path) for path in expected_targets if path.exists()]
    if existing:
        raise FileExistsError(f"immutable 1D.2 artifact already exists: {existing[0]}")
    encoder_path, execution = write_encoder_artifact(encoder, root=output_root)
    train_bank_path = write_bank_artifact(train_bank, root=output_root)
    validation_bank_path = write_bank_artifact(validation_bank, root=output_root)
    final_snapshot = _sealed_snapshot(dataset_path)
    if final_snapshot != initial_snapshot:
        raise RuntimeError("canonical dataset or TEST ledger changed during 1D.2 publication")

    report = {
        "milestone": "AQSE 1D.2",
        "scientific_scope": "encoding and bank freeze only; no training or evaluation",
        "canonical_dataset": {
            "dataset_id": manifest.dataset_id,
            "scientific_digest": manifest.scientific_digest,
            "feature_profile_fingerprint": manifest.feature_profile_fingerprint,
        },
        "test_seal": {
            "semantic_access_count": 0,
            "state": manifest.test_state,
            "ledger_event_count": 1,
            "ledger_genesis": "sequence=0,event=sealed",
            "ledger_sha256": final_snapshot[1],
        },
        "encoder": {
            "path": str(encoder_path),
            "artifact_id": encoder.artifact.artifact_id,
            "content_digest": encoder.artifact.content_digest,
            "policy_id": encoder.artifact.encoding_policy_id,
            "scaler_id": encoder.artifact.scaler_id,
            "fitting_population": encoder.artifact.fitting_population.model_dump(mode="json"),
            "fitted_mean": list(encoder.artifact.fitted_mean),
            "fitted_scale": list(encoder.artifact.fitted_scale),
            "phase_scaler_statistics_used": False,
            "repository_base_sha": execution.repository_base_sha,
            "repository_dirty": execution.repository_dirty,
        },
        "banks": {
            "train": {
                "path": str(train_bank_path),
                "artifact_id": train_bank.artifact_id,
                "content_digest": train_bank.content_digest,
                "rows": train_bank.row_count,
                "lineages": train_bank.lineage_count,
                "class_balance": train_bank.class_balance,
            },
            "validation": {
                "path": str(validation_bank_path),
                "artifact_id": validation_bank.artifact_id,
                "content_digest": validation_bank.content_digest,
                "rows": validation_bank.row_count,
                "lineages": validation_bank.lineage_count,
            },
        },
        "encoded_ranges": {
            "train": _range_report(train_encoded),
            "validation": _range_report(validation_encoded),
        },
        "fixed_theta_infrastructure_diagnostic": diagnostics,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
