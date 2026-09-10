from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event

from app.training.assembly import TrainingInput, assemble_training_input
from app.training.canonical import array_identity
from app.training.run_storage import write_training_run_artifact
from app.training.runner import run_bounded_qng
from app.training.seal import verify_canonical_test_seal
from app.training.storage import artifact_root


@dataclass(frozen=True)
class TrainingJobOutcome:
    run_artifact_id: str
    accepted_update_count: int
    stop_reason: str


def canonical_paths(root: Path | None = None) -> tuple[Path, Path, Path, Path]:
    resolved = (root or artifact_root()).resolve()
    return (
        resolved / "aqse-development-064acca20fc788c6",
        resolved / "encoders" / "aqse-encoder-f9cf4bc12a767410",
        resolved / "banks" / "aqse-train-bank-32b5f93897676535",
        resolved,
    )


def validate_training_input_arrays(training_input: TrainingInput) -> None:
    identity = training_input.identity
    if array_identity(training_input.X_train) != identity.encoded_matrix_identity:
        raise ValueError("assembled TRAIN matrix differs from its frozen identity")
    if array_identity(training_input.y_train) != identity.ordered_label_identity:
        raise ValueError("assembled TRAIN labels differ from their frozen identity")
    if len(set(training_input.ordered_lineage_ids)) != 32:
        raise ValueError("assembled TRAIN input must contain 32 distinct lineages")


def execute_canonical_training(cancellation: Event) -> TrainingJobOutcome:
    dataset_path, encoder_path, bank_path, root = canonical_paths()
    seal_before = verify_canonical_test_seal(dataset_path)
    training_input = assemble_training_input(
        dataset_path=dataset_path,
        encoder_path=encoder_path,
        train_bank_path=bank_path,
    )
    validate_training_input_arrays(training_input)
    if verify_canonical_test_seal(dataset_path) != seal_before:
        raise RuntimeError("canonical TEST seal changed during training-input assembly")
    artifact = run_bounded_qng(training_input, cancellation=cancellation)
    path, _ = write_training_run_artifact(artifact, root=root)
    if path.name != artifact.run_id:
        raise RuntimeError("published training-run path has an incompatible identity")
    if verify_canonical_test_seal(dataset_path) != seal_before:
        raise RuntimeError("canonical TEST seal changed during QNG execution")
    return TrainingJobOutcome(
        run_artifact_id=artifact.run_id,
        accepted_update_count=artifact.accepted_update_count,
        stop_reason=artifact.stop_reason,
    )
