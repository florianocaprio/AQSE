from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from app.quantum.user_pipeline.tqk8 import N_PARAMS, StateEngine
from app.training.generation import build_episode_plans, generate_episode
from app.training.models import DatasetPartition
from app.training.splits import stratified_lineage_split

CENTRAL_WINDOW_ORDINAL = 9


def _readonly(value: ArrayLike, *, dtype: Any) -> NDArray[Any]:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReplicatedDataset:
    """One independent TRAIN/VALIDATION/TEST split.

    The container preserves split isolation but does not itself demonstrate
    that a generator represents field sensor behavior.
    """

    X_train: NDArray[np.float64]
    y_train: NDArray[np.int8]
    X_validation: NDArray[np.float64]
    y_validation: NDArray[np.int8]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.int8]
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    metadata: dict[str, Any]

    def with_train_labels(self, labels: ArrayLike) -> ReplicatedDataset:
        """Return a control copy; changing TRAIN labels proves no real-data claim."""

        values = _readonly(labels, dtype=np.int8)
        if values.shape != self.y_train.shape:
            raise ValueError("replacement TRAIN labels have an invalid shape")
        if set(values.tolist()) != {-1, 1}:
            raise ValueError("replacement TRAIN labels must contain -1 and +1")
        return replace(self, y_train=values)


def validate_replicated_dataset(dataset: ReplicatedDataset) -> None:
    """Validate mechanics and isolation, not scientific representativeness."""

    partitions = (
        ("TRAIN", dataset.X_train, dataset.y_train, dataset.train_ids),
        (
            "VALIDATION",
            dataset.X_validation,
            dataset.y_validation,
            dataset.validation_ids,
        ),
        ("TEST", dataset.X_test, dataset.y_test, dataset.test_ids),
    )
    all_ids: list[str] = []
    for name, X, y, identities in partitions:
        matrix = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y, dtype=np.int8)
        if matrix.ndim != 2 or matrix.shape[1] != 8:
            raise ValueError(f"{name} features must have shape (N, 8)")
        if labels.shape != (len(matrix),) or len(identities) != len(matrix):
            raise ValueError(f"{name} features, labels and identities must align")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} features contain NaN or infinity")
        if set(labels.tolist()) != {-1, 1}:
            raise ValueError(f"{name} must contain both -1 and +1 labels")
        counts = [int(np.count_nonzero(labels == target)) for target in (-1, 1)]
        minimum = 4 if name == "TRAIN" else 2
        if min(counts) < minimum:
            raise ValueError(f"{name} has too few independent examples per class")
        if len(set(identities)) != len(identities):
            raise ValueError(f"{name} identities must be distinct")
        all_ids.extend(identities)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("sample identities cross partitions")


def _row_key(row: NDArray[np.float64], label: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(row, dtype="<f8").tobytes(order="C"))
    digest.update(int(label).to_bytes(1, "little", signed=True))
    return digest.hexdigest()


def split_binary_dataset(
    X: ArrayLike,
    y: ArrayLike,
    *,
    split_seed: int,
) -> ReplicatedDataset:
    """Create a deterministic 60/20/20 split for controls and fixtures.

    Samples are canonicalized before seeded assignment, so reordering distinct
    input rows does not change partition content.  This convenience splitter
    does not establish lineage independence for an external dataset.
    """

    matrix = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[1] != 8 or labels.shape != (len(matrix),):
        raise ValueError("binary study data must contain aligned Nx8 features and labels")
    if not np.isfinite(matrix).all() or set(labels.tolist()) != {-1, 1}:
        raise ValueError("binary study data require finite features and labels -1/+1")
    if len(matrix) < 20:
        raise ValueError("binary study data require at least 20 samples")

    keys = np.asarray(
        [_row_key(row, int(label)) for row, label in zip(matrix, labels, strict=True)]
    )
    if len(set(keys.tolist())) != len(keys):
        raise ValueError("binary study data contain duplicate feature/label rows")
    assignments: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for class_index, target in enumerate((-1, 1)):
        candidates = np.flatnonzero(labels == target)
        if len(candidates) < 10:
            raise ValueError("each class requires at least 10 samples")
        candidates = candidates[np.argsort(keys[candidates])]
        rng = np.random.default_rng(np.random.SeedSequence([split_seed, class_index]))
        ordered = candidates[rng.permutation(len(candidates))]
        validation_count = max(2, int(round(0.2 * len(ordered))))
        test_count = max(2, int(round(0.2 * len(ordered))))
        train_count = len(ordered) - validation_count - test_count
        if train_count < 4:
            raise ValueError("split leaves fewer than four TRAIN rows per class")
        assignments["train"].extend(int(item) for item in ordered[:train_count])
        assignments["validation"].extend(
            int(item) for item in ordered[train_count : train_count + validation_count]
        )
        assignments["test"].extend(int(item) for item in ordered[train_count + validation_count :])

    def partition(name: str) -> tuple[NDArray[np.float64], NDArray[np.int8], tuple[str, ...]]:
        indices = np.asarray(sorted(assignments[name], key=lambda item: keys[item]), dtype=np.int64)
        return (
            _readonly(matrix[indices], dtype=np.float64),
            _readonly(labels[indices], dtype=np.int8),
            tuple(f"sample-{keys[index][:24]}" for index in indices),
        )

    X_train, y_train, train_ids = partition("train")
    X_validation, y_validation, validation_ids = partition("validation")
    X_test, y_test, test_ids = partition("test")
    result = ReplicatedDataset(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        metadata={
            "generator": "caller-supplied-binary-matrix",
            "split_policy": "canonicalized-stratified-60-20-20",
            "split_seed": split_seed,
        },
    )
    validate_replicated_dataset(result)
    return result


def sensor_replica_dataset(dataset_seed: int, split_seed: int) -> ReplicatedDataset:
    """Generate one full AQSE simulator replica with the existing feature path.

    The function reuses the approved white-noise-regime simulator and central
    feature window with a new seed namespace.  It does not reuse or open the
    historical frozen TEST and does not establish field representativeness.
    """

    specifications = build_episode_plans(
        master_seed=dataset_seed,
        episodes_per_class=60,
        purpose="paper-replicated-study-v1",
    )
    assignments = stratified_lineage_split(
        tuple(item.label for item in specifications),
        split_seed=split_seed,
    )
    partition_by_episode = {item.episode_id: item.partition for item in assignments}
    rows: dict[DatasetPartition, list[tuple[str, NDArray[np.float64], int]]] = {
        DatasetPartition.TRAIN: [],
        DatasetPartition.VALIDATION: [],
        DatasetPartition.TEST: [],
    }
    for specification in specifications:
        episode = generate_episode(specification)
        if len(episode.features) <= CENTRAL_WINDOW_ORDINAL:
            raise RuntimeError("generated episode lacks the frozen central feature window")
        if not bool(episode.valid_mask[CENTRAL_WINDOW_ORDINAL]):
            raise RuntimeError("generated central feature window is not quantum eligible")
        partition = partition_by_episode[episode.plan.episode_id]
        rows[partition].append(
            (
                episode.plan.episode_id,
                np.asarray(episode.features[CENTRAL_WINDOW_ORDINAL], dtype=np.float64),
                episode.label.target,
            )
        )

    def convert(partition: DatasetPartition) -> tuple[NDArray[np.float64], NDArray[np.int8], tuple[str, ...]]:
        ordered = sorted(rows[partition], key=lambda item: item[0])
        return (
            _readonly([item[1] for item in ordered], dtype=np.float64),
            _readonly([item[2] for item in ordered], dtype=np.int8),
            tuple(item[0] for item in ordered),
        )

    X_train, y_train, train_ids = convert(DatasetPartition.TRAIN)
    X_validation, y_validation, validation_ids = convert(DatasetPartition.VALIDATION)
    X_test, y_test, test_ids = convert(DatasetPartition.TEST)
    result = ReplicatedDataset(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        metadata={
            "generator": "aqse.milestone-1d1.white-noise.v1",
            "purpose": "paper-replicated-study-v1",
            "dataset_seed": dataset_seed,
            "split_seed": split_seed,
            "central_window_ordinal": CENTRAL_WINDOW_ORDINAL,
        },
    )
    validate_replicated_dataset(result)
    return result


def positive_control_dataset(n: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Build a TQK-teacher nonlinear positive-control dataset.

    Labels are the sign of the fidelity contrast to two fixed TQK states.  This
    intentionally mechanism-aligned construction is only a software sanity
    control: success cannot be transferred to sensor data and cannot support a
    quantum-advantage claim.  It avoids asserting that an arbitrary XOR sample
    must favor this particular protected circuit.
    """

    if n < 40 or n % 2:
        raise ValueError("positive control requires an even n >= 40")
    rng = np.random.default_rng(seed)
    theta = np.random.default_rng(np.random.SeedSequence([seed, 17])).uniform(
        -0.8, 0.8, N_PARAMS
    )
    engine = StateEngine("numpy")
    prototypes = np.asarray(
        [
            (-1.15, 0.75, -0.85, 1.05, -0.65, 0.95, -1.25, 0.55),
            (1.10, -0.80, 0.90, -1.00, 0.70, -0.90, 1.20, -0.60),
        ],
        dtype=np.float64,
    )
    selected: dict[int, list[tuple[float, NDArray[np.float64]]]] = {-1: [], 1: []}
    while min(len(selected[-1]), len(selected[1])) < n // 2:
        candidates = rng.uniform(-1.4, 1.4, size=(max(64, n), 8))
        prototype_kernel = engine.gram(candidates, theta, prototypes)
        contrast = prototype_kernel[:, 1] - prototype_kernel[:, 0]
        for row, score in zip(candidates, contrast, strict=True):
            label = 1 if score >= 0.0 else -1
            selected[label].append((abs(float(score)), row.copy()))
    rows: list[NDArray[np.float64]] = []
    labels: list[int] = []
    for label in (-1, 1):
        strongest = sorted(selected[label], key=lambda item: item[0], reverse=True)[: n // 2]
        rows.extend(item[1] for item in strongest)
        labels.extend([label] * len(strongest))
    order = rng.permutation(n)
    X = _readonly(np.asarray(rows)[order], dtype=np.float64)
    y = _readonly(np.asarray(labels)[order], dtype=np.int8)
    return X, y
