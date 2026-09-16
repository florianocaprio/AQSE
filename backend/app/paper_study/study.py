from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.svm import SVC

from app.quantum.user_pipeline.tqk8 import (
    N_PARAMS,
    AngleScaler,
    StateEngine,
    fit_qng,
)
from app.training.encoding import wrap_phase_direct

from .baselines import FittedComparator, fit_classical_comparators
from .datasets import (
    ReplicatedDataset,
    sensor_replica_dataset,
    split_binary_dataset,
    validate_replicated_dataset,
)
from .diagnostics import kernel_diagnostics
from .models import (
    PermutationControlResult,
    QuantumSelection,
    ReplicaResult,
    ReplicaSeeds,
    ReplicatedStudyResult,
    SplitSizes,
)
from .statistics import aggregate_replicas, bootstrap_mean_interval
from .storage import ReplicaStore

DatasetFactory = Callable[[int, int], ReplicatedDataset]


@dataclass(frozen=True)
class _QuantumCandidate:
    checkpoint_index: int
    theta: NDArray[np.float64]
    svc_c: float
    model: SVC
    validation_balanced_accuracy: float
    validation_macro_f1: float
    diagnostics_index: int


@dataclass(frozen=True)
class _FittedQuantum:
    candidate: _QuantumCandidate
    engine: StateEngine
    scaler: AngleScaler
    X_train_encoded: NDArray[np.float64]
    diagnostics: tuple[Any, ...]
    accepted_qng_updates: int

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int8]:
        encoded = _encode_with_scaler(X, self.scaler)
        kernel = self.engine.gram(
            encoded,
            self.candidate.theta,
            self.X_train_encoded,
        )
        return np.asarray(self.candidate.model.predict(kernel), dtype=np.int8)


def _derive_seed(base_seed: int, replica_index: int, domain: str) -> int:
    payload = f"aqse-paper-v1:{base_seed}:{replica_index}:{domain}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _replica_seeds(base_seed: int, replica_index: int) -> ReplicaSeeds:
    return ReplicaSeeds(
        replica=_derive_seed(base_seed, replica_index, "replica"),
        dataset=_derive_seed(base_seed, replica_index, "dataset"),
        split=_derive_seed(base_seed, replica_index, "split"),
        qng=_derive_seed(base_seed, replica_index, "qng"),
        classical=_derive_seed(base_seed, replica_index, "classical"),
        permutation=_derive_seed(base_seed, replica_index, "permutation-control"),
    )


def _default_output_root() -> Path:
    return Path(os.environ.get("AQSE_ARTIFACT_ROOT", "/artifacts")) / (
        "paper-replicated-study-v1"
    )


def _encode_with_scaler(
    X: NDArray[np.float64], scaler: AngleScaler
) -> NDArray[np.float64]:
    raw = np.asarray(X, dtype=np.float64)
    encoded = np.asarray(scaler.transform(raw), dtype=np.float64)
    encoded[:, 1] = wrap_phase_direct(raw[:, 1])
    return encoded


def _balanced_bank_indices(y: NDArray[np.int8], *, seed: int) -> NDArray[np.int64]:
    selected: list[int] = []
    per_class = min(16, *(int(np.count_nonzero(y == target)) for target in (-1, 1)))
    if per_class < 4:
        raise ValueError("QNG requires at least four TRAIN rows per class")
    for class_index, target in enumerate((-1, 1)):
        candidates = np.flatnonzero(y == target)
        rng = np.random.default_rng(np.random.SeedSequence([seed, class_index]))
        selected.extend(int(item) for item in rng.permutation(candidates)[:per_class])
    return np.asarray(selected, dtype=np.int64)


def _fit_quantum(
    dataset: ReplicatedDataset,
    *,
    qng_seed: int,
    qng_steps: int,
) -> _FittedQuantum:
    scaler = AngleScaler.fit(dataset.X_train)
    X_train = _encode_with_scaler(dataset.X_train, scaler)
    X_validation = _encode_with_scaler(dataset.X_validation, scaler)
    engine = StateEngine("numpy")
    bank_indices = _balanced_bank_indices(dataset.y_train, seed=qng_seed)
    theta = np.random.default_rng(qng_seed).uniform(-0.8, 0.8, N_PARAMS)
    checkpoints = [theta.copy()]
    for _ in range(qng_steps):
        candidate, history = fit_qng(
            engine,
            X_train[bank_indices],
            dataset.y_train[bank_indices],
            theta,
            steps=1,
            learning_rate=0.2,
            damping=1.0e-3,
            max_step=0.4,
            verbose=False,
        )
        if not history:
            break
        theta = np.asarray(candidate, dtype=np.float64)
        checkpoints.append(theta.copy())

    candidates: list[_QuantumCandidate] = []
    diagnostics: list[Any] = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        K_train = np.asarray(engine.gram(X_train, checkpoint), dtype=np.float64)
        K_validation = np.asarray(
            engine.gram(X_validation, checkpoint, X_train), dtype=np.float64
        )
        diagnostics.append(kernel_diagnostics(K_train))
        for c_value in (0.1, 1.0, 10.0):
            model = SVC(kernel="precomputed", C=c_value).fit(K_train, dataset.y_train)
            predictions = np.asarray(model.predict(K_validation), dtype=np.int8)
            candidates.append(
                _QuantumCandidate(
                    checkpoint_index=checkpoint_index,
                    theta=checkpoint.copy(),
                    svc_c=c_value,
                    model=model,
                    validation_balanced_accuracy=float(
                        balanced_accuracy_score(dataset.y_validation, predictions)
                    ),
                    validation_macro_f1=float(
                        f1_score(
                            dataset.y_validation,
                            predictions,
                            labels=[-1, 1],
                            average="macro",
                            zero_division=0.0,
                        )
                    ),
                    diagnostics_index=checkpoint_index,
                )
            )
    selected = min(
        candidates,
        key=lambda item: (
            -item.validation_balanced_accuracy,
            -item.validation_macro_f1,
            item.checkpoint_index,
            item.svc_c,
        ),
    )
    return _FittedQuantum(
        candidate=selected,
        engine=engine,
        scaler=scaler,
        X_train_encoded=X_train,
        diagnostics=tuple(diagnostics),
        accepted_qng_updates=len(checkpoints) - 1,
    )


def _selection_payload(
    quantum: _FittedQuantum,
    baselines: dict[str, FittedComparator],
) -> dict[str, Any]:
    selected = quantum.candidate
    return {
        "schema_version": "aqse.paper-model-selection-freeze.v1",
        "selection_role": "VALIDATION-only",
        "quantum": {
            "checkpoint_index": selected.checkpoint_index,
            "theta": selected.theta.tolist(),
            "svc_c": selected.svc_c,
            "validation_balanced_accuracy": selected.validation_balanced_accuracy,
            "validation_macro_f1": selected.validation_macro_f1,
        },
        "baselines": {
            name: {
                "configuration": item.configuration,
                "validation_balanced_accuracy": item.validation_balanced_accuracy,
                "validation_macro_f1": item.validation_macro_f1,
            }
            for name, item in baselines.items()
        },
        "scientific_scope": (
            "selection freeze before single TEST access; no claim of quantum advantage"
        ),
    }


def _run_replica(
    *,
    replica_index: int,
    seeds: ReplicaSeeds,
    dataset: ReplicatedDataset,
    output_root: Path,
    qng_steps: int,
    permute_train_labels: bool,
) -> ReplicaResult:
    validate_replicated_dataset(dataset)
    if permute_train_labels:
        rng = np.random.default_rng(seeds.permutation)
        dataset = dataset.with_train_labels(rng.permutation(dataset.y_train))

    store = ReplicaStore(output_root / f"replica-{replica_index:03d}")
    protocol = {
        "schema_version": "aqse.paper-replica-protocol.v1",
        "replica_index": replica_index,
        "seeds": seeds.model_dump(mode="json"),
        "split_sizes": {
            "train": len(dataset.X_train),
            "validation": len(dataset.X_validation),
            "test": len(dataset.X_test),
        },
        "qng_steps": qng_steps,
        "training_label_permutation_control": permute_train_labels,
        "test_access_policy": "sealed then one observations/labels/evaluation sequence",
    }
    store.publish(dataset, protocol)

    quantum = _fit_quantum(dataset, qng_seed=seeds.qng, qng_steps=qng_steps)
    baselines = fit_classical_comparators(
        dataset.X_train,
        dataset.y_train,
        dataset.X_validation,
        dataset.y_validation,
        seed=seeds.classical,
    )
    store.freeze_selection(_selection_payload(quantum, baselines))

    X_test = store.open_test_observations()
    quantum_predictions = quantum.predict(X_test)
    baseline_predictions = {
        name: comparator.predict(X_test) for name, comparator in baselines.items()
    }
    y_test = store.open_test_labels()
    quantum_score = float(balanced_accuracy_score(y_test, quantum_predictions))
    baseline_scores = {
        name: float(balanced_accuracy_score(y_test, predictions))
        for name, predictions in baseline_predictions.items()
    }
    selected_diagnostics = quantum.diagnostics[
        quantum.candidate.diagnostics_index
    ]
    result = ReplicaResult(
        replica_index=replica_index,
        seeds=seeds,
        split_sizes=SplitSizes(
            train=len(dataset.X_train),
            validation=len(dataset.X_validation),
            test=len(dataset.X_test),
        ),
        dataset_metadata=dataset.metadata,
        quantum_balanced_accuracy=quantum_score,
        baseline_balanced_accuracy=baseline_scores,
        delta_by_baseline={
            name: quantum_score - score for name, score in baseline_scores.items()
        },
        quantum_selection=QuantumSelection(
            checkpoint_index=quantum.candidate.checkpoint_index,
            accepted_qng_updates=quantum.accepted_qng_updates,
            svc_c=quantum.candidate.svc_c,
            theta=tuple(float(value) for value in quantum.candidate.theta),
            validation_balanced_accuracy=quantum.candidate.validation_balanced_accuracy,
            validation_macro_f1=quantum.candidate.validation_macro_f1,
        ),
        kernel_diagnostics_by_checkpoint=quantum.diagnostics,
        selected_kernel_diagnostics=selected_diagnostics,
        test_ledger_path=str(store.ledger_path),
    )
    store.publish_evaluation(result.model_dump(mode="json"))
    store.verify_closed()
    return result


def _run_replicas(
    *,
    n_replicas: int,
    base_seed: int,
    output_dir: Path,
    dataset_factory: DatasetFactory,
    qng_steps: int,
    bootstrap_resamples: int,
    permute_train_labels: bool,
) -> ReplicatedStudyResult:
    if n_replicas < 2:
        raise ValueError("replicated study requires at least two replicas")
    if not 0 <= base_seed <= 2**32 - 1:
        raise ValueError("base_seed must fit uint32")
    if qng_steps < 0 or qng_steps > 10:
        raise ValueError("qng_steps must remain within the frozen 0..10 budget")
    if output_dir.exists():
        raise FileExistsError(f"study output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    replicas: list[ReplicaResult] = []
    for replica_index in range(n_replicas):
        seeds = _replica_seeds(base_seed, replica_index)
        dataset = dataset_factory(seeds.dataset, seeds.split)
        replicas.append(
            _run_replica(
                replica_index=replica_index,
                seeds=seeds,
                dataset=dataset,
                output_root=output_dir,
                qng_steps=qng_steps,
                permute_train_labels=permute_train_labels,
            )
        )
    aggregate = aggregate_replicas(
        replicas,
        bootstrap_seed=_derive_seed(base_seed, 0, "aggregate-bootstrap"),
        bootstrap_resamples=bootstrap_resamples,
    )
    result = ReplicatedStudyResult(
        base_seed=base_seed,
        replica_count=n_replicas,
        qng_steps=qng_steps,
        replicas=tuple(replicas),
        aggregate=aggregate,
    )
    aggregate_path = output_dir / "aggregate.json"
    aggregate_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_replicated_study(
    n_replicas: int = 30,
    base_seed: int = 3_001_001,
    *,
    output_dir: Path | None = None,
    dataset_factory: DatasetFactory = sensor_replica_dataset,
    qng_steps: int = 10,
    bootstrap_resamples: int = 2_000,
) -> ReplicatedStudyResult:
    """Run isolated simulator replicas with one TEST access per replica.

    Every replica derives independent dataset, split, optimizer and classical
    seeds; selects theta and hyperparameters using VALIDATION; then opens its
    own newly generated TEST once.  The function never reads the historical
    AQSE TEST.  Even 30 successful replicas quantify only this simulator and do
    not prove field validity or quantum advantage.
    """

    root = output_dir or (_default_output_root() / f"study-{base_seed}")
    return _run_replicas(
        n_replicas=n_replicas,
        base_seed=base_seed,
        output_dir=root,
        dataset_factory=dataset_factory,
        qng_steps=qng_steps,
        bootstrap_resamples=bootstrap_resamples,
        permute_train_labels=False,
    )


def permutation_control(
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    *,
    n_replicas: int = 30,
    base_seed: int = 3_002_001,
    output_dir: Path | None = None,
    qng_steps: int = 10,
    bootstrap_resamples: int = 2_000,
) -> PermutationControlResult:
    """Shuffle TRAIN labels independently and repeat the sealed pipeline.

    Compatibility with 0.5 is a negative-control check, not evidence that the
    unpermuted model is correct.  VALIDATION and TEST labels remain untouched;
    TEST access remains single-use inside every control replica.
    """

    matrix = np.asarray(X, dtype=np.float64).copy()
    labels = np.asarray(y, dtype=np.int8).copy()

    def factory(_: int, split_seed: int) -> ReplicatedDataset:
        return split_binary_dataset(matrix, labels, split_seed=split_seed)

    root = output_dir or (_default_output_root() / f"permutation-{base_seed}")
    study = _run_replicas(
        n_replicas=n_replicas,
        base_seed=base_seed,
        output_dir=root,
        dataset_factory=factory,
        qng_steps=qng_steps,
        bootstrap_resamples=bootstrap_resamples,
        permute_train_labels=True,
    )
    quantum_values = [item.quantum_balanced_accuracy for item in study.replicas]
    estimates = {
        "quantum": bootstrap_mean_interval(
            quantum_values,
            seed=_derive_seed(base_seed, 0, "permutation-quantum-bootstrap"),
            resamples=bootstrap_resamples,
        )
    }
    for index, name in enumerate(sorted(study.aggregate.baseline_balanced_accuracy)):
        estimates[name] = bootstrap_mean_interval(
            [item.baseline_balanced_accuracy[name] for item in study.replicas],
            seed=_derive_seed(base_seed, index, f"permutation-{name}-bootstrap"),
            resamples=bootstrap_resamples,
        )
    compatible = {
        name: interval.lower <= 0.5 <= interval.upper
        for name, interval in estimates.items()
    }
    return PermutationControlResult(
        replica_count=n_replicas,
        balanced_accuracy=estimates,
        compatible_with_chance=compatible,
        study=study,
    )
