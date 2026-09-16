from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

BASELINE_NAMES = (
    "rbf_svc",
    "mlp_11_parameter",
    "rff_256",
    "gradient_boosting",
)


@dataclass(frozen=True)
class FittedComparator:
    name: str
    estimator: Any
    validation_balanced_accuracy: float
    validation_macro_f1: float
    configuration: dict[str, Any]

    def predict(self, X: ArrayLike) -> NDArray[np.int8]:
        """Predict labels; the result alone proves no scientific superiority."""

        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != 8 or not np.isfinite(matrix).all():
            raise ValueError("comparator input must be a finite Nx8 matrix")
        return np.asarray(self.estimator.predict(matrix), dtype=np.int8)


def _validate_fit_input(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_validation: ArrayLike,
    y_validation: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.int8], NDArray[np.float64], NDArray[np.int8]]:
    train = np.asarray(X_train, dtype=np.float64)
    labels = np.asarray(y_train, dtype=np.int8)
    validation = np.asarray(X_validation, dtype=np.float64)
    validation_labels = np.asarray(y_validation, dtype=np.int8)
    if train.ndim != 2 or train.shape[1] != 8 or validation.ndim != 2 or validation.shape[1] != 8:
        raise ValueError("classical comparators require Nx8 TRAIN and VALIDATION matrices")
    if labels.shape != (len(train),) or validation_labels.shape != (len(validation),):
        raise ValueError("classical comparator labels must align with features")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("classical comparator features contain NaN or infinity")
    if set(labels.tolist()) != {-1, 1} or set(validation_labels.tolist()) != {-1, 1}:
        raise ValueError("classical comparators require both -1 and +1 labels")
    return train, labels, validation, validation_labels


def _score(
    name: str,
    estimator: Any,
    validation: NDArray[np.float64],
    labels: NDArray[np.int8],
    configuration: dict[str, Any],
) -> FittedComparator:
    predicted = np.asarray(estimator.predict(validation), dtype=np.int8)
    return FittedComparator(
        name=name,
        estimator=estimator,
        validation_balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
        validation_macro_f1=float(
            f1_score(labels, predicted, labels=[-1, 1], average="macro", zero_division=0.0)
        ),
        configuration=configuration,
    )


def fit_classical_comparators(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_validation: ArrayLike,
    y_validation: ArrayLike,
    *,
    seed: int,
) -> dict[str, FittedComparator]:
    """Fit frozen classical comparators on the same TRAIN/VALIDATION split.

    The one-hidden-unit MLP has ``8*1 + 1 + 1*1 + 1 = 11`` trainable weights
    and biases.  Eleven is the closest under-capacity integer-width MLP to the
    VQC's 16 theta parameters (two hidden units would have 21).  RFF uses 256
    features, matching the dimension of an eight-qubit Hilbert space, but this
    dimensional match is only a capacity reference and not an equivalence.
    Fitting these baselines does not establish fairness beyond the declared
    split and fixed configurations.
    """

    train, labels, validation, validation_labels = _validate_fit_input(
        X_train, y_train, X_validation, y_validation
    )
    fitted: dict[str, FittedComparator] = {}

    rbf_candidates: list[FittedComparator] = []
    for c_value in (0.1, 1.0, 10.0):
        for gamma in (0.01, 0.1, 1.0):
            estimator = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", SVC(kernel="rbf", C=c_value, gamma=gamma)),
                ]
            ).fit(train, labels)
            rbf_candidates.append(
                _score(
                    "rbf_svc",
                    estimator,
                    validation,
                    validation_labels,
                    {"C": c_value, "gamma": gamma, "selection": "VALIDATION"},
                )
            )
    fitted["rbf_svc"] = min(
        rbf_candidates,
        key=lambda item: (
            -item.validation_balanced_accuracy,
            -item.validation_macro_f1,
            float(item.configuration["C"]),
            float(item.configuration["gamma"]),
        ),
    )

    mlp = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(1,),
                    activation="tanh",
                    solver="lbfgs",
                    alpha=1.0e-3,
                    max_iter=500,
                    random_state=seed,
                ),
            ),
        ]
    ).fit(train, labels)
    fitted["mlp_11_parameter"] = _score(
        "mlp_11_parameter",
        mlp,
        validation,
        validation_labels,
        {
            "hidden_layer_sizes": [1],
            "trainable_parameter_count": 11,
            "solver": "lbfgs",
            "alpha": 1.0e-3,
        },
    )

    rff = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "rff",
                RBFSampler(gamma=0.125, n_components=256, random_state=seed),
            ),
            ("model", LinearSVC(C=1.0, random_state=seed, max_iter=10_000)),
        ]
    ).fit(train, labels)
    fitted["rff_256"] = _score(
        "rff_256",
        rff,
        validation,
        validation_labels,
        {
            "gamma": 0.125,
            "n_components": 256,
            "linear_svc_C": 1.0,
            "capacity_reference": "2**8 Hilbert-space dimension",
        },
    )

    boosting = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=seed,
    ).fit(train, labels)
    fitted["gradient_boosting"] = _score(
        "gradient_boosting",
        boosting,
        validation,
        validation_labels,
        {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
        },
    )
    if tuple(fitted) != BASELINE_NAMES:
        raise RuntimeError("classical comparator set is incomplete or misordered")
    return fitted
