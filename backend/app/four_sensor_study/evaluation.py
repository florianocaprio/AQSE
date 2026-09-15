from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.demo.bundle_models import (
    ABSTAIN_CLASS,
    NETWORK_TASK_ID,
    BundleQueryContext,
    BundleRuntime,
)
from app.demo.bundle_storage import load_active_bundle_set, load_bundle_by_id
from app.training.canonical import canonical_json_bytes

from .models import FourSensorStudyLabel, FourSensorStudyObservation
from .protocol import SCENARIOS

BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 3_001_003
PRIMARY_BALANCED_ACCURACY_THRESHOLD = 0.75
PRIMARY_MACRO_F1_THRESHOLD = 0.75
MINIMUM_CLASS_RECALL = 0.70
MINIMUM_NORMAL_RECALL = 0.80
MINIMUM_COVERAGE = 0.90
MAXIMUM_NORMAL_FALSE_POSITIVE_RATE = 0.20
MINIMUM_ADVANTAGE_DELTA = 0.03

_OBSERVABLE_CLASS = {
    "NO_OBSERVED_CHANGE": "NORMAL",
    "COMMON_CHANGE_AMBIGUOUS": "ENVIRONMENT_COMPATIBLE",
    "SPATIAL_DISAGREEMENT_AMBIGUOUS": "DEVICE_COMPATIBLE",
    "MIXED_OBSERVABLE_CHANGE": "MIXED_OR_AMBIGUOUS",
}


def _digest(value: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def active_network_model_binding(*, root=None) -> dict[str, Any]:
    """Resolve the already-frozen network model without fitting or selection."""

    active = load_active_bundle_set(root=root)
    if active is None:
        raise RuntimeError("no active frozen AQSE bundle set is available")
    artifact = active.network.artifact
    if artifact.task_id != NETWORK_TASK_ID:
        raise RuntimeError("the active network bundle has an incompatible task")
    payload = {
        "schema_version": "aqse.four-sensor-study.model-binding.v1",
        "selection_freeze_id": active.pointer.selection_freeze_id,
        "application_id_at_freeze": active.pointer.application_id,
        "bundle_id": artifact.bundle_id,
        "bundle_content_digest": artifact.content_digest,
        "source_study_artifact_id": artifact.study_artifact_id,
        "source_study_content_digest": artifact.study_content_digest,
        "source_protocol_digest": artifact.protocol_digest,
        "task_id": artifact.task_id,
        "profile_id": artifact.profile_id,
        "profile_fingerprint": artifact.profile_fingerprint,
        "theta_candidate": artifact.theta_candidate.model_dump(mode="json"),
        "effective_source_hashes": dict(sorted(artifact.effective_source_hashes.items())),
        "fit_or_selection_performed": False,
    }
    return {**payload, "binding_digest": _digest(payload)}


def load_bound_network_runtime(
    binding: dict[str, Any],
    *,
    root=None,
) -> BundleRuntime:
    bundle = load_bundle_by_id(str(binding["bundle_id"]), root=root)
    if (
        bundle.content_digest != binding.get("bundle_content_digest")
        or bundle.task_id != NETWORK_TASK_ID
        or bundle.profile_id != binding.get("profile_id")
        or bundle.profile_fingerprint != binding.get("profile_fingerprint")
    ):
        raise RuntimeError("the frozen network bundle differs from the protocol binding")
    payload = {key: value for key, value in binding.items() if key != "binding_digest"}
    if binding.get("binding_digest") != _digest(payload):
        raise RuntimeError("the frozen network model binding digest is invalid")
    return BundleRuntime.from_artifact(bundle)


def assess_pilot(
    observations: Sequence[FourSensorStudyObservation],
    labels: Sequence[FourSensorStudyLabel],
) -> dict[str, Any]:
    """Quality-only pilot gate; it deliberately does not execute the model."""

    if len(observations) != 20 or len(labels) != 20:
        raise ValueError("the noncanonical pilot must contain exactly 20 episodes")
    label_by_episode = {item.episode_id: item for item in labels}
    if len(label_by_episode) != 20:
        raise ValueError("pilot supervisory episode identities must be unique")
    values: list[tuple[float, ...]] = []
    quality_failures: list[dict[str, Any]] = []
    for observation in observations:
        label = label_by_episode.get(observation.episode_id)
        if label is None or label.generative_lineage_id != observation.generative_lineage_id:
            raise ValueError("pilot observation/label channels are not aligned")
        feature = observation.network_feature
        complete = all(value is not None for value in feature.values)
        if not feature.quality.valid_for_quantum or not complete:
            quality_failures.append(
                {
                    "episode_id": observation.episode_id,
                    "quality_flags": list(feature.quality.flags),
                }
            )
        else:
            values.append(tuple(float(value) for value in feature.values))

    matrix = np.asarray(values, dtype=np.float64)
    finite = bool(matrix.shape == (20, 8) and np.isfinite(matrix).all())
    class_counts = Counter(item.scenario for item in labels)
    geometry_counts = Counter(item.geometry_id for item in labels)
    focal_counts = Counter(item.focal_sensor_id for item in labels)
    accepted = bool(
        not quality_failures
        and finite
        and set(class_counts.values()) == {5}
        and set(geometry_counts.values()) == {5}
        and set(focal_counts.values()) == {5}
    )
    report: dict[str, Any] = {
        "schema_version": "aqse.four-sensor-study.pilot-assessment.v1",
        "purpose": "structural-and-measurement-quality-only;not-model-selection",
        "episode_count": len(observations),
        "class_counts": dict(sorted(class_counts.items())),
        "geometry_counts": dict(sorted(geometry_counts.items())),
        "focal_sensor_counts": dict(sorted(focal_counts.items())),
        "valid_primary_feature_count": len(values),
        "quality_failures": quality_failures,
        "all_primary_features_finite": finite,
        "primary_feature_min": matrix.min(axis=0).tolist() if finite else [],
        "primary_feature_max": matrix.max(axis=0).tolist() if finite else [],
        "accepted": accepted,
        "pilot_rows_reusable_in_final_test": False,
        "model_executed": False,
    }
    return {**report, "assessment_digest": _digest(report)}


def _metric_pair(
    truth: Sequence[str],
    predicted: Sequence[str],
) -> tuple[float, float]:
    recalls: list[float] = []
    f1_values: list[float] = []
    for label in SCENARIOS:
        support = sum(value == label for value in truth)
        if support == 0:
            continue
        true_positive = sum(
            actual == label and result == label
            for actual, result in zip(truth, predicted, strict=True)
        )
        false_positive = sum(
            actual != label and result == label
            for actual, result in zip(truth, predicted, strict=True)
        )
        false_negative = support - true_positive
        recalls.append(true_positive / support)
        denominator = (2 * true_positive) + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(recalls)), float(np.mean(f1_values))


def _stratified_bootstrap_indices(
    truth: Sequence[str],
) -> tuple[NDArray[np.int64], ...]:
    truth_array = np.asarray(truth, dtype=str)
    by_class = [np.flatnonzero(truth_array == label) for label in SCENARIOS]
    if any(len(indices) == 0 for indices in by_class):
        raise ValueError("bootstrap requires support for every frozen class")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    return tuple(
        np.concatenate(
            [generator.choice(indices, size=len(indices), replace=True) for indices in by_class]
        ).astype(np.int64, copy=False)
        for _ in range(BOOTSTRAP_REPLICATES)
    )


def _interval(values: Sequence[float]) -> dict[str, Any]:
    lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
    return {
        "lower": float(lower),
        "upper": float(upper),
        "replicates": BOOTSTRAP_REPLICATES,
        "method": "episode-level-stratified-percentile",
    }


def classification_metrics(
    truth: Sequence[str],
    predicted: Sequence[str],
    *,
    displayed: Sequence[str] | None = None,
    bootstrap_indices: Sequence[NDArray[np.int64]] | None = None,
) -> dict[str, Any]:
    truth_values = tuple(truth)
    predictions = tuple(predicted)
    displayed_values = predictions if displayed is None else tuple(displayed)
    if not truth_values or not (
        len(truth_values) == len(predictions) == len(displayed_values)
    ):
        raise ValueError("metric inputs must be aligned and non-empty")
    if any(value not in SCENARIOS for value in truth_values):
        raise ValueError("metric truth contains an undeclared class")
    if any(value not in SCENARIOS for value in predictions):
        raise ValueError("metric predictions contain an undeclared class")

    confusion = tuple(
        tuple(
            sum(
                actual == row and result == column
                for actual, result in zip(truth_values, predictions, strict=True)
            )
            for column in SCENARIOS
        )
        for row in SCENARIOS
    )
    class_rows: dict[str, Any] = {}
    for index, label in enumerate(SCENARIOS):
        support = sum(truth == label for truth in truth_values)
        true_positive = confusion[index][index]
        predicted_count = sum(row[index] for row in confusion)
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted_count if predicted_count else 0.0
        f1_denominator = precision + recall
        class_rows[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": 0.0 if f1_denominator == 0.0 else 2 * precision * recall / f1_denominator,
        }
    balanced, macro = _metric_pair(truth_values, predictions)
    accuracy = sum(
        actual == result
        for actual, result in zip(truth_values, predictions, strict=True)
    ) / len(truth_values)
    accepted = sum(value != ABSTAIN_CLASS for value in displayed_values)
    indices = tuple(bootstrap_indices or _stratified_bootstrap_indices(truth_values))
    truth_array = np.asarray(truth_values, dtype=str)
    predicted_array = np.asarray(predictions, dtype=str)
    bootstrap_scores = [
        _metric_pair(truth_array[item].tolist(), predicted_array[item].tolist())
        for item in indices
    ]
    return {
        "sample_count": len(truth_values),
        "class_order": list(SCENARIOS),
        "confusion_matrix": [list(row) for row in confusion],
        "accuracy": float(accuracy),
        "balanced_accuracy": balanced,
        "balanced_accuracy_interval": _interval([item[0] for item in bootstrap_scores]),
        "macro_f1": macro,
        "macro_f1_interval": _interval([item[1] for item in bootstrap_scores]),
        "coverage": accepted / len(displayed_values),
        "abstention_count": len(displayed_values) - accepted,
        "class_metrics": class_rows,
        "normal_false_positive_episode_rate": 1.0 - class_rows["NORMAL"]["recall"],
    }


def _paired_comparison(
    truth: Sequence[str],
    candidate: Sequence[str],
    reference: Sequence[str],
    bootstrap_indices: Sequence[NDArray[np.int64]],
) -> dict[str, Any]:
    candidate_point = _metric_pair(truth, candidate)
    reference_point = _metric_pair(truth, reference)
    truth_array = np.asarray(truth, dtype=str)
    candidate_array = np.asarray(candidate, dtype=str)
    reference_array = np.asarray(reference, dtype=str)
    balanced_deltas: list[float] = []
    macro_deltas: list[float] = []
    for indices in bootstrap_indices:
        candidate_score = _metric_pair(
            truth_array[indices].tolist(), candidate_array[indices].tolist()
        )
        reference_score = _metric_pair(
            truth_array[indices].tolist(), reference_array[indices].tolist()
        )
        balanced_deltas.append(candidate_score[0] - reference_score[0])
        macro_deltas.append(candidate_score[1] - reference_score[1])
    return {
        "balanced_accuracy_delta": candidate_point[0] - reference_point[0],
        "balanced_accuracy_delta_interval": _interval(balanced_deltas),
        "macro_f1_delta": candidate_point[1] - reference_point[1],
        "macro_f1_delta_interval": _interval(macro_deltas),
    }


def evaluate_frozen_network_model(
    observations: Sequence[FourSensorStudyObservation],
    labels: Sequence[FourSensorStudyLabel],
    *,
    model_binding: dict[str, Any],
    protocol_freeze_id: str,
    dataset_id: str,
    authorization_id: str,
    root=None,
) -> dict[str, Any]:
    """Evaluate one pre-existing bundle once; this function has no fit path."""

    if len(observations) != 100 or len(labels) != 100:
        raise ValueError("final held-out evaluation requires exactly 100 episodes")
    runtime = load_bound_network_runtime(model_binding, root=root)
    label_by_episode = {item.episode_id: item for item in labels}
    if len(label_by_episode) != 100:
        raise ValueError("final TEST labels must use unique episode identities")
    ordered_labels: list[FourSensorStudyLabel] = []
    feature_rows: list[tuple[float, ...]] = []
    sample_ids: list[str] = []
    for observation in observations:
        label = label_by_episode.get(observation.episode_id)
        if label is None or label.generative_lineage_id != observation.generative_lineage_id:
            raise ValueError("final TEST observation and label channels are misaligned")
        values = observation.network_feature.values
        if not observation.network_feature.quality.valid_for_quantum or any(
            value is None for value in values
        ):
            raise ValueError("final TEST contains an ineligible primary State8 feature")
        ordered_labels.append(label)
        feature_rows.append(tuple(float(value) for value in values))
        sample_ids.append(observation.episode_id)

    matrix = np.asarray(feature_rows, dtype=np.float64)
    query_id = f"{dataset_id}:single-authorized-final-test"
    prediction = runtime.predict(
        matrix,
        sample_ids=tuple(sample_ids),
        context=BundleQueryContext(
            bundle_id=runtime.artifact.bundle_id,
            task_id=runtime.artifact.task_id,
            profile_id=runtime.artifact.profile_id,
            profile_fingerprint=runtime.artifact.profile_fingerprint,
            query_acquisition_id=query_id,
        ),
    )
    truth = tuple(item.scenario for item in ordered_labels)
    quantum = tuple(prediction.model_scores.predicted_classes)
    raw = tuple(prediction.raw_baseline_scores.predicted_classes)
    observable = tuple(_OBSERVABLE_CLASS[value] for value in prediction.observable_rule_statuses)
    bootstrap = _stratified_bootstrap_indices(truth)
    metrics = {
        "frozen_quantum_afse_mlp": classification_metrics(
            truth,
            quantum,
            displayed=prediction.displayed_classes,
            bootstrap_indices=bootstrap,
        ),
        "raw_state8_mlp": classification_metrics(
            truth,
            raw,
            displayed=prediction.raw_baseline_scores.displayed_classes,
            bootstrap_indices=bootstrap,
        ),
        "observable_p99_rule": classification_metrics(
            truth,
            observable,
            bootstrap_indices=bootstrap,
        ),
    }
    comparisons = {
        "quantum_vs_raw_state8_mlp": _paired_comparison(
            truth, quantum, raw, bootstrap
        ),
        "quantum_vs_observable_p99_rule": _paired_comparison(
            truth, quantum, observable, bootstrap
        ),
    }

    def subgroup(field: str) -> dict[str, Any]:
        keys = sorted({str(getattr(item, field)) for item in ordered_labels})
        rows: dict[str, Any] = {}
        for key in keys:
            indices = [
                index
                for index, item in enumerate(ordered_labels)
                if str(getattr(item, field)) == key
            ]
            rows[key] = {
                "sample_count": len(indices),
                "quantum_balanced_accuracy": _metric_pair(
                    [truth[index] for index in indices],
                    [quantum[index] for index in indices],
                )[0],
                "raw_balanced_accuracy": _metric_pair(
                    [truth[index] for index in indices],
                    [raw[index] for index in indices],
                )[0],
            }
        return rows

    primary = metrics["frozen_quantum_afse_mlp"]
    raw_delta = comparisons["quantum_vs_raw_state8_mlp"]
    criteria = {
        "balanced_accuracy_at_least_0_75": (
            primary["balanced_accuracy"] >= PRIMARY_BALANCED_ACCURACY_THRESHOLD
        ),
        "macro_f1_at_least_0_75": primary["macro_f1"] >= PRIMARY_MACRO_F1_THRESHOLD,
        "each_class_recall_at_least_0_70": all(
            row["recall"] >= MINIMUM_CLASS_RECALL
            for row in primary["class_metrics"].values()
        ),
        "normal_recall_at_least_0_80": (
            primary["class_metrics"]["NORMAL"]["recall"] >= MINIMUM_NORMAL_RECALL
        ),
        "coverage_at_least_0_90": primary["coverage"] >= MINIMUM_COVERAGE,
        "normal_false_positive_rate_at_most_0_20": (
            primary["normal_false_positive_episode_rate"]
            <= MAXIMUM_NORMAL_FALSE_POSITIVE_RATE
        ),
        "quantum_advantage_vs_raw_predeclared_rule": (
            raw_delta["balanced_accuracy_delta"] >= MINIMUM_ADVANTAGE_DELTA
            and raw_delta["balanced_accuracy_delta_interval"]["lower"] > 0.0
            and primary["coverage"] >= metrics["raw_state8_mlp"]["coverage"]
            and primary["class_metrics"]["NORMAL"]["recall"]
            >= metrics["raw_state8_mlp"]["class_metrics"]["NORMAL"]["recall"]
        ),
    }
    scientific = {
        "schema_version": "aqse.four-sensor-study.final-evaluation.v1",
        "scientific_label": "simulation-only research evidence;not field validation",
        "protocol_freeze_id": protocol_freeze_id,
        "dataset_id": dataset_id,
        "authorization_id": authorization_id,
        "query_acquisition_id": query_id,
        "model_binding": model_binding,
        "sample_count": len(truth),
        "class_counts": dict(sorted(Counter(truth).items())),
        "geometry_counts": dict(
            sorted(Counter(item.geometry_id for item in ordered_labels).items())
        ),
        "focal_sensor_counts": dict(
            sorted(Counter(item.focal_sensor_id for item in ordered_labels).items())
        ),
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "by_geometry": subgroup("geometry_id"),
        "by_focal_sensor": subgroup("focal_sensor_id"),
        "success_criteria": criteria,
        "all_primary_criteria_met": all(
            value
            for key, value in criteria.items()
            if key != "quantum_advantage_vs_raw_predeclared_rule"
        ),
        "quantum_advantage_demonstrated": criteria[
            "quantum_advantage_vs_raw_predeclared_rule"
        ],
        "fit_performed": False,
        "selection_changed": False,
        "test_access": "single-authorized-held-out-evaluation",
        "predictions": [
            {
                "episode_id": sample_ids[index],
                "geometry_id": ordered_labels[index].geometry_id,
                "focal_sensor_id": ordered_labels[index].focal_sensor_id,
                "truth": truth[index],
                "quantum_prediction": quantum[index],
                "quantum_displayed": prediction.displayed_classes[index],
                "raw_prediction": raw[index],
                "observable_prediction": observable[index],
            }
            for index in range(len(sample_ids))
        ],
    }
    digest = _digest(scientific)
    return {
        **scientific,
        "evaluation_id": f"aqse-four-sensor-final-{digest[:16]}",
        "content_digest": digest,
    }


def validate_evaluation_binding(
    evaluation: dict[str, Any],
    *,
    protocol_freeze_id: str,
    dataset_id: str,
    authorization_id: str,
) -> None:
    if (
        evaluation.get("protocol_freeze_id") != protocol_freeze_id
        or evaluation.get("dataset_id") != dataset_id
        or evaluation.get("authorization_id") != authorization_id
        or evaluation.get("fit_performed") is not False
        or evaluation.get("selection_changed") is not False
    ):
        raise ValueError("final evaluation binding is invalid")
    payload = {
        key: value
        for key, value in evaluation.items()
        if key not in {"evaluation_id", "content_digest"}
    }
    digest = _digest(payload)
    if evaluation.get("content_digest") != digest:
        raise ValueError("final evaluation content digest is invalid")
    if evaluation.get("evaluation_id") != f"aqse-four-sensor-final-{digest[:16]}":
        raise ValueError("final evaluation identity is invalid")
