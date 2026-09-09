from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.classical.mlp import (
    FittedMLP,
    fit_mlp_classifier,
    fit_raw_feature_baseline,
)
from app.classical.mlp import (
    query_context_for as mlp_context_for,
)
from app.embeddings.nystrom import (
    NystromAFSE,
    fit_nystrom_afse,
)
from app.embeddings.nystrom import (
    query_context_for as afse_context_for,
)
from app.features.state8 import state8_profile, state8_profile_fingerprint
from app.features.state8_encoding import State8AngleEncoder, fit_state8_encoder
from app.training.canonical import file_sha256

from .bundle_models import (
    ABSTAIN_CLASS,
    LOCAL_TASK_ID,
    NETWORK_TASK_ID,
    BootstrapInterval,
    BundleQueryContext,
    BundleRuntime,
    CandidateValidationEvidence,
    DemoEvaluationMetrics,
    DemoFinalEvaluation,
    DemoModelSelectionFreeze,
    DemoResearchBundle,
    DemoTaskExample,
    DemoTaskPartition,
    DemoThetaCandidate,
    TaskId,
    TaskSelection,
    canonical_digest,
)
from .protocol import FROZEN_NETWORK_DEMO_PROTOCOL
from .study_models import LoadedStudyPartition
from .training import (
    DemoQngCandidate,
    DemoQngStep,
    canonical_demo_theta0,
    train_demo_qng,
)

BOOTSTRAP_REPLICATES = 1_000


class DemoTrainingCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class DemoTaskFitResult:
    task_id: TaskId
    train_dataset_id: str
    train_dataset_digest: str
    bundles: tuple[DemoResearchBundle, ...]
    qng_candidate: DemoQngCandidate


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "app/demo/bundle_models.py",
        "app/demo/evaluation.py",
        "app/demo/training.py",
        "app/features/state8_encoding.py",
        "app/embeddings/nystrom.py",
        "app/classical/mlp.py",
        "app/quantum/user_pipeline/tqk8.py",
    )
    return {name: file_sha256(root / name) for name in names}


def assemble_task_partition(
    loaded: LoadedStudyPartition,
    *,
    task_id: TaskId,
    study_artifact_id: str,
    study_content_digest: str,
) -> DemoTaskPartition:
    """Join separately loaded labels to observable features for an explicit study task."""

    if loaded.labels is None:
        raise PermissionError("task assembly requires an explicitly loaded label channel")
    label_by_episode = {item.episode_id: item for item in loaded.labels}
    if len(label_by_episode) != len(loaded.labels):
        raise ValueError("study label episode identifiers must be distinct")
    profile_id = (
        "aqse.local-state8.v1" if task_id == LOCAL_TASK_ID else "aqse.network-state8.v1"
    )
    class_order = (
        ("NORMAL", "CHANGE_DETECTED")
        if task_id == LOCAL_TASK_ID
        else (
            "NORMAL",
            "ENVIRONMENT_COMPATIBLE",
            "DEVICE_COMPATIBLE",
            "MIXED_OR_AMBIGUOUS",
        )
    )
    examples: list[DemoTaskExample] = []
    for observation in loaded.observations:
        if observation.partition != loaded.partition:
            raise ValueError("study observation belongs to a different partition")
        if task_id == NETWORK_TASK_ID and observation.node_count < 3:
            continue
        label = label_by_episode.get(observation.episode_id)
        if label is None:
            raise ValueError("study observation has no matching supervisory record")
        if label.partition != loaded.partition:
            raise ValueError("study label belongs to a different partition")
        if label.generative_lineage_id != observation.generative_lineage_id:
            raise ValueError("observation and label generative lineages differ")
        record = (
            observation.local_feature
            if task_id == LOCAL_TASK_ID
            else observation.network_feature
        )
        if record is None or record.profile_id != profile_id:
            raise ValueError("study observation lacks its compatible task feature")
        target = label.local_label if task_id == LOCAL_TASK_ID else label.network_label
        examples.append(
            DemoTaskExample(
                sample_id=observation.episode_id,
                episode_id=observation.episode_id,
                generative_lineage_id=observation.generative_lineage_id,
                node_count=observation.node_count,
                label=target,
                feature_values=record.values,
                eligible=record.quality.valid_for_quantum,
                quality_flags=record.quality.flags,
            )
        )
    if set(label_by_episode) != {item.episode_id for item in loaded.observations}:
        raise ValueError("study label and observation channels are not exactly aligned")
    profile = state8_profile(profile_id)
    return DemoTaskPartition(
        study_artifact_id=study_artifact_id,
        study_content_digest=study_content_digest,
        partition=loaded.partition,
        task_id=task_id,
        profile_id=profile_id,
        profile_fingerprint=state8_profile_fingerprint(profile),
        class_order=class_order,
        examples=tuple(examples),
    )


def _compatible_partitions(train: DemoTaskPartition, validation: DemoTaskPartition) -> None:
    if train.partition != "train" or validation.partition != "validation":
        raise ValueError("candidate fitting requires TRAIN and VALIDATION partitions")
    fields = (
        "study_artifact_id",
        "study_content_digest",
        "task_id",
        "profile_id",
        "profile_fingerprint",
        "class_order",
    )
    if any(getattr(train, field) != getattr(validation, field) for field in fields):
        raise ValueError("TRAIN and VALIDATION task partitions are incompatible")
    train_lineages = {item.generative_lineage_id for item in train.examples}
    validation_lineages = {item.generative_lineage_id for item in validation.examples}
    if train_lineages & validation_lineages:
        raise ValueError("TRAIN and VALIDATION generative lineages overlap")


def _eligible_rows(
    partition: DemoTaskPartition,
) -> tuple[
    NDArray[np.float64],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    eligible = tuple(item for item in partition.examples if item.eligible)
    if not eligible:
        raise ValueError(f"{partition.partition} contains no eligible task examples")
    matrix = np.asarray([item.feature_values for item in eligible], dtype=np.float64)
    return (
        matrix,
        tuple(item.sample_id for item in eligible),
        tuple(item.generative_lineage_id for item in eligible),
        tuple(item.label for item in eligible),
    )


def _fit_dataset_identity(partition: DemoTaskPartition) -> tuple[str, str]:
    matrix, sample_ids, lineages, labels = _eligible_rows(partition)
    payload = {
        "schema_version": "aqse.network-demo.fit-dataset.v1",
        "study_artifact_id": partition.study_artifact_id,
        "study_content_digest": partition.study_content_digest,
        "partition": "TRAIN",
        "task_id": partition.task_id,
        "profile_id": partition.profile_id,
        "profile_fingerprint": partition.profile_fingerprint,
        "sample_ids": sample_ids,
        "lineage_ids": lineages,
        "labels": labels,
        "raw_features": matrix.tolist(),
    }
    digest = canonical_digest(payload)
    return f"aqse-demo-fit-{digest[:16]}", digest


def _theta0_candidate() -> DemoThetaCandidate:
    theta = tuple(float(value) for value in canonical_demo_theta0())
    candidate_id = f"aqse-demo-theta0-{canonical_digest({'theta': theta})[:16]}"
    return DemoThetaCandidate(
        candidate_name="theta0",
        candidate_id=candidate_id,
        eligible=True,
        theta=theta,
        accepted_updates=0,
        stop_reason="DETERMINISTIC_INITIALIZATION",
    )


def _qng_summary(candidate: DemoQngCandidate) -> DemoThetaCandidate:
    return DemoThetaCandidate(
        candidate_name="protected_qng",
        candidate_id=candidate.candidate_id,
        eligible=candidate.eligible,
        theta=candidate.final_theta,
        qng_bank_sample_ids=candidate.bank_sample_ids,
        qng_bank_lineage_ids=candidate.bank_lineage_ids,
        accepted_updates=candidate.accepted_updates,
        stop_reason=candidate.stop_reason,
        initial_alignment_loss=candidate.initial_loss,
        final_alignment_loss=candidate.final_loss,
    )


def _ineligible_qng_candidate(
    *,
    profile_id: str,
    reason: str,
) -> DemoQngCandidate:
    theta0 = tuple(float(value) for value in canonical_demo_theta0())
    payload = {
        "profile_id": profile_id,
        "eligible": False,
        "theta0": theta0,
        "reason": reason,
    }
    return DemoQngCandidate(
        candidate_id=f"aqse-demo-qng-{canonical_digest(payload)[:16]}",
        eligible=False,
        profile_id=profile_id,
        bank_sample_ids=(),
        bank_lineage_ids=(),
        theta0=theta0,
        final_theta=theta0,
        accepted_updates=0,
        stop_reason=reason,
        initial_loss=None,
        final_loss=None,
        steps=(),
        total_wall_time_ms=0.0,
    )


def _metric_pair(
    truth: Sequence[str],
    predicted: Sequence[str],
    class_order: Sequence[str],
) -> tuple[float, float]:
    recalls: list[float] = []
    f1_values: list[float] = []
    for label in class_order:
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
        f1_values.append((2 * true_positive / denominator) if denominator else 0.0)
    if not recalls:
        return 0.0, 0.0
    return float(np.mean(recalls)), float(np.mean(f1_values))


def _confusion(
    truth: Sequence[str],
    predicted: Sequence[str],
    rows: Sequence[str],
    columns: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            sum(
                actual == row and result == column
                for actual, result in zip(truth, predicted, strict=True)
            )
            for column in columns
        )
        for row in rows
    )


def _bootstrap_intervals(
    truth: Sequence[str],
    predicted: Sequence[str],
    class_order: Sequence[str],
) -> tuple[BootstrapInterval, BootstrapInterval]:
    generator = np.random.default_rng(FROZEN_NETWORK_DEMO_PROTOCOL.seeds.paired_bootstrap)
    size = len(truth)
    balanced: list[float] = []
    macro: list[float] = []
    truth_array = np.asarray(truth, dtype=str)
    predicted_array = np.asarray(predicted, dtype=str)
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = generator.integers(0, size, size=size)
        score = _metric_pair(
            truth_array[indices].tolist(),
            predicted_array[indices].tolist(),
            class_order,
        )
        balanced.append(score[0])
        macro.append(score[1])

    def interval(values: list[float]) -> BootstrapInterval:
        lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
        return BootstrapInterval(
            lower=float(lower),
            upper=float(upper),
            replicates=BOOTSTRAP_REPLICATES,
            degenerate=bool(lower == upper),
        )

    return interval(balanced), interval(macro)


def _metrics_from_predictions(
    partition: DemoTaskPartition,
    *,
    class_order: tuple[str, ...],
    raw_by_sample: Mapping[str, str],
    displayed_by_sample: Mapping[str, str],
    uncertain_samples: frozenset[str],
    ood_samples: frozenset[str],
) -> DemoEvaluationMetrics:
    truth = tuple(item.label for item in partition.examples)
    raw = tuple(raw_by_sample.get(item.sample_id, ABSTAIN_CLASS) for item in partition.examples)
    displayed = tuple(
        displayed_by_sample.get(item.sample_id, ABSTAIN_CLASS)
        for item in partition.examples
    )
    columns = (*class_order, ABSTAIN_CLASS)
    balanced, macro = _metric_pair(truth, raw, class_order)
    balanced_interval, macro_interval = _bootstrap_intervals(truth, raw, class_order)
    abstentions = sum(value == ABSTAIN_CLASS for value in displayed)
    quality_rejected = sum(not item.eligible for item in partition.examples)
    return DemoEvaluationMetrics(
        partition=partition.partition,
        task_id=partition.task_id,
        sample_count=len(partition.examples),
        eligible_count=len(partition.examples) - quality_rejected,
        class_order=class_order,
        class_support={label: Counter(truth)[label] for label in class_order},
        raw_confusion_columns=columns,
        raw_confusion_matrix=_confusion(truth, raw, class_order, columns),
        operational_confusion_columns=columns,
        operational_confusion_matrix=_confusion(truth, displayed, class_order, columns),
        balanced_accuracy=balanced,
        macro_f1=macro,
        coverage=(len(partition.examples) - abstentions) / len(partition.examples),
        abstention_count=abstentions,
        uncertain_count=len(uncertain_samples),
        heuristic_ood_count=len(ood_samples),
        quality_rejected_count=quality_rejected,
        balanced_accuracy_interval=balanced_interval,
        macro_f1_interval=macro_interval,
        predictions=raw,
        displayed_predictions=displayed,
        episode_ids=tuple(item.episode_id for item in partition.examples),
    )


def _evaluate_fitted_candidate(
    partition: DemoTaskPartition,
    *,
    encoder: State8AngleEncoder,
    afse: NystromAFSE,
    classifier: FittedMLP,
) -> DemoEvaluationMetrics:
    raw, sample_ids, _, _ = _eligible_rows(partition)
    profile = state8_profile(partition.profile_id)
    encoded = encoder.transform(raw, profile=profile)
    embedding = afse.transform(
        encoded,
        sample_ids=sample_ids,
        context=afse_context_for(afse.artifact),
    )
    scored = classifier.runtime.score(
        np.asarray(embedding.vectors, dtype=np.float64),
        sample_ids=sample_ids,
        context=mlp_context_for(classifier.artifact),
    )
    raw_by_sample = dict(zip(sample_ids, scored.predicted_classes, strict=True))
    displayed_by_sample = {
        sample_id: (
            ABSTAIN_CLASS if uncertain or ood else predicted
        )
        for sample_id, predicted, uncertain, ood in zip(
            sample_ids,
            scored.predicted_classes,
            scored.uncertain,
            embedding.heuristic_ood,
            strict=True,
        )
    }
    return _metrics_from_predictions(
        partition,
        class_order=classifier.artifact.classes,
        raw_by_sample=raw_by_sample,
        displayed_by_sample=displayed_by_sample,
        uncertain_samples=frozenset(
            sample_id
            for sample_id, uncertain in zip(sample_ids, scored.uncertain, strict=True)
            if uncertain
        ),
        ood_samples=frozenset(
            sample_id
            for sample_id, ood in zip(sample_ids, embedding.heuristic_ood, strict=True)
            if ood
        ),
    )


def _evaluate_raw_baseline(
    partition: DemoTaskPartition,
    classifier: FittedMLP,
) -> DemoEvaluationMetrics:
    raw, sample_ids, _, _ = _eligible_rows(partition)
    scored = classifier.runtime.score(
        raw,
        sample_ids=sample_ids,
        context=mlp_context_for(classifier.artifact),
    )
    raw_by_sample = dict(zip(sample_ids, scored.predicted_classes, strict=True))
    displayed_by_sample = dict(zip(sample_ids, scored.displayed_classes, strict=True))
    return _metrics_from_predictions(
        partition,
        class_order=classifier.artifact.classes,
        raw_by_sample=raw_by_sample,
        displayed_by_sample=displayed_by_sample,
        uncertain_samples=frozenset(
            sample_id
            for sample_id, uncertain in zip(sample_ids, scored.uncertain, strict=True)
            if uncertain
        ),
        ood_samples=frozenset(),
    )


def _make_bundle(
    *,
    train: DemoTaskPartition,
    candidate: DemoThetaCandidate,
    encoder: State8AngleEncoder,
    afse: NystromAFSE,
    classifier: FittedMLP,
    raw_baseline: FittedMLP,
    validation_metrics: DemoEvaluationMetrics,
    raw_validation_metrics: DemoEvaluationMetrics,
    train_dataset_id: str,
    train_dataset_digest: str,
) -> DemoResearchBundle:
    scientific: dict[str, Any] = {
        "schema_version": "aqse.network-demo.bundle.v1",
        "scientific_label": "research / not validated for field deployment",
        "study_artifact_id": train.study_artifact_id,
        "study_content_digest": train.study_content_digest,
        "protocol_digest": FROZEN_NETWORK_DEMO_PROTOCOL.digest,
        "task_id": train.task_id,
        "profile_id": train.profile_id,
        "profile_fingerprint": train.profile_fingerprint,
        "fitted_on_partition": "TRAIN",
        "fitted_on_dataset_id": train_dataset_id,
        "fitted_on_dataset_digest": train_dataset_digest,
        "class_order": list(classifier.artifact.classes),
        "theta_candidate": candidate.model_dump(mode="json"),
        "encoder": encoder.artifact.model_dump(mode="json"),
        "afse": afse.artifact.model_dump(mode="json"),
        "classifier": classifier.artifact.model_dump(mode="json"),
        "raw_feature_baseline": raw_baseline.artifact.model_dump(mode="json"),
        "validation_metrics": validation_metrics.model_dump(mode="json"),
        "raw_baseline_validation_metrics": raw_validation_metrics.model_dump(
            mode="json"
        ),
        "effective_source_hashes": _source_hashes(),
    }
    content_digest = canonical_digest(scientific)
    return DemoResearchBundle.model_validate(
        {
            **scientific,
            "bundle_id": f"aqse-demo-bundle-{content_digest[:16]}",
            "content_digest": content_digest,
        }
    )


def fit_demo_task_candidates(
    train: DemoTaskPartition,
    validation: DemoTaskPartition,
    *,
    cancellation: Event | None = None,
    progress: Callable[[TaskId, DemoQngStep], None] | None = None,
) -> DemoTaskFitResult:
    """Fit the fixed theta0/QNG candidate budget and downstream TRAIN-only models."""

    _compatible_partitions(train, validation)
    cancellation = cancellation or Event()
    if cancellation.is_set():
        raise DemoTrainingCancelled("demo training cancelled before fitting")
    train_raw, train_ids, train_lineages, train_labels = _eligible_rows(train)
    profile = state8_profile(train.profile_id)
    if state8_profile_fingerprint(profile) != train.profile_fingerprint:
        raise ValueError("task partition profile fingerprint is incompatible")
    train_dataset_id, train_dataset_digest = _fit_dataset_identity(train)
    encoder = fit_state8_encoder(
        train_raw,
        profile=profile,
        partition="train",
        training_identity=train_dataset_id,
    )
    encoded_train = encoder.transform(train_raw, profile=profile)

    if train.task_id == LOCAL_TASK_ID:
        qng_indices = np.arange(len(train_labels), dtype=np.int64)
        qng_labels = tuple(-1 if value == "NORMAL" else 1 for value in train_labels)
    else:
        qng_indices = np.asarray(
            [
                index
                for index, value in enumerate(train_labels)
                if value in {"ENVIRONMENT_COMPATIBLE", "DEVICE_COMPATIBLE"}
            ],
            dtype=np.int64,
        )
        qng_labels = tuple(
            -1 if train_labels[int(index)] == "ENVIRONMENT_COMPATIBLE" else 1
            for index in qng_indices
        )
    if set(qng_labels) != {-1, 1}:
        qng = _ineligible_qng_candidate(
            profile_id=train.profile_id,
            reason="INSUFFICIENT_BINARY_LINEAGES",
        )
    else:
        qng = train_demo_qng(
            encoded_train[qng_indices],
            qng_labels,
            sample_ids=tuple(train_ids[int(index)] for index in qng_indices),
            lineage_ids=tuple(train_lineages[int(index)] for index in qng_indices),
            profile_id=train.profile_id,
            cancellation=cancellation,
            progress=(
                None
                if progress is None
                else lambda step: progress(train.task_id, step)
            ),
        )
    if cancellation.is_set() or qng.stop_reason == "CANCELLED":
        raise DemoTrainingCancelled("demo training cancelled during QNG")
    candidate_specs = [_theta0_candidate()]
    if qng.eligible:
        candidate_specs.append(_qng_summary(qng))

    raw_baseline = fit_raw_feature_baseline(
        train_raw,
        train_labels,
        sample_ids=train_ids,
        task_id=train.task_id,
        input_space_id=f"{encoder.artifact.encoder_id}:raw",
        fitted_on_dataset_id=train_dataset_id,
        fitted_on_dataset_digest=train_dataset_digest,
        feature_order=encoder.artifact.feature_order,
    )
    raw_validation_metrics = _evaluate_raw_baseline(validation, raw_baseline)
    bundles: list[DemoResearchBundle] = []
    for candidate in candidate_specs:
        if cancellation.is_set():
            raise DemoTrainingCancelled("demo training cancelled before AFSE fitting")
        afse = fit_nystrom_afse(
            encoded_train,
            sample_ids=train_ids,
            lineage_ids=train_lineages,
            classes=train_labels,
            theta=candidate.theta,
            feature_profile_id=train.profile_id,
            scaler_id=encoder.artifact.scaler_id,
            encoding_policy_id=encoder.artifact.encoding_policy_id,
            fitted_on_dataset_id=train_dataset_id,
            fitted_on_dataset_digest=train_dataset_digest,
        )
        latent_train = afse.transform(
            encoded_train,
            sample_ids=train_ids,
            context=afse_context_for(afse.artifact),
        )
        classifier = fit_mlp_classifier(
            np.asarray(latent_train.vectors, dtype=np.float64),
            train_labels,
            sample_ids=train_ids,
            task_id=train.task_id,
            input_space_id=afse.artifact.artifact_id,
            fitted_on_dataset_id=train_dataset_id,
            fitted_on_dataset_digest=train_dataset_digest,
            feature_order=tuple(
                f"z{index}" for index in range(afse.artifact.output_dimension)
            ),
        )
        validation_metrics = _evaluate_fitted_candidate(
            validation,
            encoder=encoder,
            afse=afse,
            classifier=classifier,
        )
        bundles.append(
            _make_bundle(
                train=train,
                candidate=candidate,
                encoder=encoder,
                afse=afse,
                classifier=classifier,
                raw_baseline=raw_baseline,
                validation_metrics=validation_metrics,
                raw_validation_metrics=raw_validation_metrics,
                train_dataset_id=train_dataset_id,
                train_dataset_digest=train_dataset_digest,
            )
        )
    return DemoTaskFitResult(
        task_id=train.task_id,
        train_dataset_id=train_dataset_id,
        train_dataset_digest=train_dataset_digest,
        bundles=tuple(bundles),
        qng_candidate=qng,
    )


def select_task_candidate(result: DemoTaskFitResult) -> TaskSelection:
    if not result.bundles:
        raise ValueError("task selection requires at least one complete candidate bundle")
    for bundle in result.bundles:
        if bundle.task_id != result.task_id:
            raise ValueError("candidate bundle belongs to a different task")
    selected = max(
        result.bundles,
        key=lambda item: (
            item.validation_metrics.balanced_accuracy,
            item.validation_metrics.macro_f1,
            item.theta_candidate.candidate_name == "theta0",
        ),
    )
    evidence = tuple(
        CandidateValidationEvidence(
            candidate_name=item.theta_candidate.candidate_name,
            bundle_id=item.bundle_id,
            balanced_accuracy=item.validation_metrics.balanced_accuracy,
            macro_f1=item.validation_metrics.macro_f1,
        )
        for item in result.bundles
    )
    return TaskSelection(
        task_id=result.task_id,
        candidate_evidence=evidence,
        selected_bundle_id=selected.bundle_id,
        selected_candidate_name=selected.theta_candidate.candidate_name,
    )


def freeze_model_selection(
    local: DemoTaskFitResult,
    network: DemoTaskFitResult,
) -> DemoModelSelectionFreeze:
    if local.task_id != LOCAL_TASK_ID or network.task_id != NETWORK_TASK_ID:
        raise ValueError("model-selection freeze requires local then network fit results")
    all_bundles = (*local.bundles, *network.bundles)
    study_ids = {item.study_artifact_id for item in all_bundles}
    study_digests = {item.study_content_digest for item in all_bundles}
    protocol_digests = {item.protocol_digest for item in all_bundles}
    if len(study_ids) != 1 or len(study_digests) != 1 or protocol_digests != {
        FROZEN_NETWORK_DEMO_PROTOCOL.digest
    }:
        raise ValueError("local and network bundles do not share one frozen study")
    scientific = {
        "schema_version": "aqse.network-demo.selection-freeze.v1",
        "study_artifact_id": next(iter(study_ids)),
        "study_content_digest": next(iter(study_digests)),
        "protocol_digest": FROZEN_NETWORK_DEMO_PROTOCOL.digest,
        "selections": [
            select_task_candidate(local).model_dump(mode="json"),
            select_task_candidate(network).model_dump(mode="json"),
        ],
        "test_state": "sealed",
        "final_evaluation_rule": (
            "single-new-test-evaluation-after-selection-freeze;no-refit-or-reselection"
        ),
    }
    digest = canonical_digest(scientific)
    return DemoModelSelectionFreeze.model_validate(
        {
            **scientific,
            "freeze_id": f"aqse-demo-freeze-{digest[:16]}",
            "content_digest": digest,
        }
    )


def evaluate_bundle(
    bundle: DemoResearchBundle,
    partition: DemoTaskPartition,
    *,
    query_acquisition_id: str,
) -> DemoEvaluationMetrics:
    """Evaluate a frozen bundle without fitting or changing model selection."""

    if partition.partition not in {"validation", "test"}:
        raise ValueError("frozen-bundle evaluation accepts VALIDATION or TEST only")
    if (
        partition.task_id != bundle.task_id
        or partition.profile_id != bundle.profile_id
        or partition.profile_fingerprint != bundle.profile_fingerprint
    ):
        raise ValueError("evaluation partition is incompatible with the frozen bundle")
    raw, sample_ids, _, _ = _eligible_rows(partition)
    runtime = BundleRuntime.from_artifact(bundle)
    prediction = runtime.predict(
        raw,
        sample_ids=sample_ids,
        context=BundleQueryContext(
            bundle_id=bundle.bundle_id,
            task_id=bundle.task_id,
            profile_id=bundle.profile_id,
            profile_fingerprint=bundle.profile_fingerprint,
            query_acquisition_id=query_acquisition_id,
        ),
    )
    raw_by_sample = dict(
        zip(sample_ids, prediction.model_scores.predicted_classes, strict=True)
    )
    displayed_by_sample = dict(zip(sample_ids, prediction.displayed_classes, strict=True))
    return _metrics_from_predictions(
        partition,
        class_order=bundle.class_order,
        raw_by_sample=raw_by_sample,
        displayed_by_sample=displayed_by_sample,
        uncertain_samples=frozenset(
            sample_id
            for sample_id, uncertain in zip(
                sample_ids,
                prediction.model_scores.uncertain,
                strict=True,
            )
            if uncertain
        ),
        ood_samples=frozenset(
            sample_id
            for sample_id, ood in zip(
                sample_ids,
                prediction.heuristic_ood,
                strict=True,
            )
            if ood
        ),
    )


def evaluate_frozen_test(
    freeze: DemoModelSelectionFreeze,
    bundles: Mapping[str, DemoResearchBundle],
    *,
    local_test: DemoTaskPartition,
    network_test: DemoTaskPartition,
    query_acquisition_id: str,
) -> DemoFinalEvaluation:
    """Run the single TEST evaluation from an immutable selection; never fit/select here."""

    partitions = (local_test, network_test)
    if tuple(item.task_id for item in partitions) != (LOCAL_TASK_ID, NETWORK_TASK_ID):
        raise ValueError("final evaluation requires ordered local and network TEST data")
    if any(item.partition != "test" for item in partitions):
        raise ValueError("final evaluation cannot use a non-TEST partition")
    metrics: list[DemoEvaluationMetrics] = []
    for selection, partition in zip(freeze.selections, partitions, strict=True):
        bundle = bundles.get(selection.selected_bundle_id)
        if bundle is None:
            raise ValueError("selected frozen bundle is unavailable")
        if (
            bundle.study_artifact_id != freeze.study_artifact_id
            or bundle.study_content_digest != freeze.study_content_digest
            or partition.study_artifact_id != freeze.study_artifact_id
            or partition.study_content_digest != freeze.study_content_digest
        ):
            raise ValueError("TEST data or selected bundle differs from the frozen study")
        metrics.append(
            evaluate_bundle(
                bundle,
                partition,
                query_acquisition_id=query_acquisition_id,
            )
        )
    scientific = {
        "schema_version": "aqse.network-demo.final-evaluation.v1",
        "freeze_id": freeze.freeze_id,
        "study_artifact_id": freeze.study_artifact_id,
        "study_content_digest": freeze.study_content_digest,
        "query_acquisition_id": query_acquisition_id,
        "test_access": "single-authorized-access",
        "selection_changed": False,
        "refit_performed": False,
        "task_metrics": [item.model_dump(mode="json") for item in metrics],
    }
    digest = canonical_digest(scientific)
    return DemoFinalEvaluation.model_validate(
        {
            **scientific,
            "evaluation_id": f"aqse-demo-final-{digest[:16]}",
            "content_digest": digest,
        }
    )
