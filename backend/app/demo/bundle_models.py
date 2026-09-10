from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from app.classical.mlp import NumpyMLPClassifier
from app.classical.mlp import query_context_for as mlp_context_for
from app.classical.models import MLPClassifierArtifact, MLPScoreBatch
from app.classical.observable_rule import (
    ObservableRule,
    ObservableRuleArtifact,
    ObservableRuleStatus,
    observable_rule_query,
    validate_observable_rule_artifact,
)
from app.embeddings.nystrom import (
    NystromAFSE,
    NystromAFSEArtifact,
    validate_afse_artifact,
)
from app.embeddings.nystrom import (
    query_context_for as afse_context_for,
)
from app.features.state8_encoding import (
    State8AngleEncoder,
    State8EncodingArtifact,
    load_state8_encoder,
    validate_state8_encoder_artifact,
)
from app.features.state8_models import State8FeatureProfile
from app.training.canonical import canonical_json_bytes
from app.training.models import FrozenModel

TaskId = Literal["aqse.local-change.v1", "aqse.network-pattern.v1"]
ThetaCandidateName = Literal["theta0", "protected_qng"]
TaskPartitionName = Literal["train", "validation", "test"]
FeatureValues = tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]

LOCAL_TASK_ID: Literal["aqse.local-change.v1"] = "aqse.local-change.v1"
NETWORK_TASK_ID: Literal["aqse.network-pattern.v1"] = "aqse.network-pattern.v1"
ABSTAIN_CLASS = "ABSTAIN"


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class DemoTaskExample(FrozenModel):
    """One independent episode/focal-node reporting unit."""

    sample_id: str = Field(min_length=1, max_length=256)
    episode_id: str = Field(min_length=1, max_length=256)
    generative_lineage_id: str = Field(min_length=1, max_length=256)
    node_count: int = Field(ge=1, le=8)
    label: str = Field(min_length=1, max_length=128)
    feature_values: FeatureValues
    eligible: bool
    quality_flags: tuple[str, ...] = ()
    replay_feature_values: tuple[FeatureValues, ...] = ()
    replay_window_end_s: tuple[float, ...] = ()
    event_start_s: float | None = None

    @model_validator(mode="after")
    def validate_feature_availability(self) -> DemoTaskExample:
        complete = all(value is not None for value in self.feature_values)
        if self.eligible != complete:
            raise ValueError("task-example eligibility must match feature completeness")
        if self.eligible and self.quality_flags:
            raise ValueError("eligible task examples cannot carry quality flags")
        if not self.eligible and not self.quality_flags:
            raise ValueError("ineligible task examples must explain their quality failure")
        if len(self.replay_feature_values) != len(self.replay_window_end_s):
            raise ValueError("replay features and causal window times must align")
        if self.replay_feature_values:
            if len(self.replay_feature_values) != 17 or self.event_start_s is None:
                raise ValueError("replay evidence requires 17 windows and an event onset")
            if any(
                any(value is None for value in row)
                for row in self.replay_feature_values
            ):
                raise ValueError("canonical replay evidence cannot impute missing features")
            if any(
                right <= left
                for left, right in zip(
                    self.replay_window_end_s,
                    self.replay_window_end_s[1:],
                )
            ):
                raise ValueError("replay causal window times must be strictly increasing")
        elif self.event_start_s is not None:
            raise ValueError("event onset cannot be attached without replay evidence")
        return self


class DemoTaskPartition(FrozenModel):
    """Predictive features joined to a separately loaded supervisory channel."""

    schema_version: Literal["aqse.network-demo.task-partition.v1"] = (
        "aqse.network-demo.task-partition.v1"
    )
    study_artifact_id: str = Field(min_length=1, max_length=256)
    study_content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    partition: TaskPartitionName
    task_id: TaskId
    profile_id: Literal["aqse.local-state8.v1", "aqse.network-state8.v1"]
    profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    class_order: tuple[str, ...] = Field(min_length=2)
    examples: tuple[DemoTaskExample, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_task_contract(self) -> DemoTaskPartition:
        expected_profile = (
            "aqse.local-state8.v1"
            if self.task_id == LOCAL_TASK_ID
            else "aqse.network-state8.v1"
        )
        expected_classes = (
            ("NORMAL", "CHANGE_DETECTED")
            if self.task_id == LOCAL_TASK_ID
            else (
                "NORMAL",
                "ENVIRONMENT_COMPATIBLE",
                "DEVICE_COMPATIBLE",
                "MIXED_OR_AMBIGUOUS",
            )
        )
        if self.profile_id != expected_profile or self.class_order != expected_classes:
            raise ValueError("task, state8 profile and class order are incompatible")
        if len({item.sample_id for item in self.examples}) != len(self.examples):
            raise ValueError("task sample identifiers must be distinct")
        if len({item.episode_id for item in self.examples}) != len(self.examples):
            raise ValueError("task reporting units must be independent episodes")
        if len({item.generative_lineage_id for item in self.examples}) != len(self.examples):
            raise ValueError("task examples must use independent generative lineages")
        if any(item.label not in self.class_order for item in self.examples):
            raise ValueError("task example contains an incompatible class")
        if self.task_id == NETWORK_TASK_ID and any(
            item.node_count < 3 for item in self.examples
        ):
            raise ValueError("network-task examples require at least three nodes")
        return self


class DemoThetaCandidate(FrozenModel):
    candidate_name: ThetaCandidateName
    candidate_id: str = Field(min_length=1, max_length=256)
    eligible: bool
    theta: tuple[float, ...] = Field(min_length=16, max_length=16)
    qng_bank_sample_ids: tuple[str, ...] = ()
    qng_bank_lineage_ids: tuple[str, ...] = ()
    accepted_updates: int = Field(ge=0, le=10)
    stop_reason: str = Field(min_length=1, max_length=256)
    initial_alignment_loss: float | None = Field(default=None, ge=0.0)
    final_alignment_loss: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_candidate(self) -> DemoThetaCandidate:
        if self.candidate_name == "theta0":
            if not self.eligible or self.accepted_updates != 0:
                raise ValueError("theta0 must be an eligible zero-update candidate")
            if self.qng_bank_sample_ids or self.qng_bank_lineage_ids:
                raise ValueError("theta0 cannot claim a QNG optimization bank")
        if len(self.qng_bank_sample_ids) != len(self.qng_bank_lineage_ids):
            raise ValueError("QNG bank sample and lineage identities must align")
        if len(set(self.qng_bank_lineage_ids)) != len(self.qng_bank_lineage_ids):
            raise ValueError("QNG bank lineages must be distinct")
        if not self.eligible and self.accepted_updates:
            raise ValueError("an ineligible QNG candidate cannot claim accepted updates")
        return self


class BootstrapInterval(FrozenModel):
    confidence_level: Literal[0.95] = 0.95
    lower: float = Field(ge=-1.0, le=1.0)
    upper: float = Field(ge=-1.0, le=1.0)
    replicates: int = Field(gt=0, le=10_000)
    seed: Literal[2001006] = 2_001_006
    resampling_unit: Literal["independent_episode"] = "independent_episode"
    degenerate: bool

    @model_validator(mode="after")
    def validate_bounds(self) -> BootstrapInterval:
        if self.lower > self.upper:
            raise ValueError("bootstrap interval bounds are reversed")
        if self.degenerate != (self.lower == self.upper):
            raise ValueError("bootstrap degeneracy flag differs from its bounds")
        return self


class DemoNodeCountMetrics(FrozenModel):
    node_count: int = Field(ge=1, le=8)
    sample_count: int = Field(gt=0)
    eligible_count: int = Field(ge=0)
    class_support: dict[str, int]
    class_recall: dict[str, float | None]
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    abstention_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    heuristic_ood_count: int = Field(ge=0)


class DemoReplayNodeCountMetrics(FrozenModel):
    node_count: int = Field(ge=1, le=8)
    episode_count: int = Field(gt=0)
    normal_episode_count: int = Field(ge=0)
    normal_window_count: int = Field(ge=0)
    false_positive_episode_count: int = Field(ge=0)
    false_positive_window_count: int = Field(ge=0)
    changed_episode_count: int = Field(ge=0)
    detected_episode_count: int = Field(ge=0)
    censored_episode_count: int = Field(ge=0)
    detection_delay_p50_s: float | None = Field(default=None, ge=0.0)
    detection_delay_p95_s: float | None = Field(default=None, ge=0.0)


class DemoReplayMetrics(FrozenModel):
    trace_policy: Literal[
        "17-causal-windows;normal-fp;first-post-onset-operational-change"
    ] = "17-causal-windows;normal-fp;first-post-onset-operational-change"
    episode_count: int = Field(gt=0)
    window_count: int = Field(gt=0)
    normal_episode_count: int = Field(ge=0)
    normal_window_count: int = Field(ge=0)
    false_positive_episode_count: int = Field(ge=0)
    false_positive_window_count: int = Field(ge=0)
    false_positive_episode_rate: float = Field(ge=0.0, le=1.0)
    false_positive_window_rate: float = Field(ge=0.0, le=1.0)
    changed_episode_count: int = Field(ge=0)
    detected_episode_count: int = Field(ge=0)
    censored_episode_count: int = Field(ge=0)
    detection_delays_s: tuple[float, ...]
    detection_delay_p50_s: float | None = Field(default=None, ge=0.0)
    detection_delay_p95_s: float | None = Field(default=None, ge=0.0)
    by_node_count: tuple[DemoReplayNodeCountMetrics, ...]

    @model_validator(mode="after")
    def validate_replay_accounting(self) -> DemoReplayMetrics:
        if self.window_count != self.episode_count * 17:
            raise ValueError("replay window count must use 17 windows per episode")
        if self.detected_episode_count + self.censored_episode_count != self.changed_episode_count:
            raise ValueError("replay detection accounting is incomplete")
        if len(self.detection_delays_s) != self.detected_episode_count:
            raise ValueError("replay delay evidence differs from detected episodes")
        return self


class DemoPairedModelComparison(FrozenModel):
    partition: Literal["validation", "test"]
    task_id: TaskId
    sample_count: int = Field(gt=0)
    comparison: Literal["quantum-afse-vs-same-architecture-raw-state8"] = (
        "quantum-afse-vs-same-architecture-raw-state8"
    )
    tie_tolerance: Literal[1e-12] = 1.0e-12
    afse_balanced_accuracy: float = Field(ge=0.0, le=1.0)
    raw_balanced_accuracy: float = Field(ge=0.0, le=1.0)
    balanced_accuracy_delta: float = Field(ge=-1.0, le=1.0)
    balanced_accuracy_delta_interval: BootstrapInterval
    afse_macro_f1: float = Field(ge=0.0, le=1.0)
    raw_macro_f1: float = Field(ge=0.0, le=1.0)
    macro_f1_delta: float = Field(ge=-1.0, le=1.0)
    macro_f1_delta_interval: BootstrapInterval
    outcome: Literal["HELPED", "TIED", "HURT"]


class DemoEvaluationMetrics(FrozenModel):
    schema_version: Literal["aqse.network-demo.metrics.v2"] = (
        "aqse.network-demo.metrics.v2"
    )
    partition: Literal["validation", "test"]
    task_id: TaskId
    profile_id: Literal["aqse.local-state8.v1", "aqse.network-state8.v1"]
    sample_count: int = Field(gt=0)
    eligible_count: int = Field(ge=0)
    class_order: tuple[str, ...] = Field(min_length=2)
    class_support: dict[str, int]
    class_recall: dict[str, float | None]
    raw_confusion_columns: tuple[str, ...]
    raw_confusion_matrix: tuple[tuple[int, ...], ...]
    operational_confusion_columns: tuple[str, ...]
    operational_confusion_matrix: tuple[tuple[int, ...], ...]
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    abstention_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    heuristic_ood_count: int = Field(ge=0)
    quality_rejected_count: int = Field(ge=0)
    balanced_accuracy_interval: BootstrapInterval
    macro_f1_interval: BootstrapInterval
    predictions: tuple[str, ...]
    displayed_predictions: tuple[str, ...]
    episode_ids: tuple[str, ...]
    by_node_count: tuple[DemoNodeCountMetrics, ...]
    replay: DemoReplayMetrics | None = None

    @model_validator(mode="after")
    def validate_metrics(self) -> DemoEvaluationMetrics:
        size = self.sample_count
        if any(
            len(values) != size
            for values in (self.predictions, self.displayed_predictions, self.episode_ids)
        ):
            raise ValueError("metric row evidence must match the sample count")
        if len(set(self.episode_ids)) != size:
            raise ValueError("metrics must report independent episode units")
        if set(self.class_support) != set(self.class_order):
            raise ValueError("metric support keys must equal the frozen class order")
        if set(self.class_recall) != set(self.class_order):
            raise ValueError("metric recall keys must equal the frozen class order")
        if sum(self.class_support.values()) != size:
            raise ValueError("metric class support must account for every example")
        if len(self.raw_confusion_matrix) != len(self.class_order):
            raise ValueError("raw confusion rows must match the true class order")
        if len(self.operational_confusion_matrix) != len(self.class_order):
            raise ValueError("operational confusion rows must match the true class order")
        if any(len(row) != len(self.raw_confusion_columns) for row in self.raw_confusion_matrix):
            raise ValueError("raw confusion matrix has an invalid shape")
        if any(
            len(row) != len(self.operational_confusion_columns)
            for row in self.operational_confusion_matrix
        ):
            raise ValueError("operational confusion matrix has an invalid shape")
        if self.abstention_count != sum(
            value == ABSTAIN_CLASS for value in self.displayed_predictions
        ):
            raise ValueError("abstention count differs from displayed predictions")
        expected_profile = (
            "aqse.local-state8.v1"
            if self.task_id == LOCAL_TASK_ID
            else "aqse.network-state8.v1"
        )
        if self.profile_id != expected_profile:
            raise ValueError("metric task and feature profile are incompatible")
        if sum(item.sample_count for item in self.by_node_count) != size:
            raise ValueError("node-count strata must account for every episode")
        if len({item.node_count for item in self.by_node_count}) != len(
            self.by_node_count
        ):
            raise ValueError("node-count metric strata must be unique")
        if self.replay is not None and self.replay.episode_count != size:
            raise ValueError("replay metrics must use the same independent episodes")
        return self


class DemoResearchBundle(FrozenModel):
    """Cohesive non-executable deployment artifact for one state8 task."""

    schema_version: Literal["aqse.network-demo.bundle.v2"] = (
        "aqse.network-demo.bundle.v2"
    )
    bundle_id: str = Field(pattern=r"^aqse-demo-bundle-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    scientific_label: Literal["research / not validated for field deployment"] = (
        "research / not validated for field deployment"
    )
    study_artifact_id: str
    study_content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    protocol_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    task_id: TaskId
    profile_id: Literal["aqse.local-state8.v1", "aqse.network-state8.v1"]
    profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    fitted_on_partition: Literal["TRAIN"] = "TRAIN"
    fitted_on_dataset_id: str
    fitted_on_dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    class_order: tuple[str, ...]
    theta_candidate: DemoThetaCandidate
    encoder: State8EncodingArtifact
    afse: NystromAFSEArtifact
    classifier: MLPClassifierArtifact
    raw_feature_baseline: MLPClassifierArtifact
    observable_rule: ObservableRuleArtifact
    validation_metrics: DemoEvaluationMetrics
    raw_baseline_validation_metrics: DemoEvaluationMetrics
    validation_comparison: DemoPairedModelComparison
    effective_source_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_component_binding(self) -> DemoResearchBundle:
        validate_state8_encoder_artifact(self.encoder)
        validate_afse_artifact(self.afse)
        from app.classical.mlp import validate_mlp_artifact

        validate_mlp_artifact(self.classifier)
        validate_mlp_artifact(self.raw_feature_baseline)
        validate_observable_rule_artifact(self.observable_rule)
        expected_profile = (
            "aqse.local-state8.v1"
            if self.task_id == LOCAL_TASK_ID
            else "aqse.network-state8.v1"
        )
        if self.profile_id != expected_profile or self.encoder.profile_id != self.profile_id:
            raise ValueError("bundle task and encoder profile are incompatible")
        if self.encoder.profile_fingerprint != self.profile_fingerprint:
            raise ValueError("bundle encoder fingerprint is incompatible")
        if (
            self.afse.feature_profile_id != self.profile_id
            or self.afse.scaler_id != self.encoder.scaler_id
            or self.afse.encoding_policy_id != self.encoder.encoding_policy_id
        ):
            raise ValueError("bundle AFSE is incompatible with the fitted encoder")
        if tuple(self.afse.theta) != self.theta_candidate.theta:
            raise ValueError("bundle theta candidate and AFSE theta differ")
        if (
            self.classifier.model_role != "afse_classifier"
            or self.classifier.input_space_id != self.afse.artifact_id
            or self.classifier.feature_count != self.afse.output_dimension
        ):
            raise ValueError("bundle classifier is incompatible with the AFSE space")
        if (
            self.raw_feature_baseline.model_role != "raw_feature_baseline"
            or self.raw_feature_baseline.input_space_id != f"{self.encoder.encoder_id}:raw"
            or self.raw_feature_baseline.feature_count != 8
        ):
            raise ValueError("bundle raw baseline is incompatible with the encoder")
        if (
            self.observable_rule.profile_id != self.profile_id
            or self.observable_rule.profile_fingerprint != self.profile_fingerprint
            or self.observable_rule.fitted_on_dataset_id
            != self.fitted_on_dataset_id
            or self.observable_rule.fitted_on_dataset_digest
            != self.fitted_on_dataset_digest
        ):
            raise ValueError("bundle observable rule is incompatible with its fitted space")
        fitted_components = (self.afse, self.classifier, self.raw_feature_baseline)
        if any(
            item.fitted_on_dataset_id != self.fitted_on_dataset_id
            or item.fitted_on_dataset_digest != self.fitted_on_dataset_digest
            for item in fitted_components
        ):
            raise ValueError("bundle components use different fitted datasets")
        if self.classifier.task_id != self.task_id or self.raw_feature_baseline.task_id != self.task_id:
            raise ValueError("bundle classifiers use a different task")
        if set(self.classifier.classes) != set(self.class_order):
            raise ValueError("bundle classifier class order differs from the bundle")
        if set(self.raw_feature_baseline.classes) != set(self.class_order):
            raise ValueError("bundle raw baseline class order differs from the bundle")
        if (
            self.validation_metrics.partition != "validation"
            or self.validation_metrics.task_id != self.task_id
            or self.raw_baseline_validation_metrics.partition != "validation"
            or self.raw_baseline_validation_metrics.task_id != self.task_id
            or self.validation_comparison.partition != "validation"
            or self.validation_comparison.task_id != self.task_id
        ):
            raise ValueError("bundle validation evidence is incompatible")
        if any(not _is_sha256(value) for value in self.effective_source_hashes.values()):
            raise ValueError("bundle source provenance contains an invalid SHA-256")
        payload = self.model_dump(mode="json", exclude={"bundle_id", "content_digest"})
        expected_digest = canonical_digest(payload)
        if self.content_digest != expected_digest:
            raise ValueError("bundle scientific content digest is invalid")
        if self.bundle_id != f"aqse-demo-bundle-{expected_digest[:16]}":
            raise ValueError("bundle identity is invalid")
        return self


class CandidateValidationEvidence(FrozenModel):
    candidate_name: ThetaCandidateName
    bundle_id: str = Field(pattern=r"^aqse-demo-bundle-[a-f0-9]{16}$")
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)


class TaskSelection(FrozenModel):
    task_id: TaskId
    candidate_evidence: tuple[CandidateValidationEvidence, ...] = Field(min_length=1)
    selected_bundle_id: str = Field(pattern=r"^aqse-demo-bundle-[a-f0-9]{16}$")
    selected_candidate_name: ThetaCandidateName
    selection_rule: tuple[Literal[
        "maximum_validation_balanced_accuracy",
        "maximum_validation_macro_f1",
        "prefer_theta0_on_tie",
    ], ...] = (
        "maximum_validation_balanced_accuracy",
        "maximum_validation_macro_f1",
        "prefer_theta0_on_tie",
    )

    @model_validator(mode="after")
    def validate_selected_candidate(self) -> TaskSelection:
        if self.selection_rule != (
            "maximum_validation_balanced_accuracy",
            "maximum_validation_macro_f1",
            "prefer_theta0_on_tie",
        ):
            raise ValueError("task selection rule differs from the frozen protocol")
        selected = tuple(
            item
            for item in self.candidate_evidence
            if item.bundle_id == self.selected_bundle_id
        )
        if len(selected) != 1 or selected[0].candidate_name != self.selected_candidate_name:
            raise ValueError("selected candidate is absent from validation evidence")
        if len({item.candidate_name for item in self.candidate_evidence}) != len(
            self.candidate_evidence
        ):
            raise ValueError("candidate validation evidence contains duplicates")
        if not any(
            item.candidate_name == "theta0" for item in self.candidate_evidence
        ):
            raise ValueError("candidate validation evidence must retain theta0")
        expected = max(
            self.candidate_evidence,
            key=lambda item: (
                item.balanced_accuracy,
                item.macro_f1,
                item.candidate_name == "theta0",
            ),
        )
        if expected.bundle_id != self.selected_bundle_id:
            raise ValueError("selected candidate violates the frozen selection rule")
        return self


class DemoModelSelectionFreeze(FrozenModel):
    schema_version: Literal["aqse.network-demo.selection-freeze.v1"] = (
        "aqse.network-demo.selection-freeze.v1"
    )
    freeze_id: str = Field(pattern=r"^aqse-demo-freeze-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    study_artifact_id: str
    study_content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    protocol_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    selections: tuple[TaskSelection, TaskSelection]
    test_state: Literal["sealed"] = "sealed"
    final_evaluation_rule: Literal[
        "single-new-test-evaluation-after-selection-freeze;no-refit-or-reselection"
    ] = "single-new-test-evaluation-after-selection-freeze;no-refit-or-reselection"

    @model_validator(mode="after")
    def validate_freeze(self) -> DemoModelSelectionFreeze:
        if tuple(item.task_id for item in self.selections) != (
            LOCAL_TASK_ID,
            NETWORK_TASK_ID,
        ):
            raise ValueError("selection freeze must contain ordered local and network tasks")
        payload = self.model_dump(mode="json", exclude={"freeze_id", "content_digest"})
        expected_digest = canonical_digest(payload)
        if self.content_digest != expected_digest:
            raise ValueError("selection-freeze content digest is invalid")
        if self.freeze_id != f"aqse-demo-freeze-{expected_digest[:16]}":
            raise ValueError("selection-freeze identity is invalid")
        return self


class DemoFinalEvaluation(FrozenModel):
    schema_version: Literal["aqse.network-demo.final-evaluation.v2"] = (
        "aqse.network-demo.final-evaluation.v2"
    )
    evaluation_id: str = Field(pattern=r"^aqse-demo-final-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    freeze_id: str = Field(pattern=r"^aqse-demo-freeze-[a-f0-9]{16}$")
    study_artifact_id: str
    study_content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    query_acquisition_id: str = Field(min_length=1, max_length=256)
    test_access: Literal["single-authorized-access"] = "single-authorized-access"
    selection_changed: Literal[False] = False
    refit_performed: Literal[False] = False
    task_metrics: tuple[DemoEvaluationMetrics, DemoEvaluationMetrics]
    raw_baseline_task_metrics: tuple[DemoEvaluationMetrics, DemoEvaluationMetrics]
    paired_comparisons: tuple[
        DemoPairedModelComparison,
        DemoPairedModelComparison,
    ]

    @model_validator(mode="after")
    def validate_final_evaluation(self) -> DemoFinalEvaluation:
        if tuple(item.task_id for item in self.task_metrics) != (
            LOCAL_TASK_ID,
            NETWORK_TASK_ID,
        ) or any(item.partition != "test" for item in self.task_metrics):
            raise ValueError("final evaluation must contain ordered local/network TEST metrics")
        if tuple(item.task_id for item in self.raw_baseline_task_metrics) != (
            LOCAL_TASK_ID,
            NETWORK_TASK_ID,
        ) or any(
            item.partition != "test" for item in self.raw_baseline_task_metrics
        ):
            raise ValueError("final evaluation raw baselines must be ordered TEST metrics")
        if tuple(item.task_id for item in self.paired_comparisons) != (
            LOCAL_TASK_ID,
            NETWORK_TASK_ID,
        ) or any(item.partition != "test" for item in self.paired_comparisons):
            raise ValueError("final evaluation comparisons must be ordered TEST results")
        payload = self.model_dump(mode="json", exclude={"evaluation_id", "content_digest"})
        expected_digest = canonical_digest(payload)
        if self.content_digest != expected_digest:
            raise ValueError("final-evaluation content digest is invalid")
        if self.evaluation_id != f"aqse-demo-final-{expected_digest[:16]}":
            raise ValueError("final-evaluation identity is invalid")
        return self


class BundleQueryContext(FrozenModel):
    """Runtime compatibility context; acquisition identity is not fit provenance."""

    bundle_id: str = Field(pattern=r"^aqse-demo-bundle-[a-f0-9]{16}$")
    task_id: TaskId
    profile_id: Literal["aqse.local-state8.v1", "aqse.network-state8.v1"]
    profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    query_acquisition_id: str = Field(min_length=1, max_length=256)


class BundlePredictionBatch(FrozenModel):
    bundle_id: str
    task_id: TaskId
    query_acquisition_id: str
    sample_ids: tuple[str, ...]
    afse_vectors: tuple[tuple[float, ...], ...]
    reconstruction_residuals: tuple[float, ...]
    heuristic_ood: tuple[bool, ...]
    model_scores: MLPScoreBatch
    displayed_classes: tuple[str, ...]
    raw_baseline_scores: MLPScoreBatch
    observable_rule_statuses: tuple[ObservableRuleStatus, ...]


class ActiveBundlePointer(FrozenModel):
    schema_version: Literal["aqse.network-demo.active-bundles.v1"] = (
        "aqse.network-demo.active-bundles.v1"
    )
    application_id: str = Field(pattern=r"^aqse-demo-application-[a-f0-9]{16}$")
    generation: int = Field(gt=0)
    revision: int = Field(gt=0)
    selection_freeze_id: str = Field(pattern=r"^aqse-demo-freeze-[a-f0-9]{16}$")
    local_bundle_id: str = Field(pattern=r"^aqse-demo-bundle-[a-f0-9]{16}$")
    network_bundle_id: str = Field(pattern=r"^aqse-demo-bundle-[a-f0-9]{16}$")
    previous_application_id: str | None = None

    @model_validator(mode="after")
    def validate_application_identity(self) -> ActiveBundlePointer:
        payload = self.model_dump(mode="json", exclude={"application_id"})
        expected = canonical_digest(payload)
        if self.application_id != f"aqse-demo-application-{expected[:16]}":
            raise ValueError("active-bundle application identity is invalid")
        if self.generation != self.revision:
            raise ValueError("bundle generation and revision must advance atomically")
        return self


@dataclass(frozen=True)
class BundleRuntime:
    artifact: DemoResearchBundle
    encoder: State8AngleEncoder
    afse: NystromAFSE
    classifier: NumpyMLPClassifier
    raw_baseline: NumpyMLPClassifier
    observable_rule: ObservableRule

    @classmethod
    def from_artifact(cls, artifact: DemoResearchBundle) -> BundleRuntime:
        # Re-validating the top-level artifact also verifies every cross-component identity.
        DemoResearchBundle.model_validate(artifact.model_dump(mode="json"))
        return cls(
            artifact=artifact,
            encoder=load_state8_encoder(artifact.encoder),
            afse=NystromAFSE.from_artifact(artifact.afse),
            classifier=NumpyMLPClassifier.from_artifact(artifact.classifier),
            raw_baseline=NumpyMLPClassifier.from_artifact(
                artifact.raw_feature_baseline
            ),
            observable_rule=ObservableRule.from_artifact(artifact.observable_rule),
        )

    def predict(
        self,
        raw_features: NDArray[Any],
        *,
        sample_ids: tuple[str, ...],
        context: BundleQueryContext,
    ) -> BundlePredictionBatch:
        if context != BundleQueryContext(
            bundle_id=self.artifact.bundle_id,
            task_id=self.artifact.task_id,
            profile_id=self.artifact.profile_id,
            profile_fingerprint=self.artifact.profile_fingerprint,
            query_acquisition_id=context.query_acquisition_id,
        ):
            raise ValueError("query is incompatible with the active research bundle")
        profile = State8FeatureProfile.model_validate(self.artifact.encoder.profile)
        encoded = self.encoder.transform(raw_features, profile=profile)
        embedding = self.afse.transform(
            encoded,
            sample_ids=sample_ids,
            context=afse_context_for(self.artifact.afse),
        )
        latent = np.asarray(embedding.vectors, dtype=np.float64)
        scores = self.classifier.score(
            latent,
            sample_ids=sample_ids,
            context=mlp_context_for(self.artifact.classifier),
        )
        raw_scores = self.raw_baseline.score(
            raw_features,
            sample_ids=sample_ids,
            context=mlp_context_for(self.artifact.raw_feature_baseline),
        )
        displayed = tuple(
            ABSTAIN_CLASS if uncertain or ood else predicted
            for predicted, uncertain, ood in zip(
                scores.predicted_classes,
                scores.uncertain,
                embedding.heuristic_ood,
                strict=True,
            )
        )
        rule_statuses = tuple(
            self.observable_rule.evaluate(
                observable_rule_query(
                    self.artifact.observable_rule,
                    sample_id=sample_id,
                    feature_values=raw_features[index],
                )
            ).status
            for index, sample_id in enumerate(sample_ids)
        )
        return BundlePredictionBatch(
            bundle_id=self.artifact.bundle_id,
            task_id=self.artifact.task_id,
            query_acquisition_id=context.query_acquisition_id,
            sample_ids=sample_ids,
            afse_vectors=tuple(tuple(float(value) for value in row) for row in latent),
            reconstruction_residuals=embedding.reconstruction_residuals,
            heuristic_ood=embedding.heuristic_ood,
            model_scores=scores,
            displayed_classes=displayed,
            raw_baseline_scores=raw_scores,
            observable_rule_statuses=rule_statuses,
        )
