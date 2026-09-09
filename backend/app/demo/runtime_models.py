from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from app.demo.bundle_models import ActiveBundlePointer, TaskId
from app.features.state8_models import State8ProfileId, State8Values
from app.training.models import FrozenModel

AnalysisState = Literal[
    "awaiting_reference",
    "running",
    "paused_for_training",
    "stopped",
    "failed",
    "bundle_unavailable",
]
ContextMode = Literal[
    "local",
    "local_two_node_ambiguous",
    "network",
    "degraded_local",
]


class DemoAnalysisResult(FrozenModel):
    """One observation-only inference result for a causal node/window."""

    schema_version: Literal["aqse.demo-analysis-result.v1"] = (
        "aqse.demo-analysis-result.v1"
    )
    result_id: str = Field(pattern=r"^aqse-demo-result-[a-f0-9]{16}$")
    produced_at_utc: str
    session_id: str
    application_id: str = Field(pattern=r"^aqse-demo-application-[a-f0-9]{16}$")
    bundle_id: str = Field(pattern=r"^aqse-demo-bundle-[a-f0-9]{16}$")
    task_id: TaskId
    profile_id: State8ProfileId
    profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    theta_id: str = Field(pattern=r"^aqse-theta-[a-f0-9]{16}$")
    reference_id: str = Field(pattern=r"^aqse-state8-reference-[a-f0-9]{16}$")
    node_count: int = Field(ge=1, le=8)
    sensor_id: str
    peer_sensor_ids: tuple[str, ...]
    context_mode: ContextMode
    attribution_note: str
    window_id: str
    window_start_s: float
    window_end_exclusive_s: float
    source_frame_ids: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    feature_values: State8Values
    feature_valid: bool
    quality_flags: tuple[str, ...]
    encoded_angles: tuple[float, ...] | None
    afse_vector: tuple[float, ...] | None
    afse_reference_size: int | None = Field(default=None, gt=0, le=32)
    reconstruction_residual: float | None = Field(default=None, ge=0.0)
    heuristic_ood: bool | None
    class_order: tuple[str, ...]
    class_scores: tuple[float, ...] | None
    predicted_class: str | None
    displayed_class: str
    uncertain: bool | None
    top_score: float | None = Field(default=None, ge=0.0, le=1.0)
    top_two_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_baseline_scores: tuple[float, ...] | None
    raw_baseline_class: str | None
    score_semantics: Literal["model-score;not-probability-calibrated"] = (
        "model-score;not-probability-calibrated"
    )
    data_age_ms: float = Field(ge=0.0)
    processing_duration_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_result_payload(self) -> DemoAnalysisResult:
        if self.window_end_exclusive_s <= self.window_start_s:
            raise ValueError("analysis window end must follow its start")
        if len(self.feature_names) != 8 or len(self.feature_units) != 8:
            raise ValueError("demo analysis requires the complete state8 schema")
        inference_fields = (
            self.encoded_angles,
            self.afse_vector,
            self.reconstruction_residual,
            self.heuristic_ood,
            self.class_scores,
            self.predicted_class,
            self.uncertain,
            self.top_score,
            self.top_two_margin,
            self.raw_baseline_scores,
            self.raw_baseline_class,
        )
        if self.feature_valid:
            if any(value is None for value in inference_fields):
                raise ValueError("eligible analysis results require complete inference data")
            if self.quality_flags:
                raise ValueError("eligible analysis results cannot carry quality flags")
            if len(self.encoded_angles or ()) != 8:
                raise ValueError("eligible analysis must contain eight encoded angles")
            if len(self.afse_vector or ()) != self.afse_reference_size:
                raise ValueError("AFSE vector dimension differs from its reference")
            if len(self.class_scores or ()) != len(self.class_order):
                raise ValueError("model score vector differs from its class order")
            if len(self.raw_baseline_scores or ()) != len(self.class_order):
                raise ValueError("raw baseline score vector differs from class order")
        else:
            if any(value is not None for value in inference_fields):
                raise ValueError("abstaining on input quality cannot expose inference data")
            if self.afse_reference_size is not None:
                raise ValueError("abstaining on input quality cannot claim AFSE output")
            if not self.quality_flags or self.displayed_class != "ABSTAIN":
                raise ValueError("invalid input must explain and display abstention")
        return self


class DemoAnalysisView(FrozenModel):
    schema_version: Literal["aqse.demo-analysis-view.v1"] = (
        "aqse.demo-analysis-view.v1"
    )
    session_id: str
    state: AnalysisState
    state_detail: str
    worker_epoch: int = Field(gt=0)
    application_id: str | None = None
    active_local_bundle_id: str | None = None
    active_network_bundle_id: str | None = None
    reference_id: str | None = None
    reference_progress: float = Field(ge=0.0, le=1.0)
    latest_observation_frame_id: int = Field(ge=0)
    latest_observation_time_s: float = Field(ge=0.0)
    latest_analyzed_window_start_s: float | None = Field(default=None, ge=0.0)
    completed_window_count: int = Field(ge=0)
    skipped_window_count: int = Field(ge=0)
    quality_abstention_count: int = Field(ge=0)
    queue_depth: Literal[0, 1]
    latency_p50_ms: float | None = Field(default=None, ge=0.0)
    latency_p95_ms: float | None = Field(default=None, ge=0.0)
    result_age_ms: float | None = Field(default=None, ge=0.0)
    latest_results: tuple[DemoAnalysisResult, ...]
    error: str | None = None


class DemoAnalysisStart(FrozenModel):
    acquire_reference: Literal[True] = True


class DemoBundleApplicationRequest(FrozenModel):
    local_bundle_id: str = Field(pattern=r"^aqse-demo-bundle-[a-f0-9]{16}$")
    network_bundle_id: str = Field(pattern=r"^aqse-demo-bundle-[a-f0-9]{16}$")
    selection_freeze_id: str = Field(pattern=r"^aqse-demo-freeze-[a-f0-9]{16}$")


class DemoMetricSummary(FrozenModel):
    partition: Literal["validation", "test"]
    task_id: TaskId
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(gt=0)
    uncertain_count: int = Field(ge=0)
    heuristic_ood_count: int = Field(ge=0)


class DemoBundleSummary(FrozenModel):
    bundle_id: str = Field(pattern=r"^aqse-demo-bundle-[a-f0-9]{16}$")
    task_id: TaskId
    profile_id: State8ProfileId
    theta_candidate_name: Literal["theta0", "protected_qng"]
    theta_id: str = Field(pattern=r"^aqse-theta-[a-f0-9]{16}$")
    accepted_qng_updates: int = Field(ge=0, le=10)
    afse_method_id: str
    afse_reference_size: int = Field(gt=0, le=32)
    afse_ridge_lambda: float = Field(gt=0.0)
    classifier_model_id: str
    class_order: tuple[str, ...]
    validation: DemoMetricSummary
    raw_baseline_validation: DemoMetricSummary
    active: bool


class DemoRegistryView(FrozenModel):
    schema_version: Literal["aqse.demo-registry-view.v1"] = (
        "aqse.demo-registry-view.v1"
    )
    prepared: bool
    scientific_label: Literal["research / not validated for field deployment"] = (
        "research / not validated for field deployment"
    )
    study_artifact_id: str | None
    study_content_digest: str | None
    selection_freeze_id: str | None
    final_evaluation_id: str | None
    historical_test_ledger_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    active: ActiveBundlePointer | None
    bundles: tuple[DemoBundleSummary, ...]
    final_metrics: tuple[DemoMetricSummary, ...]
    preparation_detail: str


class DemoTrainingIntent(FrozenModel):
    schema_version: Literal["aqse.demo-training-intent.v1"] = (
        "aqse.demo-training-intent.v1"
    )
    intent_id: str = Field(pattern=r"^aqse-demo-intent-[A-Za-z0-9._:-]{1,128}$")
    study_artifact_id: str
    requested_action: Literal["fit-two-candidate-local-and-network-bundles"] = (
        "fit-two-candidate-local-and-network-bundles"
    )


class DemoTrainingStepView(FrozenModel):
    task_id: TaskId
    candidate_name: Literal["theta0", "protected_qng"]
    accepted_updates: int = Field(ge=0, le=10)
    current_loss: float | None = Field(default=None, ge=0.0)


class DemoTrainingJobView(FrozenModel):
    schema_version: Literal["aqse.demo-training-job.v1"] = (
        "aqse.demo-training-job.v1"
    )
    job_id: str = Field(pattern=r"^aqse-demo-training-[a-f0-9]{16}$")
    intent_id: str
    study_artifact_id: str
    state: Literal["created", "running", "completed", "cancelled", "failed"]
    created_at_utc: str
    updated_at_utc: str
    current_stage: str
    progress: tuple[DemoTrainingStepView, ...]
    selection_freeze_id: str | None = None
    selected_local_bundle_id: str | None = None
    selected_network_bundle_id: str | None = None
    error: str | None = None
