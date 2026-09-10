export type DemoTaskId = "aqse.local-change.v1" | "aqse.network-pattern.v1";
export type State8ProfileId = "aqse.local-state8.v1" | "aqse.network-state8.v1";

export type DemoNodeCountSummary = {
  node_count: number;
  sample_count: number;
  eligible_count: number;
  class_support: Record<string, number>;
  balanced_accuracy: number;
  macro_f1: number;
  coverage: number;
};

export type DemoReplaySummary = {
  episode_count: number;
  window_count: number;
  false_positive_episode_rate: number;
  false_positive_window_rate: number;
  changed_episode_count: number;
  detected_episode_count: number;
  censored_episode_count: number;
  detection_delay_p50_s: number | null;
  detection_delay_p95_s: number | null;
};

export type DemoBootstrapInterval = {
  confidence_level: 0.95;
  lower: number;
  upper: number;
  replicates: number;
  seed: 2001006;
  resampling_unit: "independent_episode";
  degenerate: boolean;
};

export type DemoPairedModelComparison = {
  partition: "validation" | "test";
  task_id: DemoTaskId;
  sample_count: number;
  comparison: "quantum-afse-vs-same-architecture-raw-state8";
  tie_tolerance: 1e-12;
  afse_balanced_accuracy: number;
  raw_balanced_accuracy: number;
  balanced_accuracy_delta: number;
  balanced_accuracy_delta_interval: DemoBootstrapInterval;
  afse_macro_f1: number;
  raw_macro_f1: number;
  macro_f1_delta: number;
  macro_f1_delta_interval: DemoBootstrapInterval;
  outcome: "HELPED" | "TIED" | "HURT";
};

export type DemoMetricSummary = {
  partition: "validation" | "test";
  task_id: DemoTaskId;
  profile_id: State8ProfileId;
  balanced_accuracy: number;
  macro_f1: number;
  coverage: number;
  sample_count: number;
  eligible_count: number;
  class_support: Record<string, number>;
  class_recall: Record<string, number | null>;
  uncertain_count: number;
  heuristic_ood_count: number;
  by_node_count: DemoNodeCountSummary[];
  replay: DemoReplaySummary | null;
};

export type DemoBundleSummary = {
  bundle_id: string;
  task_id: DemoTaskId;
  profile_id: State8ProfileId;
  theta_candidate_name: "theta0" | "protected_qng";
  theta_id: string;
  accepted_qng_updates: number;
  afse_method_id: string;
  afse_reference_size: number;
  afse_ridge_lambda: number;
  classifier_model_id: string;
  class_order: string[];
  validation: DemoMetricSummary;
  raw_baseline_validation: DemoMetricSummary;
  validation_comparison: DemoPairedModelComparison;
  active: boolean;
};

export type ActiveBundlePointer = {
  application_id: string;
  generation: number;
  revision: number;
  selection_freeze_id: string;
  local_bundle_id: string;
  network_bundle_id: string;
  previous_application_id: string | null;
};

export type DemoSelectionFreezeSummary = {
  freeze_id: string;
  study_artifact_id: string;
  study_content_digest: string;
  local_bundle_id: string;
  network_bundle_id: string;
};

export type DemoRegistryView = {
  schema_version: "aqse.demo-registry-view.v2";
  prepared: boolean;
  scientific_label: "research / not validated for field deployment";
  study_artifact_id: string | null;
  study_content_digest: string | null;
  selection_freeze_id: string | null;
  final_evaluation_id: string | null;
  historical_test_ledger_sha256: string;
  active: ActiveBundlePointer | null;
  bundles: DemoBundleSummary[];
  selection_freezes: DemoSelectionFreezeSummary[];
  final_metrics: DemoMetricSummary[];
  final_raw_baseline_metrics: DemoMetricSummary[];
  final_comparisons: DemoPairedModelComparison[];
  preparation_detail: string;
};

export type ObservableRuleStatus =
  | "NO_OBSERVED_CHANGE"
  | "COMMON_CHANGE_AMBIGUOUS"
  | "SPATIAL_DISAGREEMENT_AMBIGUOUS"
  | "MIXED_OBSERVABLE_CHANGE";

export type DemoContextMode =
  | "local"
  | "local_two_node_ambiguous"
  | "network"
  | "degraded_local";

export type DemoAnalysisResult = {
  schema_version: "aqse.demo-analysis-result.v1";
  result_id: string;
  produced_at_utc: string;
  session_id: string;
  application_id: string;
  bundle_id: string;
  task_id: DemoTaskId;
  profile_id: State8ProfileId;
  profile_fingerprint: string;
  theta_id: string;
  reference_id: string;
  node_count: number;
  sensor_id: string;
  peer_sensor_ids: string[];
  context_mode: DemoContextMode;
  attribution_note: string;
  window_id: string;
  window_start_s: number;
  window_end_exclusive_s: number;
  source_frame_ids: number[];
  feature_names: string[];
  feature_units: string[];
  feature_values: (number | null)[];
  feature_valid: boolean;
  quality_flags: string[];
  encoded_angles: number[] | null;
  afse_vector: number[] | null;
  afse_reference_size: number | null;
  reconstruction_residual: number | null;
  heuristic_ood: boolean | null;
  class_order: string[];
  class_scores: number[] | null;
  predicted_class: string | null;
  displayed_class: string;
  uncertain: boolean | null;
  top_score: number | null;
  top_two_margin: number | null;
  raw_baseline_scores: number[] | null;
  raw_baseline_class: string | null;
  observable_rule_status: ObservableRuleStatus | null;
  score_semantics: "model-score;not-probability-calibrated";
  data_age_ms: number;
  processing_duration_ms: number;
};

export type DemoAnalysisView = {
  schema_version: "aqse.demo-analysis-view.v1";
  session_id: string;
  state:
    | "awaiting_reference"
    | "running"
    | "paused_for_training"
    | "stopped"
    | "failed"
    | "bundle_unavailable";
  state_detail: string;
  worker_epoch: number;
  application_id: string | null;
  active_local_bundle_id: string | null;
  active_network_bundle_id: string | null;
  reference_id: string | null;
  reference_progress: number;
  latest_observation_frame_id: number;
  latest_observation_time_s: number;
  latest_analyzed_window_start_s: number | null;
  completed_window_count: number;
  skipped_window_count: number;
  quality_abstention_count: number;
  queue_depth: 0 | 1;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  result_age_ms: number | null;
  latest_results: DemoAnalysisResult[];
  error: string | null;
};

export type DemoTrainingStepView = {
  task_id: DemoTaskId;
  candidate_name: "theta0" | "protected_qng";
  accepted_updates: number;
  current_loss: number | null;
};

export type DemoTrainingJobView = {
  schema_version: "aqse.demo-training-job.v1";
  job_id: string;
  intent_id: string;
  study_artifact_id: string;
  state: "created" | "running" | "completed" | "cancelled" | "failed";
  created_at_utc: string;
  updated_at_utc: string;
  current_stage: string;
  progress: DemoTrainingStepView[];
  selection_freeze_id: string | null;
  selected_local_bundle_id: string | null;
  selected_network_bundle_id: string | null;
  error: string | null;
};

export type KnowledgeTask = "local" | "network";

export type KnowledgeState8Quality = {
  valid_for_quantum: boolean;
  flags: string[];
  per_feature_valid: [boolean, boolean, boolean, boolean, boolean, boolean, boolean, boolean];
  expected_sample_count: 400;
  received_sample_count: number;
  usable_sample_count: number;
};

export type KnowledgeState8Feature = {
  schema_version: "aqse.state8-feature-record.v1";
  window_id: string;
  session_id: string;
  sensor_id: string;
  profile_id: State8ProfileId;
  profile_fingerprint: string;
  reference_id: string;
  start_time_s: number;
  end_exclusive_time_s: number;
  source_frame_ids: number[];
  peer_sensor_ids: string[];
  values: (number | null)[];
  quality: KnowledgeState8Quality;
};

export type KnowledgeObservationEpisode = {
  schema_version: "aqse.network-demo.knowledge-observation.v1";
  observation_id: string;
  acquisition_id: string;
  content_digest: string;
  task: KnowledgeTask;
  task_id: DemoTaskId;
  profile_id: State8ProfileId;
  profile_fingerprint: string;
  node_count: number;
  feature: KnowledgeState8Feature;
};

export type KnowledgeReviewedLabel = {
  schema_version: "aqse.network-demo.human-reviewed-label.v1";
  label_id: string;
  content_digest: string;
  observation_id: string;
  task: KnowledgeTask;
  task_id: DemoTaskId;
  label: string;
  review_source: "human-review";
  reviewer_id: string;
  reviewed_at_utc: string;
  review_declaration: "label assigned by explicit human review";
};

export type KnowledgeCollectionEntry = {
  observation_id: string;
  observation_content_digest: string;
  observation_file_sha256: string;
  label_id: string;
  label_content_digest: string;
  label_file_sha256: string;
};

export type KnowledgeTrainCollection = {
  schema_version: "aqse.network-demo.train-collection.v1";
  collection_id: string;
  content_digest: string;
  partition: "TRAIN";
  task: KnowledgeTask;
  task_id: DemoTaskId;
  profile_id: State8ProfileId;
  profile_fingerprint: string;
  class_order: string[];
  approved_by: string;
  approved_at_utc: string;
  approval_declaration: "explicitly approved for bounded TRAIN-only retraining";
  entries: KnowledgeCollectionEntry[];
};

export type KnowledgeRegistryView = {
  schema_version: "aqse.network-demo.knowledge-registry.v1";
  observations: KnowledgeObservationEpisode[];
  reviewed_labels: KnowledgeReviewedLabel[];
  approved_train_collections: KnowledgeTrainCollection[];
};

export type KnowledgeCaptureResponse = {
  observation: KnowledgeObservationEpisode;
  reused: boolean;
};

export type KnowledgeLabelResponse = {
  label: KnowledgeReviewedLabel;
  reused: boolean;
};

export type KnowledgeCollectionResponse = {
  collection: KnowledgeTrainCollection;
  reused: boolean;
};
