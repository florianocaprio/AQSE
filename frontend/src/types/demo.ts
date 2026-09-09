export type DemoTaskId = "aqse.local-change.v1" | "aqse.network-pattern.v1";
export type State8ProfileId = "aqse.local-state8.v1" | "aqse.network-state8.v1";

export type DemoMetricSummary = {
  partition: "validation" | "test";
  task_id: DemoTaskId;
  balanced_accuracy: number;
  macro_f1: number;
  coverage: number;
  sample_count: number;
  uncertain_count: number;
  heuristic_ood_count: number;
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

export type DemoRegistryView = {
  schema_version: "aqse.demo-registry-view.v1";
  prepared: boolean;
  scientific_label: "research / not validated for field deployment";
  study_artifact_id: string | null;
  study_content_digest: string | null;
  selection_freeze_id: string | null;
  final_evaluation_id: string | null;
  historical_test_ledger_sha256: string;
  active: ActiveBundlePointer | null;
  bundles: DemoBundleSummary[];
  final_metrics: DemoMetricSummary[];
  preparation_detail: string;
};

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
