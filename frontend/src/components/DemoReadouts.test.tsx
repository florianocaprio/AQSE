import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type {
  DemoAnalysisResult,
  DemoAnalysisView,
  DemoBundleSummary,
  DemoRegistryView,
} from "../types/demo";
import {
  DemoMetricSummaryTable,
  activeBundleForTask,
  bundleForResult,
  latestResultForSensor,
} from "./DemoReadouts";

const RESULT: DemoAnalysisResult = {
  schema_version: "aqse.demo-analysis-result.v1",
  result_id: "result-1",
  produced_at_utc: "2026-09-10T10:00:00Z",
  session_id: "session-1",
  application_id: "application-1",
  bundle_id: "bundle-network",
  task_id: "aqse.network-pattern.v1",
  profile_id: "aqse.network-state8.v1",
  profile_fingerprint: "profile-fingerprint",
  theta_id: "theta-1",
  reference_id: "reference-1",
  node_count: 4,
  sensor_id: "S3",
  peer_sensor_ids: ["S1", "S2", "S4"],
  context_mode: "network",
  attribution_note: "Pattern is compatible with a node-specific change.",
  window_id: "window-1",
  window_start_s: 18,
  window_end_exclusive_s: 22,
  source_frame_ids: [1800, 2199],
  feature_names: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
  feature_units: ["nT", "nT", "nT/s", "dB", "K", "nT", "1", "1"],
  feature_values: [1, 2, 3, 4, 5, 6, 0.8, 1],
  feature_valid: true,
  quality_flags: [],
  encoded_angles: [0, 0, 0, 0, 0, 0, 0, 0],
  afse_vector: [0.1, 0.2],
  afse_reference_size: 2,
  reconstruction_residual: 0.02,
  heuristic_ood: false,
  class_order: ["NORMAL", "DEVICE_COMPATIBLE"],
  class_scores: [0.1, 0.9],
  predicted_class: "DEVICE_COMPATIBLE",
  displayed_class: "DEVICE_COMPATIBLE",
  uncertain: false,
  top_score: 0.9,
  top_two_margin: 0.8,
  raw_baseline_scores: [0.2, 0.8],
  raw_baseline_class: "DEVICE_COMPATIBLE",
  score_semantics: "model-score;not-probability-calibrated",
  data_age_ms: 12,
  processing_duration_ms: 4,
};

const ANALYSIS: DemoAnalysisView = {
  schema_version: "aqse.demo-analysis-view.v1",
  session_id: "session-1",
  state: "running",
  state_detail: "running",
  worker_epoch: 1,
  application_id: "application-1",
  active_local_bundle_id: "bundle-local",
  active_network_bundle_id: "bundle-network",
  reference_id: "reference-1",
  reference_progress: 1,
  latest_observation_frame_id: 2200,
  latest_observation_time_s: 22,
  latest_analyzed_window_start_s: 18,
  completed_window_count: 1,
  skipped_window_count: 0,
  quality_abstention_count: 0,
  queue_depth: 0,
  latency_p50_ms: 4,
  latency_p95_ms: 4,
  result_age_ms: 20,
  latest_results: [RESULT],
  error: null,
};

const NETWORK_BUNDLE: DemoBundleSummary = {
  bundle_id: "bundle-network",
  task_id: "aqse.network-pattern.v1",
  profile_id: "aqse.network-state8.v1",
  theta_candidate_name: "protected_qng",
  theta_id: "theta-1",
  accepted_qng_updates: 3,
  afse_method_id: "aqse.afse.nystrom-ridge32.v1",
  afse_reference_size: 32,
  afse_ridge_lambda: 1e-6,
  classifier_model_id: "classifier-1",
  class_order: ["NORMAL", "ENVIRONMENT_COMPATIBLE", "DEVICE_COMPATIBLE", "MIXED_OR_AMBIGUOUS"],
  validation: {
    task_id: "aqse.network-pattern.v1",
    partition: "validation",
    balanced_accuracy: 0.75,
    macro_f1: 0.7,
    coverage: 0.9,
    sample_count: 32,
    uncertain_count: 2,
    heuristic_ood_count: 1,
  },
  raw_baseline_validation: {
    task_id: "aqse.network-pattern.v1",
    partition: "validation",
    balanced_accuracy: 0.7,
    macro_f1: 0.65,
    coverage: 0.9,
    sample_count: 32,
    uncertain_count: 2,
    heuristic_ood_count: 0,
  },
  active: true,
};

const REGISTRY: DemoRegistryView = {
  schema_version: "aqse.demo-registry-view.v1",
  prepared: true,
  scientific_label: "research / not validated for field deployment",
  study_artifact_id: "study-1",
  study_content_digest: "digest-1",
  selection_freeze_id: "freeze-1",
  final_evaluation_id: "evaluation-1",
  historical_test_ledger_sha256: "ledger-hash",
  active: {
    application_id: "application-1",
    generation: 1,
    revision: 1,
    selection_freeze_id: "freeze-1",
    local_bundle_id: "bundle-local",
    network_bundle_id: "bundle-network",
    previous_application_id: null,
  },
  bundles: [NETWORK_BUNDLE],
  final_metrics: [],
  preparation_detail: "ready",
};

describe("demo readout selectors", () => {
  it("selects a real sensor result and resolves its exact bundle identity", () => {
    expect(latestResultForSensor(ANALYSIS, "S3")).toBe(RESULT);
    expect(latestResultForSensor(ANALYSIS, "missing")).toBeNull();
    expect(latestResultForSensor(ANALYSIS, null)).toBe(RESULT);
    expect(latestResultForSensor(ANALYSIS, "S3", "new-session")).toBeNull();
    expect(bundleForResult(REGISTRY, RESULT)).toBe(NETWORK_BUNDLE);
    expect(activeBundleForTask(REGISTRY, "aqse.network-pattern.v1")).toBe(NETWORK_BUNDLE);
  });

  it("renders measured metrics with their partition and counts", () => {
    const markup = renderToStaticMarkup(
      <DemoMetricSummaryTable metrics={NETWORK_BUNDLE.validation} label="Validation" />,
    );
    expect(markup).toContain("validation");
    expect(markup).toContain("75.0%");
    expect(markup).toContain("32");
  });
});
