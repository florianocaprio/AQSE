import { describe, expect, it } from "vitest";

import type { DemoAnalysisView, DemoRegistryView } from "../types/demo";
import { demoReducer, initialDemoState } from "./demo";

const EMPTY_REGISTRY: DemoRegistryView = {
  schema_version: "aqse.demo-registry-view.v2",
  prepared: false,
  scientific_label: "research / not validated for field deployment",
  study_artifact_id: null,
  study_content_digest: null,
  selection_freeze_id: null,
  final_evaluation_id: null,
  historical_test_ledger_sha256: "0".repeat(64),
  active: null,
  bundles: [],
  selection_freezes: [],
  final_metrics: [],
  final_raw_baseline_metrics: [],
  final_comparisons: [],
  preparation_detail: "not prepared",
};

function analysis(
  sessionId: string,
  workerEpoch: number,
  frameId: number,
): DemoAnalysisView {
  return {
    schema_version: "aqse.demo-analysis-view.v1",
    session_id: sessionId,
    state: "running",
    state_detail: "running",
    worker_epoch: workerEpoch,
    application_id: null,
    active_local_bundle_id: null,
    active_network_bundle_id: null,
    reference_id: null,
    reference_progress: 0,
    latest_observation_frame_id: frameId,
    latest_observation_time_s: frameId / 100,
    latest_analyzed_window_start_s: null,
    completed_window_count: 0,
    skipped_window_count: 0,
    quality_abstention_count: 0,
    queue_depth: 0,
    latency_p50_ms: null,
    latency_p95_ms: null,
    result_age_ms: null,
    latest_results: [],
    error: null,
  };
}

describe("demo reducer stale-response guards", () => {
  it("ignores a registry response superseded by a newer request", () => {
    const first = demoReducer(initialDemoState, {
      type: "REGISTRY_REQUEST",
      request_id: "registry-1",
    });
    const second = demoReducer(first, {
      type: "REGISTRY_REQUEST",
      request_id: "registry-2",
    });
    const stale = demoReducer(second, {
      type: "REGISTRY_READY",
      request_id: "registry-1",
      registry: EMPTY_REGISTRY,
    });

    expect(stale).toBe(second);
    expect(stale.registry).toBeNull();
  });

  it("rejects polls from another session or an older worker/frame epoch", () => {
    const current = {
      ...initialDemoState,
      analysis: analysis("session-current", 3, 500),
    };

    expect(demoReducer(current, {
      type: "ANALYSIS_POLL_READY",
      analysis: analysis("session-old", 4, 600),
    })).toBe(current);
    expect(demoReducer(current, {
      type: "ANALYSIS_POLL_READY",
      analysis: analysis("session-current", 2, 700),
    })).toBe(current);
    expect(demoReducer(current, {
      type: "ANALYSIS_POLL_READY",
      analysis: analysis("session-current", 3, 499),
    })).toBe(current);

    const advanced = demoReducer(current, {
      type: "ANALYSIS_POLL_READY",
      analysis: analysis("session-current", 4, 1),
    });
    expect(advanced.analysis?.worker_epoch).toBe(4);
    expect(advanced.analysis?.latest_observation_frame_id).toBe(1);
  });
});
