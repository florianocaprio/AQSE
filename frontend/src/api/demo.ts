import type {
  ActiveBundlePointer,
  DemoAnalysisView,
  DemoRegistryView,
  DemoTrainingJobView,
} from "../types/demo";
import { requestJson } from "./client";

const DEMO_BASE = "/api/demo";

export function getDemoRegistry(signal?: AbortSignal): Promise<DemoRegistryView> {
  return requestJson<DemoRegistryView>(`${DEMO_BASE}/registry`, { signal });
}

export function startDemoAnalysis(
  sessionId: string,
  signal?: AbortSignal,
): Promise<DemoAnalysisView> {
  return requestJson<DemoAnalysisView>(`${analysisPath(sessionId)}/start`, {
    method: "POST",
    body: { acquire_reference: true },
    signal,
  });
}

export function getDemoAnalysis(
  sessionId: string,
  signal?: AbortSignal,
): Promise<DemoAnalysisView> {
  return requestJson<DemoAnalysisView>(analysisPath(sessionId), { signal });
}

export function stopDemoAnalysis(
  sessionId: string,
  signal?: AbortSignal,
): Promise<DemoAnalysisView> {
  return requestJson<DemoAnalysisView>(`${analysisPath(sessionId)}/stop`, {
    method: "POST",
    signal,
  });
}

export function applyDemoBundles(
  request: {
    local_bundle_id: string;
    network_bundle_id: string;
    selection_freeze_id: string;
  },
  signal?: AbortSignal,
): Promise<ActiveBundlePointer> {
  return requestJson<ActiveBundlePointer>(`${DEMO_BASE}/bundles/apply`, {
    method: "POST",
    body: request,
    signal,
  });
}

export function startDemoTraining(
  request: {
    intent_id: string;
    study_artifact_id: string;
    requested_action: "fit-two-candidate-local-and-network-bundles";
  },
  signal?: AbortSignal,
): Promise<DemoTrainingJobView> {
  return requestJson<DemoTrainingJobView>(`${DEMO_BASE}/training/jobs`, {
    method: "POST",
    body: request,
    signal,
  });
}

export function getDemoTrainingJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<DemoTrainingJobView> {
  return requestJson<DemoTrainingJobView>(
    `${DEMO_BASE}/training/jobs/${encodeURIComponent(jobId)}`,
    { signal },
  );
}

export function cancelDemoTrainingJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<DemoTrainingJobView> {
  return requestJson<DemoTrainingJobView>(
    `${DEMO_BASE}/training/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: "POST", signal },
  );
}

function analysisPath(sessionId: string): string {
  return `${DEMO_BASE}/analysis/${encodeURIComponent(sessionId)}`;
}
