import type {
  ActiveBundlePointer,
  DemoAnalysisView,
  DemoRegistryView,
  DemoTrainingJobView,
  KnowledgeCaptureResponse,
  KnowledgeCollectionResponse,
  KnowledgeLabelResponse,
  KnowledgeRegistryView,
  KnowledgeTask,
} from "../types/demo";
import { requestJson } from "./client";

const DEMO_BASE = "/api/demo";

export function getDemoRegistry(signal?: AbortSignal): Promise<DemoRegistryView> {
  return requestJson<DemoRegistryView>(`${DEMO_BASE}/registry`, { signal });
}

export function getDemoKnowledge(
  signal?: AbortSignal,
): Promise<KnowledgeRegistryView> {
  return requestJson<KnowledgeRegistryView>(`${DEMO_BASE}/knowledge`, { signal });
}

export function captureDemoKnowledgeObservation(
  request: {
    schema_version: "aqse.network-demo.knowledge-capture-request.v1";
    session_id: string;
    sensor_id: string;
    task: KnowledgeTask;
  },
  signal?: AbortSignal,
): Promise<KnowledgeCaptureResponse> {
  return requestJson<KnowledgeCaptureResponse>(`${DEMO_BASE}/knowledge/observations`, {
    method: "POST",
    body: request,
    signal,
  });
}

export function addDemoKnowledgeLabel(
  request: {
    schema_version: "aqse.network-demo.reviewed-label-request.v1";
    observation_id: string;
    task: KnowledgeTask;
    label: string;
    reviewer_id: string;
    reviewed_at_utc: string;
  },
  signal?: AbortSignal,
): Promise<KnowledgeLabelResponse> {
  return requestJson<KnowledgeLabelResponse>(`${DEMO_BASE}/knowledge/labels`, {
    method: "POST",
    body: request,
    signal,
  });
}

export function approveDemoTrainCollection(
  request: {
    schema_version: "aqse.network-demo.train-approval-request.v1";
    partition: "TRAIN";
    task: KnowledgeTask;
    observation_ids: string[];
    approved_by: string;
    approved_at_utc: string;
    approval_declaration: "explicitly approved for bounded TRAIN-only retraining";
  },
  signal?: AbortSignal,
): Promise<KnowledgeCollectionResponse> {
  return requestJson<KnowledgeCollectionResponse>(
    `${DEMO_BASE}/knowledge/train-collections`,
    { method: "POST", body: request, signal },
  );
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
    local_train_collection_id?: string;
    network_train_collection_id?: string;
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
