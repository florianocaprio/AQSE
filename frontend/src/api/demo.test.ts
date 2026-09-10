import { afterEach, describe, expect, it, vi } from "vitest";

import {
  addDemoKnowledgeLabel,
  approveDemoTrainCollection,
  captureDemoKnowledgeObservation,
  getDemoKnowledge,
  startDemoTraining,
} from "./demo";

function jsonResponse(payload: unknown = {}): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("demo knowledge API", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("reads the registry without triggering capture, labelling or training", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => jsonResponse());
    vi.stubGlobal("fetch", fetchMock);

    await getDemoKnowledge();

    expect(fetchMock).toHaveBeenCalledOnce();
    const [path, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe("/api/demo/knowledge");
    expect(options.method).toBeUndefined();
    expect(options.body).toBeUndefined();
  });

  it("uses three distinct explicit writes for observation, human label and TRAIN approval", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => jsonResponse());
    vi.stubGlobal("fetch", fetchMock);

    await captureDemoKnowledgeObservation({
      schema_version: "aqse.network-demo.knowledge-capture-request.v1",
      session_id: "session-1",
      sensor_id: "S1",
      task: "local",
    });
    await addDemoKnowledgeLabel({
      schema_version: "aqse.network-demo.reviewed-label-request.v1",
      observation_id: "aqse-knowledge-observation-0123456789abcdef",
      task: "local",
      label: "CHANGE_DETECTED",
      reviewer_id: "reviewer-1",
      reviewed_at_utc: "2026-09-10T00:00:00.000Z",
    });
    await approveDemoTrainCollection({
      schema_version: "aqse.network-demo.train-approval-request.v1",
      partition: "TRAIN",
      task: "local",
      observation_ids: ["aqse-knowledge-observation-0123456789abcdef"],
      approved_by: "approver-1",
      approved_at_utc: "2026-09-10T00:00:00.000Z",
      approval_declaration: "explicitly approved for bounded TRAIN-only retraining",
    });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/api/demo/knowledge/observations",
      "/api/demo/knowledge/labels",
      "/api/demo/knowledge/train-collections",
    ]);
    for (const [, options] of fetchMock.mock.calls as [string, RequestInit][]) {
      expect(options.method).toBe("POST");
      expect(options.body).toEqual(expect.any(String));
    }
    const labelPayload = JSON.parse(String(fetchMock.mock.calls[1][1]?.body)) as Record<string, unknown>;
    expect(labelPayload).toMatchObject({
      reviewer_id: "reviewer-1",
      label: "CHANGE_DETECTED",
    });
    expect(labelPayload).not.toHaveProperty("review_source");
  });

  it("includes both approved TRAIN collections only in an explicit retraining intent", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => jsonResponse());
    vi.stubGlobal("fetch", fetchMock);

    await startDemoTraining({
      intent_id: "aqse-demo-intent-ui-test",
      study_artifact_id: "study-1",
      requested_action: "fit-two-candidate-local-and-network-bundles",
      local_train_collection_id: "aqse-knowledge-train-0123456789abcdef",
      network_train_collection_id: "aqse-knowledge-train-fedcba9876543210",
    });

    const [path, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    const payload = JSON.parse(String(options.body)) as Record<string, unknown>;
    expect(path).toBe("/api/demo/training/jobs");
    expect(options.method).toBe("POST");
    expect(payload).toMatchObject({
      local_train_collection_id: "aqse-knowledge-train-0123456789abcdef",
      network_train_collection_id: "aqse-knowledge-train-fedcba9876543210",
    });
  });
});
