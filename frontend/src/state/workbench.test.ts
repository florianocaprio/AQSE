import { describe, expect, it } from "vitest";

import type { FeatureExtractionResponse } from "../types/features";
import type {
  ObservationFrame,
  SessionStatus,
  TruthFrame,
} from "../types/network";
import {
  createInitialWorkbenchState,
  featureResultIsStale,
  workbenchReducer,
} from "./workbench";

describe("workbench reducer", () => {
  it("increments theta draft revisions without mutating the prior vector", () => {
    const initial = createInitialWorkbenchState();
    const next = workbenchReducer(initial, { type: "THETA_VALUE", index: 3, value: 0.5 });
    expect(initial.quantum.theta_draft.value[3]).toBe(0);
    expect(next.quantum.theta_draft.value[3]).toBe(0.5);
    expect(next.quantum.theta_draft.revision).toBe(1);
  });

  it("marks a feature result stale when the selected sensor changes", () => {
    const initial = createInitialWorkbenchState();
    const state = {
      ...initial,
      network: {
        ...initial.network,
        selected_node_id: "S1",
        session: { session_id: "session-1" } as SessionStatus,
      },
      features: {
        ...initial.features,
        source_session_id: "session-1",
        source_data_revision: initial.network.data_revision,
        executed: {
          id: "feature-snapshot",
          value: initial.features.draft.value,
          source_revision: initial.features.draft.revision,
          executed_at: "2026-09-08T00:00:00Z",
        },
        result: { windows: [{ sensor_id: "S1" }] } as FeatureExtractionResponse,
      },
    };
    expect(featureResultIsStale(state)).toBe(false);
    const changed = workbenchReducer(state, { type: "NETWORK_NODE_SELECTED", sensor_id: "S2" });
    expect(featureResultIsStale(changed)).toBe(true);
  });

  it("does not retain truth frames while blind mode is enabled", () => {
    const initial = createInitialWorkbenchState();
    const next = workbenchReducer(initial, {
      type: "NETWORK_TRUTH",
      frames: [{ session_id: "session-1", frame_id: 1 }] as TruthFrame[],
    });
    expect(next.network.truth).toHaveLength(0);
  });

  it("ignores events from the superseded SSE revision across reconnect and clear", () => {
    const initial = workbenchReducer(createInitialWorkbenchState(), {
      type: "NETWORK_BLIND_MODE",
      enabled: false,
    });
    const reconnected = workbenchReducer(initial, {
      type: "NETWORK_STREAM_RECONNECT",
    });
    const cleared = workbenchReducer(reconnected, { type: "NETWORK_CLEAR_DATA" });
    const oldObservation = { frame_id: 91 } as ObservationFrame;
    const oldTruth = { frame_id: 91 } as TruthFrame;
    const oldSession = { session_id: "session-1" } as SessionStatus;

    expect(workbenchReducer(cleared, {
      type: "NETWORK_STREAM",
      status: "connected",
      stream_revision: 0,
    })).toBe(cleared);
    expect(workbenchReducer(cleared, {
      type: "NETWORK_SESSION",
      status: oldSession,
      stream_revision: 0,
    })).toBe(cleared);
    expect(workbenchReducer(cleared, {
      type: "NETWORK_OBSERVATIONS",
      frames: [oldObservation],
      stream_revision: 0,
    })).toBe(cleared);
    expect(workbenchReducer(cleared, {
      type: "NETWORK_TRUTH",
      frames: [oldTruth],
      stream_revision: 0,
    })).toBe(cleared);

    const current = workbenchReducer(cleared, {
      type: "NETWORK_OBSERVATIONS",
      frames: [{ frame_id: 1 } as ObservationFrame],
      stream_revision: 1,
    });
    expect(current.network.observations).toHaveLength(1);
    expect(current.network.observations[0].frame_id).toBe(1);
  });

  it("retains the requested configuration snapshot in the experiment ledger", () => {
    const initial = createInitialWorkbenchState();
    const requestSnapshot = {
      domain: "scalar_simulation" as const,
      configuration: {
        sensor_id: "M1",
        duration: 2,
        sampling_rate: 100,
        background_field: 4.5e-5,
        amplitude: 1e-6,
        frequency: 5,
        phase: 0,
        drift_rate: 0,
        noise_std: 1e-8,
        temperature: 293.15,
        anomaly_enabled: false,
        anomaly_time: null,
        anomaly_amplitude: 0,
        random_seed: 17,
      },
    };
    const next = workbenchReducer(initial, {
      type: "EXPERIMENT_ADD",
      experiment: {
        id: "experiment-scalar-1",
        created_at: "2026-09-08T00:00:00Z",
        kind: "scalar_simulation",
        status: "completed",
        title: "Scalar simulation · M1",
        request_snapshot: requestSnapshot,
        warnings: [],
        timings_ms: {},
      },
    });

    expect(next.experiments[0]?.request_snapshot).toEqual(requestSnapshot);
    expect(
      JSON.parse(JSON.stringify(next.experiments))[0].request_snapshot,
    ).toEqual(requestSnapshot);
  });
});
