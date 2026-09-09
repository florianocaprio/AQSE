import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import { describe, expect, it, vi } from "vitest";

import type {
  NetworkSessionConfiguration,
  ScheduledEventResponse,
  SensorNodeConfiguration,
} from "../types/network";
import {
  buildLiveNetworkEvent,
  LiveEventControls,
  nextRandomSeed,
  submitLiveNetworkEvent,
  targetsForEventKind,
} from "./SensorsWorksheet";

describe("sensor worksheet draft helpers", () => {
  it("generates a distinct bounded unsigned seed without starting a session", () => {
    const next = nextRandomSeed(42);
    expect(Number.isInteger(next)).toBe(true);
    expect(next).toBeGreaterThanOrEqual(0);
    expect(next).toBeLessThanOrEqual(2 ** 32 - 1);
    expect(next).not.toBe(42);
  });

  it("normalizes event target counts for world and shared events", () => {
    const nodes = [node("S1"), node("S2"), node("S3")];
    expect(targetsForEventKind("world_field_offset", ["S1"], nodes)).toEqual([]);
    expect(targetsForEventKind("shared_instrument_offset", ["S1"], nodes)).toEqual(["S1", "S2"]);
    expect(targetsForEventKind("dropout", [], nodes)).toEqual(["S1"]);
  });

  it("builds a common event strictly in the future without device targets", () => {
    const event = buildLiveNetworkEvent({
      eventId: "live-common",
      intent: "environment_common",
      simTimeS: 12,
      leadTimeS: 0.01,
      durationS: 2,
      magnitude: 125,
      targetSensorId: "S1",
      configuration: networkConfiguration(),
    });

    expect(event.kind).toBe("world_field_offset");
    expect(event.target_sensor_ids).toEqual([]);
    expect(event.start_time_s).toBe(13);
    expect(event.start_time_s).toBeGreaterThan(12);
    expect(event.field_offset_T[0]).toBeCloseTo(125e-9, 15);
    expect(event.field_offset_T.slice(1)).toEqual([0, 0]);
  });

  it("maps shared and clipping intents to honest backend event contracts", () => {
    const configuration = networkConfiguration();
    const shared = buildLiveNetworkEvent({
      eventId: "live-shared",
      intent: "shared_instrument",
      simTimeS: 0,
      leadTimeS: 5,
      durationS: 1,
      magnitude: 50,
      targetSensorId: "S2",
      configuration,
    });
    const clipping = buildLiveNetworkEvent({
      eventId: "live-clipping",
      intent: "clipping",
      simTimeS: 0,
      leadTimeS: 5,
      durationS: 1,
      magnitude: 0,
      targetSensorId: "S2",
      configuration,
    });

    expect(shared.kind).toBe("shared_instrument_offset");
    expect(shared.target_sensor_ids).toEqual(["S1", "S2"]);
    expect(clipping.kind).toBe("node_bias");
    expect(clipping.target_sensor_ids).toEqual(["S2"]);
    expect(clipping.field_offset_T).toEqual([0, 0.0016, 0]);
  });

  it("reports backend acceptance, rejection, and stale responses distinctly", async () => {
    const event = buildLiveNetworkEvent({
      eventId: "live-dropout",
      intent: "dropout",
      simTimeS: 0,
      leadTimeS: 5,
      durationS: 1,
      magnitude: 0,
      targetSensorId: "S1",
      configuration: networkConfiguration(),
    });
    const response: ScheduledEventResponse = {
      event,
      configuration_version: 2,
      effective_frame_id: 42,
    };
    const scheduler = vi.fn().mockResolvedValue(response);

    const accepted = await submitLiveNetworkEvent({
      sessionId: "session-1",
      event,
      isCurrent: () => true,
      scheduler,
    });
    expect(accepted.status).toBe("scheduled");
    expect(accepted.detail).toContain("Backend accepted");

    let current = true;
    const stale = await submitLiveNetworkEvent({
      sessionId: "session-1",
      event,
      isCurrent: () => current,
      scheduler: async () => {
        current = false;
        return response;
      },
    });
    expect(stale.status).toBe("stale");
    expect(stale.detail).toContain("previous session");

    const rejected = await submitLiveNetworkEvent({
      sessionId: "session-1",
      event,
      isCurrent: () => true,
      scheduler: async () => { throw new Error("event start is no longer in the future"); },
    });
    expect(rejected.status).toBe("rejected");
    expect(rejected.detail).toContain("no longer in the future");
  });

  it("exposes every approved live event while disabling submission without a session", () => {
    const markup = renderToStaticMarkup(
      createElement(LiveEventControls, {
        session: null,
        configuration: null,
        selectedNodeId: null,
      }),
    );

    expect(markup).toContain("Common environmental field step");
    expect(markup).toContain("Single-node drift");
    expect(markup).toContain("Single-node noise burst");
    expect(markup).toContain("Shared instrument offset");
    expect(markup).toContain("Sensor dropout");
    expect(markup).toContain("Clipping stress");
    expect(markup).toContain("Stuck reading");
    expect(markup).toContain("Create a session before scheduling");
    expect(markup).toContain("disabled");
  });
});

function node(sensorId: string, limits: readonly [number, number, number] | null = null): SensorNodeConfiguration {
  return {
    sensor_id: sensorId,
    errors: {
      saturation_limit_T: 0.0008,
      saturation_limits_T: limits,
    },
  } as SensorNodeConfiguration;
}

function networkConfiguration(): NetworkSessionConfiguration {
  return {
    sampling_rate_Hz: 100,
    time_scale: 4,
    nodes: [node("S1"), node("S2", [0.0008, 0.0004, 0.0009])],
  } as NetworkSessionConfiguration;
}
