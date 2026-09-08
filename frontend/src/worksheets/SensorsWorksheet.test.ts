import { describe, expect, it } from "vitest";

import type { SensorNodeConfiguration } from "../types/network";
import { nextRandomSeed, targetsForEventKind } from "./SensorsWorksheet";

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
});

function node(sensorId: string): SensorNodeConfiguration {
  return { sensor_id: sensorId } as SensorNodeConfiguration;
}
