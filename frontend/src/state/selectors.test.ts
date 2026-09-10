import { describe, expect, it } from "vitest";

import type { ObservationFrame, SensorNodeConfiguration, SensorReading } from "../types/network";
import {
  chartPointsForSensor,
  differenceCompatibility,
  latestContiguousVectorSeries,
  observedDifferenceTesla,
} from "./selectors";

describe("observation-only selectors", () => {
  it("uses the latest contiguous vector segment and never fills a dropout", () => {
    const frames = [
      ...Array.from({ length: 18 }, (_, index) => frame(index + 1, true)),
      frame(19, false),
      ...Array.from({ length: 16 }, (_, index) => frame(index + 20, true)),
    ];

    const series = latestContiguousVectorSeries(frames, "S1", 100);

    expect(series?.time_s).toHaveLength(16);
    expect(series?.acquisition_id).toContain(":20-35");
    expect(series?.measured_field[0]).toEqual([20e-9, 40e-9, 60e-9]);
  });

  it("renders invalid readings as chart gaps", () => {
    const points = chartPointsForSensor([frame(1, true), frame(2, false)], "S1");
    expect(points[0].x_nt).toBeCloseTo(1);
    expect(points[1].x_nt).toBeNull();
    expect(points[1].quality_flags).toContain("signal_absent");
  });

  it("renders received stuck and clock-errored payloads while preserving flags", () => {
    const validFrame = frame(1, true);
    const stuckBase = frame(2, true);
    const clockBase = frame(3, true);
    const stuckFrame = {
      ...stuckBase,
      readings: [{ ...stuckBase.readings[0], valid: false, quality_flags: ["stuck"] as SensorReading["quality_flags"] }],
    };
    const timedFrame = {
      ...clockBase,
      readings: [{ ...clockBase.readings[0], quality_flags: ["clock_error"] as SensorReading["quality_flags"] }],
    };

    const points = chartPointsForSensor([validFrame, stuckFrame, timedFrame], "S1");
    expect(points[1].x_nt).toBeCloseTo(2);
    expect(points[1].quality_flags).toContain("stuck");
    expect(points[2].x_nt).toBeCloseTo(3);
    expect(points[2].quality_flags).toContain("clock_error");
  });

  it("keeps observed clipped vectors so the backend quality gate can reject them", () => {
    const frames = Array.from({ length: 16 }, (_, index) => frame(index + 1, true));
    frames[8] = clippedFrame(9);
    const series = latestContiguousVectorSeries(frames, "S1", 100);
    expect(series?.time_s).toHaveLength(16);
    expect(series?.saturation_mask[8]).toEqual([true, false, false]);
  });

  it("starts a new feature segment after an observed clock error", () => {
    const frames = [
      ...Array.from({ length: 16 }, (_, index) => frame(index + 1, true)),
      clockErrorFrame(17),
      ...Array.from({ length: 16 }, (_, index) => frame(index + 18, true)),
    ];

    const series = latestContiguousVectorSeries(frames, "S1", 100);

    expect(series?.time_s).toHaveLength(16);
    expect(series?.acquisition_id).toContain(":18-33");
  });

  it("keeps the first valid frame after a retained-buffer gap", () => {
    const frames = [
      ...Array.from({ length: 20 }, (_, index) => frame(index + 1, true)),
      ...Array.from({ length: 16 }, (_, index) => frame(index + 22, true)),
    ];

    const series = latestContiguousVectorSeries(frames, "S1", 100);

    expect(series?.time_s).toHaveLength(16);
    expect(series?.acquisition_id).toContain(":22-37");
  });

  it("computes differences only for compatible, valid readouts", () => {
    const compatibility = differenceCompatibility(node("vector"), node("vector"));
    const selected = frame(3, true).readings[0];
    const reference = frame(1, true).readings[0];
    expect(compatibility.compatible).toBe(true);
    expect(observedDifferenceTesla(selected, reference, compatibility)).toBeCloseTo(
      Math.hypot(3e-9, 6e-9, 9e-9) - Math.hypot(1e-9, 2e-9, 3e-9),
    );

    const stuck = { ...reference, quality_flags: ["stuck"] as SensorReading["quality_flags"] };
    expect(observedDifferenceTesla(selected, stuck, compatibility)).toBeNull();
    expect(differenceCompatibility(node("vector"), node("total_field")).compatible).toBe(false);
    expect(differenceCompatibility(node("monoaxial", [1, 0, 0]), node("monoaxial", [0, 1, 0])).compatible).toBe(false);
  });
});

function frame(frameId: number, valid: boolean): ObservationFrame {
  const reading: SensorReading = {
    sensor_id: "S1",
    sequence_id: frameId,
    acquisition_time: "2026-09-08T00:00:00Z",
    arrival_time: "2026-09-08T00:00:00Z",
    measurement_mode: "vector",
    value_T: null,
    components_T: valid ? [frameId * 1e-9, frameId * 2e-9, frameId * 3e-9] : null,
    saturation_mask: valid ? [false, false, false] : null,
    quality_flags: valid ? [] : ["signal_absent"],
    valid,
    observed_temperature_K: 293.15,
    position_m: [0, 0, 0],
    orientation_world_to_sensor_wxyz: [1, 0, 0, 0],
    calibration_version: "test-v1",
  };
  return {
    schema_version: "1.0",
    session_id: "session-test",
    frame_id: frameId,
    configuration_version: 1,
    sim_time_s: (frameId - 1) / 100,
    readings: [reading],
  };
}

function clippedFrame(frameId: number): ObservationFrame {
  const value = frame(frameId, true);
  return {
    ...value,
    readings: [{
      ...value.readings[0],
      valid: false,
      saturation_mask: [true, false, false],
      quality_flags: ["clipped"],
    }],
  };
}

function clockErrorFrame(frameId: number): ObservationFrame {
  const value = frame(frameId, true);
  return {
    ...value,
    readings: [{
      ...value.readings[0],
      quality_flags: ["clock_error"],
    }],
  };
}

function node(
  measurementMode: SensorNodeConfiguration["measurement_mode"],
  axis: SensorNodeConfiguration["monoaxial_axis_sensor"] = [1, 0, 0],
): SensorNodeConfiguration {
  return {
    measurement_mode: measurementMode,
    monoaxial_axis_sensor: axis,
    orientation_world_to_sensor_wxyz: [1, 0, 0, 0],
  } as unknown as SensorNodeConfiguration;
}
