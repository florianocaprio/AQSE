import type { MeasuredVectorSeries } from "../types/features";
import type {
  ObservationFrame,
  SensorReading,
  SensorNodeConfiguration,
  TruthFrame,
} from "../types/network";
import type { Vector3 } from "../types/workbench";

export type NetworkChartPoint = {
  frame_id: number;
  time_s: number;
  x_nt: number | null;
  y_nt: number | null;
  z_nt: number | null;
  scalar_nt: number | null;
  valid: boolean;
  quality_flags: string[];
};

export type DifferenceCompatibility = {
  compatible: boolean;
  mode: SensorNodeConfiguration["measurement_mode"] | null;
  quantity: string;
  reason: string | null;
};

export function readingForSensor(
  frame: ObservationFrame,
  sensorId: string,
): SensorReading | null {
  return frame.readings.find((reading) => reading.sensor_id === sensorId) ?? null;
}

export function chartPointsForSensor(
  frames: ObservationFrame[],
  sensorId: string,
  maximumPoints = 600,
): NetworkChartPoint[] {
  return frames.slice(-maximumPoints).map((frame) => {
    const reading = readingForSensor(frame, sensorId);
    const hasReceivedPayload = Boolean(
      reading &&
      !reading.quality_flags.includes("signal_absent") &&
      (reading.components_T !== null || reading.value_T !== null),
    );
    const components = hasReceivedPayload ? reading?.components_T : null;
    return {
      frame_id: frame.frame_id,
      time_s: frame.sim_time_s,
      x_nt: components ? components[0] * 1e9 : null,
      y_nt: components ? components[1] * 1e9 : null,
      z_nt: components ? components[2] * 1e9 : null,
      scalar_nt:
        !hasReceivedPayload || reading?.value_T === null || reading?.value_T === undefined
          ? null
          : reading.value_T * 1e9,
      valid: reading?.valid ?? false,
      quality_flags: reading?.quality_flags ?? ["signal_absent"],
    };
  });
}

/**
 * Produces a feature-service input from the latest uninterrupted run of valid
 * vector observations. Missing readings, clock errors, incompatible invalid
 * readings, and frame gaps reset the run; they are never interpolated or
 * bridged in the browser.
 */
export function latestContiguousVectorSeries(
  frames: ObservationFrame[],
  sensorId: string,
  samplingRateHz: number,
): MeasuredVectorSeries | null {
  let segment: Array<{
    frame: ObservationFrame;
    reading: SensorReading;
    components: Vector3;
  }> = [];
  let previousFrameId: number | null = null;

  for (const frame of [...frames].sort((a, b) => a.frame_id - b.frame_id)) {
    const reading = readingForSensor(frame, sensorId);
    const frameIsContiguous =
      previousFrameId === null || frame.frame_id === previousFrameId + 1;
    const isClippedObservation = Boolean(
      reading?.components_T &&
      reading.saturation_mask?.some(Boolean) &&
      reading.quality_flags.every((flag) => flag === "clipped"),
    );
    const isUsableReading =
      reading?.measurement_mode === "vector" &&
      !reading.quality_flags.includes("clock_error") &&
      !reading.quality_flags.includes("stuck") &&
      (reading.valid || isClippedObservation) &&
      reading.components_T !== null;

    if (!frameIsContiguous) segment = [];
    if (!isUsableReading) {
      segment = [];
    } else {
      segment.push({ frame, reading, components: reading.components_T as Vector3 });
    }
    previousFrameId = frame.frame_id;
  }

  if (segment.length < 16) return null;

  const first = segment[0];
  const last = segment[segment.length - 1];
  return {
    acquisition_id: `${last.frame.session_id}:${sensorId}:${first.frame.frame_id}-${last.frame.frame_id}`,
    sensor_id: sensorId,
    sampling_rate_hz: samplingRateHz,
    time_s: segment.map(({ frame }) => frame.sim_time_s),
    measured_field: segment.map(({ components }) => components),
    field_unit: "T",
    temperature_k: segment.map(({ reading }) => reading.observed_temperature_K),
    saturation_mask: segment.map(({ reading }) =>
      reading.saturation_mask ?? ([false, false, false] as const),
    ),
  };
}

export function differenceCompatibility(
  selected: SensorNodeConfiguration | undefined,
  reference: SensorNodeConfiguration | undefined,
): DifferenceCompatibility {
  if (!selected || !reference) {
    return {
      compatible: false,
      mode: null,
      quantity: "unavailable",
      reason: "Both executed node configurations are required.",
    };
  }
  if (selected.measurement_mode !== reference.measurement_mode) {
    return {
      compatible: false,
      mode: null,
      quantity: "unavailable",
      reason: "Selected and reference nodes use different measurement modes.",
    };
  }
  if (selected.measurement_mode === "monoaxial") {
    const selectedAxis = sensorAxisInWorld(selected);
    const referenceAxis = sensorAxisInWorld(reference);
    if (Math.hypot(...selectedAxis.map((value, index) => value - referenceAxis[index])) > 1e-8) {
      return {
        compatible: false,
        mode: null,
        quantity: "unavailable",
        reason: "Monoaxial nodes do not declare equivalent world-frame readout axes.",
      };
    }
  }
  return {
    compatible: true,
    mode: selected.measurement_mode,
    quantity: selected.measurement_mode === "vector"
      ? "Euclidean vector magnitude"
      : selected.measurement_mode === "monoaxial"
        ? "signed monoaxial scalar"
        : "total-field scalar",
    reason: null,
  };
}

export function observedDifferenceTesla(
  selected: SensorReading | undefined,
  reference: SensorReading | undefined,
  compatibility: DifferenceCompatibility,
): number | null {
  if (!compatibility.compatible || compatibility.mode === null) return null;
  const selectedValue = comparableValue(selected, compatibility.mode);
  const referenceValue = comparableValue(reference, compatibility.mode);
  return selectedValue === null || referenceValue === null
    ? null
    : selectedValue - referenceValue;
}

export function truthForSensor(
  frames: TruthFrame[],
  sensorId: string,
): Array<{
  frame_id: number;
  time_s: number;
  field_T: Vector3 | null;
  model_valid: boolean;
}> {
  return frames.map((frame) => {
    const field = frame.fields.find((item) => item.sensor_id === sensorId);
    return {
      frame_id: frame.frame_id,
      time_s: frame.sim_time_s,
      field_T: field?.field_true_world_T ?? null,
      model_valid: field?.model_valid ?? false,
    };
  });
}

function comparableValue(
  reading: SensorReading | undefined,
  mode: SensorNodeConfiguration["measurement_mode"],
): number | null {
  if (
    !reading?.valid ||
    reading.measurement_mode !== mode ||
    reading.quality_flags.includes("stuck") ||
    reading.quality_flags.includes("clock_error")
  ) return null;
  if (mode === "vector") {
    return reading.components_T ? Math.hypot(...reading.components_T) : null;
  }
  return reading.value_T;
}

function sensorAxisInWorld(node: SensorNodeConfiguration): Vector3 {
  const [w, x, y, z] = node.orientation_world_to_sensor_wxyz;
  const [vx, vy, vz] = node.monoaxial_axis_sensor;
  const qx = -x;
  const qy = -y;
  const qz = -z;
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  return [
    vx + w * tx + (qy * tz - qz * ty),
    vy + w * ty + (qz * tx - qx * tz),
    vz + w * tz + (qx * ty - qy * tx),
  ];
}
