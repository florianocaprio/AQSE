import { describe, expect, it, vi } from "vitest";

import type { NetworkSessionConfiguration } from "../types/network";
import {
  emitNetworkExperiment,
  networkStreamCursor,
  shouldReconnectAfterControl,
} from "./useWorkbenchController";

describe("network stream control lifecycle", () => {
  it("forces a fresh SSE subscription after reset from paused or stepped-created state", () => {
    expect(shouldReconnectAfterControl("reset", "paused")).toBe(true);
    expect(shouldReconnectAfterControl("reset", "created")).toBe(true);
  });

  it("reconnects after replay but not after ordinary run-state controls", () => {
    expect(shouldReconnectAfterControl("replay", "stopped")).toBe(true);
    expect(shouldReconnectAfterControl("pause", "running")).toBe(false);
    expect(shouldReconnectAfterControl("resume", "paused")).toBe(false);
  });

  it("starts a revision-forced EventSource from frame zero", () => {
    const previous = { session_id: "session-1", stream_revision: 3 };

    expect(networkStreamCursor(previous, {
      session_id: "session-1",
      stream_revision: 4,
    }, 275)).toBe(0);
    expect(networkStreamCursor(previous, previous, 275)).toBe(275);
    expect(networkStreamCursor(null, previous, 275)).toBe(0);
  });

  it.each(["reset", "delete"] as const)(
    "emits one detached experiment action for a successful %s path",
    (operation) => {
      const configuration = networkConfiguration();
      const dispatch = vi.fn();

      emitNetworkExperiment(dispatch, {
        id: `experiment-${operation}`,
        created_at: "2026-09-08T00:00:00Z",
        operation,
        session_id: "session-1",
        configuration,
        status: "completed",
        title: `${operation} complete`,
        sample_count: 25,
      });
      configuration.session_name = "Mutated draft";
      configuration.nodes[0].errors.ou_tau_s = 99;

      expect(dispatch).toHaveBeenCalledTimes(1);
      const action = dispatch.mock.calls[0][0];
      expect(action.type).toBe("EXPERIMENT_ADD");
      expect(action.experiment.request_snapshot).toMatchObject({
        domain: "network",
        operation,
        configuration: {
          session_name: "Factory test",
          nodes: [{ errors: { ou_tau_s: 1 } }],
        },
      });
    },
  );
});

function networkConfiguration(): NetworkSessionConfiguration {
  return {
    session_name: "Factory test",
    random_seed: 42,
    sampling_rate_Hz: 100,
    ui_refresh_rate_Hz: 5,
    time_scale: 1,
    buffer_duration_s: 120,
    environment: {
      uniform_field_T: [20e-6, 0, 45e-6],
      gradient_reference_position_m: [0, 0, 0],
      gradient_T_per_m: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
      common_ou_sigma_T: [0, 0, 0],
      common_ou_tau_s: 1,
      dipoles: [],
      gaussian_anomalies: [],
      periodic_fields: [],
    },
    nodes: [{
      sensor_id: "S1",
      role: "sensor",
      profile: "synthetic_magnetometer",
      measurement_mode: "vector",
      position_m: [0, 0, 0],
      orientation_world_to_sensor_wxyz: [1, 0, 0, 0],
      monoaxial_axis_sensor: [1, 0, 0],
      calibration_version: "synthetic-calibration-v1",
      motion: {
        kind: "static",
        velocity_m_per_s: [0, 0, 0],
        angular_rate_rad_per_s: [0, 0, 0],
        modulation_frequency_Hz: [0.31, 0.47, 0.59],
      },
      errors: {
        gain_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        soft_iron_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        cross_axis_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        bias_T: [0, 0, 0],
        deterministic_drift_T_per_s: [0, 0, 0],
        white_noise_std_T_per_sample: [0, 0, 0],
        random_walk_q_T2_per_s: [0, 0, 0],
        ou_sigma_T: [0, 0, 0],
        ou_tau_s: 1,
        initial_temperature_K: 293.15,
        ambient_temperature_K: 293.15,
        temperature_driver: null,
        thermal_time_constant_s: 5,
        thermal_bias_T_per_K: [0, 0, 0],
        reference_temperature_K: 293.15,
        bandwidth_Hz: null,
        saturation_limit_T: 800e-6,
        saturation_limits_T: null,
        clock_offset_s: 0,
        clock_drift_ppm: 0,
        dropout_rate_per_s: 0,
        dropout_duration_s: 0.5,
        stuck_rate_per_s: 0,
        stuck_duration_s: 0.5,
      },
    }],
    events: [],
  };
}
