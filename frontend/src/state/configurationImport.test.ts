import { describe, expect, it } from "vitest";

import type { FeatureExtractionResponse } from "../types/features";
import type { QuantumPreviewResponse } from "../types/quantum";
import type { ThetaVector } from "../types/workbench";
import {
  ConfigurationImportError,
  parseNetworkConfigurationJson,
} from "./configurationImport";
import {
  createInitialWorkbenchState,
  featureResultIsStale,
  networkResultIsStale,
  quantumResultIsStale,
  workbenchReducer,
} from "./workbench";

const VALID_CONFIGURATION = {
  session_name: "Imported laboratory network",
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

describe("network configuration import", () => {
  it("decodes a complete network configuration without a blind root cast", () => {
    const configuration = parseNetworkConfigurationJson(
      JSON.stringify(VALID_CONFIGURATION),
    );

    expect(configuration.session_name).toBe("Imported laboratory network");
    expect(configuration.nodes).toHaveLength(1);
    expect(configuration.nodes[0].sensor_id).toBe("S1");
    expect(configuration.environment.uniform_field_T).toEqual([20e-6, 0, 45e-6]);
  });

  it("rejects malformed JSON and configurations outside the 1–8 node boundary", () => {
    expect(() => parseNetworkConfigurationJson("{not-json}"))
      .toThrowError("The selected file is not valid JSON.");
    expect(() => parseNetworkConfigurationJson(JSON.stringify({
      ...VALID_CONFIGURATION,
      nodes: [],
    }))).toThrowError("configuration.nodes must contain between 1 and 8 sensor nodes.");
    expect(() => parseNetworkConfigurationJson(JSON.stringify({
      schema_version: "unsupported-wrapper",
      snapshot: VALID_CONFIGURATION,
    }))).toThrowError(ConfigurationImportError);
  });

  it("accepts disabled periodic sources above Nyquist and enforces shared targets", () => {
    const withDisabledSource = {
      ...VALID_CONFIGURATION,
      environment: {
        ...VALID_CONFIGURATION.environment,
        periodic_fields: [{
          source_id: "disabled-high-frequency",
          amplitude_world_T: [1e-9, 0, 0],
          frequency_Hz: 80,
          phase_rad: 0,
          enabled: false,
        }],
      },
    };
    expect(parseNetworkConfigurationJson(JSON.stringify(withDisabledSource))
      .environment.periodic_fields[0].frequency_Hz).toBe(80);

    const invalidSharedEvent = {
      ...VALID_CONFIGURATION,
      events: [{
        event_id: "shared-1",
        kind: "shared_instrument_offset",
        start_time_s: 1,
        duration_s: 0.5,
        target_sensor_ids: ["S1"],
        field_offset_T: [1e-9, 0, 0],
        drift_rate_T_per_s: [0, 0, 0],
        noise_multiplier: 1,
      }],
    };
    expect(() => parseNetworkConfigurationJson(JSON.stringify(invalidSharedEvent)))
      .toThrowError("configuration.events[0].target_sensor_ids must identify at least two sensors for shared_instrument_offset.");
  });

  it("round-trips axis saturation and temperature-driver configuration", () => {
    const configured = {
      ...VALID_CONFIGURATION,
      nodes: [{
        ...VALID_CONFIGURATION.nodes[0],
        errors: {
          ...VALID_CONFIGURATION.nodes[0].errors,
          saturation_limits_T: [100e-6, 200e-6, 300e-6],
          temperature_driver: {
            kind: "sinusoidal",
            start_time_s: 2,
            ramp_rate_K_per_s: 0,
            ramp_duration_s: 1,
            sinusoidal_amplitude_K: 2,
            sinusoidal_frequency_Hz: 0.2,
            sinusoidal_phase_rad: 0.5,
          },
        },
      }],
    };

    const parsed = parseNetworkConfigurationJson(JSON.stringify(configured));
    expect(parsed.nodes[0].errors.saturation_limits_T).toEqual([100e-6, 200e-6, 300e-6]);
    expect(parsed.nodes[0].errors.temperature_driver).toEqual(configured.nodes[0].errors.temperature_driver);
  });

  it("loads only a new draft and marks dependent executed artifacts stale", () => {
    const configuration = parseNetworkConfigurationJson(
      JSON.stringify(VALID_CONFIGURATION),
    );
    const initial = createInitialWorkbenchState();
    const withDraft = workbenchReducer(initial, {
      type: "NETWORK_DEFAULTS",
      configuration,
    });
    const session = {
      schema_version: "1.0" as const,
      session_id: "session-1",
      state: "paused" as const,
      configuration_version: 1,
      sim_time_s: 3,
      latest_frame_id: 300,
      created_at: "2026-09-08T00:00:00Z",
      updated_at: "2026-09-08T00:00:03Z",
      simulation_lag_s: 0,
      buffer: {
        size: 300,
        capacity: 12_000,
        oldest_frame_id: 1,
        newest_frame_id: 300,
        overwritten_frames: 0,
      },
    };
    const withSession = workbenchReducer(withDraft, {
      type: "NETWORK_SESSION",
      status: session,
      snapshot: {
        id: "network-snapshot-1",
        value: configuration,
        source_revision: withDraft.network.draft?.revision ?? 0,
        executed_at: "2026-09-08T00:00:00Z",
      },
    });
    const featureResult = createFeatureResult();
    const featureLoading = workbenchReducer(withSession, {
      type: "FEATURE_REQUEST",
      status: "loading",
      request_id: "feature-request-1",
    });
    const withFeatures = workbenchReducer(featureLoading, {
      type: "FEATURE_RESULT",
      result: featureResult,
      source_session_id: session.session_id,
      source_data_revision: withSession.network.data_revision,
      snapshot: {
        id: "feature-snapshot-1",
        value: withSession.features.draft.value,
        source_revision: withSession.features.draft.revision,
        executed_at: "2026-09-08T00:00:04Z",
      },
      request_id: "feature-request-1",
      input_artifact_ids: ["network-snapshot-1"],
    });
    const theta = Array.from({ length: 16 }, () => 0) as unknown as ThetaVector;
    const quantumLoading = workbenchReducer(withFeatures, {
      type: "QUANTUM_REQUEST",
      status: "loading",
      request_id: "quantum-request-1",
    });
    const withQuantum = workbenchReducer(quantumLoading, {
      type: "QUANTUM_RESULT",
      preview: createQuantumPreview(featureResult),
      theta_snapshot: {
        id: "theta-snapshot-1",
        value: theta,
        source_revision: withFeatures.quantum.theta_draft.revision,
        executed_at: "2026-09-08T00:00:05Z",
      },
      request_id: "quantum-request-1",
      input_artifact_ids: ["feature-artifact-feature-snapshot-1"],
    });

    expect(networkResultIsStale(withQuantum)).toBe(false);
    expect(featureResultIsStale(withQuantum)).toBe(false);
    expect(quantumResultIsStale(withQuantum)).toBe(false);

    const afterDataReset = workbenchReducer(withQuantum, {
      type: "NETWORK_CLEAR_DATA",
    });
    expect(afterDataReset.features.result).toBe(withQuantum.features.result);
    expect(afterDataReset.quantum.preview).toBe(withQuantum.quantum.preview);
    expect(afterDataReset.network.data_revision).toBe(1);
    expect(featureResultIsStale(afterDataReset)).toBe(true);
    expect(quantumResultIsStale(afterDataReset)).toBe(true);

    const imported = {
      ...configuration,
      session_name: "Replacement draft",
    };
    const next = workbenchReducer(withQuantum, {
      type: "NETWORK_CONFIGURATION_IMPORTED",
      configuration: imported,
    });

    expect(next.network.draft?.value.session_name).toBe("Replacement draft");
    expect(next.network.draft?.revision).toBe(1);
    expect(next.network.session).toBe(session);
    expect(next.network.executed).toBe(withQuantum.network.executed);
    expect(networkResultIsStale(next)).toBe(true);
    expect(featureResultIsStale(next)).toBe(true);
    expect(quantumResultIsStale(next)).toBe(true);
  });
});

function createFeatureResult(): FeatureExtractionResponse {
  return {
    profile: {
      profile_id: "aqse.features.v1",
      extractor_version: "1.0",
      sensor_type: "synthetic_magnetometer",
      readout: "vector",
      channel: "magnitude",
      feature_names: ["amplitude", "phase", "frequency", "variance", "drift", "snr", "spectral_peak", "temperature"],
      feature_units: ["nT", "rad", "Hz", "nT²", "nT/s", "dB", "Hz", "K"],
      sampling_rate_hz: 100,
      window_duration_s: 1,
      overlap_fraction: 0.5,
      window_samples: 100,
      hop_samples: 50,
      phase_reference: "window_start",
      quality_policy_version: "1.0",
      minimum_cycles: 2,
      minimum_peak_prominence_db: 6,
      minimum_snr_db: 0,
      maximum_saturation_fraction: 0,
    },
    windows: [{
      window_id: "window-1",
      acquisition_id: "session-1:S1",
      sensor_id: "S1",
      start_index: 0,
      end_index: 100,
      start_time_s: 0,
      end_time_s: 1,
      center_time_s: 0.5,
      features: {
        names: ["amplitude", "phase", "frequency", "variance", "drift", "snr", "spectral_peak", "temperature"],
        values: [1, 0, 5, 0.1, 0, 20, 5, 293.15],
        units: ["nT", "rad", "Hz", "nT²", "nT/s", "dB", "Hz", "K"],
      },
      quality: {
        status: "valid",
        valid_for_quantum: true,
        flags: [],
        per_feature_valid: [true, true, true, true, true, true, true, true],
        sample_count: 100,
        saturation_fraction: 0,
        cycles_in_window: 5,
        peak_prominence_db: 12,
        signal_standard_deviation_nt: 1,
      },
      provenance_token: "signed-token",
    }],
    valid_window_count: 1,
    invalid_window_count: 0,
    discarded_sample_count: 0,
    warnings: [],
  };
}

function createQuantumPreview(
  featureResult: FeatureExtractionResponse,
): QuantumPreviewResponse {
  const raw = featureResult.windows[0].features.values;
  return {
    preview_id: "preview-1",
    mode: "self_reference",
    backend: "qiskit_statevector",
    feature_profile: featureResult.profile,
    reference_window_ids: ["window-1"],
    query_window_ids: [],
    raw_reference_features: [raw],
    encoded_reference_angles: [[0, 0, 0, 0, 0, 0, 0, 0]],
    raw_query_features: [],
    encoded_query_angles: [],
    executed_theta: Array.from({ length: 16 }, () => 0),
    scaler: {
      algorithm: "standard_scale_to_angle",
      version: "1.0",
      reference_dataset_id: "session-1",
      feature_profile_id: featureResult.profile.profile_id,
      mean: raw,
      scale: [1, 1, 1, 1, 1, 1, 1, 1],
    },
    reference_kernel: [[1]],
    query_reference_kernel: null,
    diagnostics: {
      minimum: 1,
      maximum: 1,
      mean_off_diagonal: 0,
      maximum_diagonal_deviation: 0,
      maximum_symmetry_deviation: 0,
      minimum_eigenvalue: 1,
    },
    execution_duration_ms: 1,
    scientific_scope: "infrastructure preview",
  };
}
