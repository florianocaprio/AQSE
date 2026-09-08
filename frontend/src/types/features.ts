import type { FeatureVector } from "../types";
import type { SignalChannel, Vector3 } from "./workbench";

export type FeatureWindowConfiguration = {
  duration_s: number;
  overlap_fraction: number;
  minimum_cycles: number;
  minimum_peak_prominence_db: number;
  minimum_snr_db: number;
  reject_clipped: boolean;
};

export type MeasuredVectorSeries = {
  acquisition_id: string;
  sensor_id: string;
  sampling_rate_hz: number;
  time_s: number[];
  measured_field: Vector3[];
  field_unit: "T" | "nT";
  temperature_k: number[];
  saturation_mask: Array<readonly [boolean, boolean, boolean]>;
};

export type FeatureProfile = {
  profile_id: string;
  extractor_version: string;
  sensor_type: string;
  readout: string;
  channel: SignalChannel;
  feature_names: string[];
  feature_units: string[];
  sampling_rate_hz: number;
  window_duration_s: number;
  overlap_fraction: number;
  window_samples: number;
  hop_samples: number;
  phase_reference: string;
  quality_policy_version: string;
  minimum_cycles: number;
  minimum_peak_prominence_db: number;
  minimum_snr_db: number;
  maximum_saturation_fraction: 0;
};

export type FeatureQuality = {
  status: "valid" | "warning" | "invalid";
  valid_for_quantum: boolean;
  flags: string[];
  per_feature_valid: readonly [boolean, boolean, boolean, boolean, boolean, boolean, boolean, boolean];
  sample_count: number;
  saturation_fraction: number;
  cycles_in_window: number;
  peak_prominence_db: number;
  signal_standard_deviation_nt: number;
};

export type WindowFeatureRecord = {
  window_id: string;
  acquisition_id: string;
  sensor_id: string;
  start_index: number;
  end_index: number;
  start_time_s: number;
  end_time_s: number;
  center_time_s: number;
  features: FeatureVector;
  quality: FeatureQuality;
  provenance_token: string;
};

export type FeatureExtractionRequest = {
  series: MeasuredVectorSeries;
  channel: SignalChannel;
  window: FeatureWindowConfiguration;
};

export type FeatureExtractionResponse = {
  profile: FeatureProfile;
  windows: WindowFeatureRecord[];
  valid_window_count: number;
  invalid_window_count: number;
  discarded_sample_count: number;
  warnings: string[];
};
