import type { Quaternion, Vector3 } from "./workbench";

export type Matrix3 = readonly [Vector3, Vector3, Vector3];
export type MeasurementMode = "vector" | "monoaxial" | "total_field";
export type SessionLifecycle = "created" | "running" | "paused" | "stopped";
export type NodeRole = "sensor" | "remote_reference";
export type EventKind =
  | "world_field_offset"
  | "node_bias"
  | "node_drift"
  | "node_noise_burst"
  | "shared_instrument_offset"
  | "dropout"
  | "stuck";
export type QualityFlag =
  | "clipped"
  | "signal_absent"
  | "stuck"
  | "degraded"
  | "clock_error";

export type DipoleSourceConfiguration = {
  source_id: string;
  initial_position_m: Vector3;
  velocity_m_per_s: Vector3;
  moment_A_m2: Vector3;
  minimum_distance_m: number;
  enabled: boolean;
};

export type GaussianAnomalyConfiguration = {
  anomaly_id: string;
  peak_amplitude_T: number;
  direction_world: Vector3;
  center_position_m: Vector3;
  spatial_scale_m: number;
  enabled: boolean;
};

export type PeriodicFieldConfiguration = {
  source_id: string;
  amplitude_world_T: Vector3;
  frequency_Hz: number;
  phase_rad: number;
  enabled: boolean;
};

export type EnvironmentConfiguration = {
  uniform_field_T: Vector3;
  gradient_reference_position_m: Vector3;
  gradient_T_per_m: Matrix3;
  common_ou_sigma_T: Vector3;
  common_ou_tau_s: number;
  dipoles: DipoleSourceConfiguration[];
  gaussian_anomalies: GaussianAnomalyConfiguration[];
  periodic_fields: PeriodicFieldConfiguration[];
};

export type NodeErrorConfiguration = {
  gain_matrix: Matrix3;
  soft_iron_matrix: Matrix3;
  cross_axis_matrix: Matrix3;
  bias_T: Vector3;
  deterministic_drift_T_per_s: Vector3;
  white_noise_std_T_per_sample: Vector3;
  random_walk_q_T2_per_s: Vector3;
  ou_sigma_T: Vector3;
  ou_tau_s: number;
  initial_temperature_K: number;
  ambient_temperature_K: number;
  temperature_driver: TemperatureDriverConfiguration | null;
  thermal_time_constant_s: number;
  thermal_bias_T_per_K: Vector3;
  reference_temperature_K: number;
  bandwidth_Hz: number | null;
  saturation_limit_T: number;
  saturation_limits_T: Vector3 | null;
  clock_offset_s: number;
  clock_drift_ppm: number;
  dropout_rate_per_s: number;
  dropout_duration_s: number;
  stuck_rate_per_s: number;
  stuck_duration_s: number;
};

export type TemperatureDriverConfiguration = {
  kind: "constant" | "ramp" | "sinusoidal";
  ramp_rate_K_per_s: number;
  ramp_duration_s: number;
  sinusoidal_amplitude_K: number;
  sinusoidal_frequency_Hz: number;
  sinusoidal_phase_rad: number;
};

export type NodeMotionConfiguration = {
  kind:
    | "static"
    | "calibration_tumble"
    | "high_dynamic"
    | "linear_translation"
    | "local_anomaly_crossing"
    | "combined_stress";
  velocity_m_per_s: Vector3;
  angular_rate_rad_per_s: Vector3;
  modulation_frequency_Hz: Vector3;
};

export type SensorNodeConfiguration = {
  sensor_id: string;
  role: NodeRole;
  profile: "synthetic_magnetometer";
  measurement_mode: MeasurementMode;
  position_m: Vector3;
  orientation_world_to_sensor_wxyz: Quaternion;
  monoaxial_axis_sensor: Vector3;
  calibration_version: string;
  motion: NodeMotionConfiguration;
  errors: NodeErrorConfiguration;
};

export type NetworkEventConfiguration = {
  event_id: string;
  kind: EventKind;
  start_time_s: number;
  duration_s: number;
  target_sensor_ids: string[];
  field_offset_T: Vector3;
  drift_rate_T_per_s: Vector3;
  noise_multiplier: number;
};

export type NetworkSessionConfiguration = {
  session_name: string;
  random_seed: number;
  sampling_rate_Hz: number;
  ui_refresh_rate_Hz: number;
  time_scale: number;
  buffer_duration_s: number;
  environment: EnvironmentConfiguration;
  nodes: SensorNodeConfiguration[];
  events: NetworkEventConfiguration[];
};

export type SensorReading = {
  sensor_id: string;
  sequence_id: number;
  acquisition_time: string;
  arrival_time: string;
  measurement_mode: MeasurementMode;
  value_T: number | null;
  components_T: Vector3 | null;
  saturation_mask: readonly [boolean, boolean, boolean] | null;
  quality_flags: QualityFlag[];
  valid: boolean;
  observed_temperature_K: number;
  position_m: Vector3;
  orientation_world_to_sensor_wxyz: Quaternion;
  calibration_version: string;
};

export type ObservationFrame = {
  schema_version: "1.0";
  session_id: string;
  frame_id: number;
  configuration_version: number;
  sim_time_s: number;
  readings: SensorReading[];
};

export type FieldTruthAtNode = {
  sensor_id: string;
  position_m: Vector3;
  uniform_field_world_T: Vector3;
  common_field_world_T: Vector3;
  gradient_field_world_T: Vector3;
  dipole_field_world_T: Vector3 | null;
  anomaly_field_world_T: Vector3;
  periodic_field_world_T: Vector3;
  event_field_world_T: Vector3;
  field_true_world_T: Vector3 | null;
  ideal_field_sensor_T: Vector3 | null;
  model_valid: boolean;
};

export type DeviceTruth = {
  sensor_id: string;
  random_walk_bias_T: Vector3;
  correlated_noise_T: Vector3;
  device_temperature_K: number;
  dropout_active: boolean;
  stuck_active: boolean;
  active_instrument_event_ids: string[];
};

export type ActiveCause = {
  event_id: string;
  kind: EventKind;
  target_sensor_ids: string[];
};

export type TruthFrame = {
  schema_version: "1.0";
  session_id: string;
  frame_id: number;
  configuration_version: number;
  sim_time_s: number;
  fields: FieldTruthAtNode[];
  devices: DeviceTruth[];
  active_causes: ActiveCause[];
};

export type BufferStatus = {
  size: number;
  capacity: number;
  oldest_frame_id: number | null;
  newest_frame_id: number | null;
  overwritten_frames: number;
};

export type SessionStatus = {
  schema_version: "1.0";
  session_id: string;
  state: SessionLifecycle;
  configuration_version: number;
  sim_time_s: number;
  latest_frame_id: number;
  created_at: string;
  updated_at: string;
  simulation_lag_s: number;
  buffer: BufferStatus;
};

export type SessionView = {
  status: SessionStatus;
  configuration: NetworkSessionConfiguration;
};

export type FrameBatch = {
  schema_version: "1.0";
  session_id: string;
  from_frame_id: number | null;
  to_frame_id: number | null;
  gap_detected: boolean;
  frames: ObservationFrame[];
  status: SessionStatus;
};

export type TruthFrameBatch = Omit<FrameBatch, "frames"> & {
  frames: TruthFrame[];
};

export type ObservationSnapshot = {
  status: SessionStatus;
  observation: ObservationFrame | null;
};

export type TruthSnapshot = {
  status: SessionStatus;
  truth: TruthFrame | null;
};

export type NetworkHealth = {
  status: "ok";
  service: "AQSE Sensor Network";
  session_store: "in_memory_bounded";
  streaming: "sse";
  active_sessions: number;
  maximum_sessions: number;
};

export type NetworkPresetDescriptor = {
  preset_id: string;
  display_name: string;
  recommended_node_count: number;
  description: string;
};

export type NetworkPresetCatalog = {
  presets: NetworkPresetDescriptor[];
};

export type FieldProviderStatus = {
  provider_id: string;
  display_name: string;
  available: boolean;
  configured: boolean;
  description: string;
};

export type FieldProviderCatalog = {
  providers: FieldProviderStatus[];
};

export type ScheduledEventResponse = {
  event: NetworkEventConfiguration;
  configuration_version: number;
  effective_frame_id: number;
};
