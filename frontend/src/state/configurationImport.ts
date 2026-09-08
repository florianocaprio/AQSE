import type {
  DipoleSourceConfiguration,
  EnvironmentConfiguration,
  GaussianAnomalyConfiguration,
  Matrix3,
  NetworkEventConfiguration,
  NetworkSessionConfiguration,
  NodeErrorConfiguration,
  NodeMotionConfiguration,
  PeriodicFieldConfiguration,
  SensorNodeConfiguration,
  TemperatureDriverConfiguration,
} from "../types/network";
import type { Quaternion, Vector3 } from "../types/workbench";

const MEASUREMENT_MODES = ["vector", "monoaxial", "total_field"] as const;
const NODE_ROLES = ["sensor", "remote_reference"] as const;
const MOTION_KINDS = [
  "static",
  "calibration_tumble",
  "high_dynamic",
  "linear_translation",
  "local_anomaly_crossing",
  "combined_stress",
] as const;
const EVENT_KINDS = [
  "world_field_offset",
  "node_bias",
  "node_drift",
  "node_noise_burst",
  "shared_instrument_offset",
  "dropout",
  "stuck",
] as const;
const TEMPERATURE_DRIVER_KINDS = ["constant", "ramp", "sinusoidal"] as const;

export class ConfigurationImportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigurationImportError";
  }
}

export function parseNetworkConfigurationJson(text: string): NetworkSessionConfiguration {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text) as unknown;
  } catch {
    throw new ConfigurationImportError("The selected file is not valid JSON.");
  }
  return parseNetworkConfiguration(parsed);
}

export function parseNetworkConfiguration(value: unknown): NetworkSessionConfiguration {
  const root = record(value, "configuration");
  exactKeys(root, [
    "session_name",
    "random_seed",
    "sampling_rate_Hz",
    "ui_refresh_rate_Hz",
    "time_scale",
    "buffer_duration_s",
    "environment",
    "nodes",
    "events",
  ], "configuration");

  const sessionName = nonemptyString(root.session_name, "configuration.session_name");
  const randomSeed = integer(root.random_seed, "configuration.random_seed", 0, 2 ** 32 - 1);
  const samplingRate = positiveNumber(root.sampling_rate_Hz, "configuration.sampling_rate_Hz", 2_000);
  const refreshRate = positiveNumber(root.ui_refresh_rate_Hz, "configuration.ui_refresh_rate_Hz", 60);
  const timeScale = positiveNumber(root.time_scale, "configuration.time_scale", 100);
  const bufferDuration = positiveNumber(root.buffer_duration_s, "configuration.buffer_duration_s", 600);
  if (refreshRate > samplingRate) {
    fail("configuration.ui_refresh_rate_Hz cannot exceed sampling_rate_Hz.");
  }
  if (Math.round(samplingRate * bufferDuration) > 20_000) {
    fail("The imported sampling rate and buffer duration exceed 20,000 retained frames.");
  }

  const environment = parseEnvironment(root.environment);
  const nodes = array(root.nodes, "configuration.nodes").map((node, index) =>
    parseNode(node, `configuration.nodes[${index}]`),
  );
  if (nodes.length < 1 || nodes.length > 8) {
    fail("configuration.nodes must contain between 1 and 8 sensor nodes.");
  }
  const sensorIds = nodes.map(({ sensor_id }) => sensor_id);
  if (new Set(sensorIds).size !== sensorIds.length) {
    fail("configuration.nodes must use unique sensor_id values.");
  }
  for (const node of nodes) {
    if (node.errors.bandwidth_Hz !== null && node.errors.bandwidth_Hz >= samplingRate / 2) {
      fail(`Node ${node.sensor_id} bandwidth_Hz must be below the Nyquist frequency.`);
    }
    if (
      node.errors.temperature_driver?.kind === "sinusoidal" &&
      node.errors.temperature_driver.sinusoidal_frequency_Hz >= samplingRate / 2
    ) {
      fail(`Node ${node.sensor_id} temperature-driver frequency must be below the Nyquist frequency.`);
    }
  }
  for (const source of environment.periodic_fields) {
    if (source.enabled && source.frequency_Hz >= samplingRate / 2) {
      fail(`Periodic source ${source.source_id} frequency_Hz must be below the Nyquist frequency.`);
    }
  }

  const events = array(root.events, "configuration.events").map((event, index) =>
    parseEvent(event, `configuration.events[${index}]`, new Set(sensorIds)),
  );
  if (events.length > 128) fail("configuration.events may contain at most 128 events.");

  return {
    session_name: sessionName,
    random_seed: randomSeed,
    sampling_rate_Hz: samplingRate,
    ui_refresh_rate_Hz: refreshRate,
    time_scale: timeScale,
    buffer_duration_s: bufferDuration,
    environment,
    nodes,
    events,
  };
}

function parseEnvironment(value: unknown): EnvironmentConfiguration {
  const data = record(value, "configuration.environment");
  exactKeys(data, [
    "uniform_field_T",
    "gradient_reference_position_m",
    "gradient_T_per_m",
    "common_ou_sigma_T",
    "common_ou_tau_s",
    "dipoles",
    "gaussian_anomalies",
    "periodic_fields",
  ], "configuration.environment");
  return {
    uniform_field_T: vector3(data.uniform_field_T, "configuration.environment.uniform_field_T"),
    gradient_reference_position_m: vector3(data.gradient_reference_position_m, "configuration.environment.gradient_reference_position_m"),
    gradient_T_per_m: matrix3(data.gradient_T_per_m, "configuration.environment.gradient_T_per_m"),
    common_ou_sigma_T: nonnegativeVector(data.common_ou_sigma_T, "configuration.environment.common_ou_sigma_T"),
    common_ou_tau_s: positiveNumber(data.common_ou_tau_s, "configuration.environment.common_ou_tau_s"),
    dipoles: array(data.dipoles, "configuration.environment.dipoles").map((item, index) => parseDipole(item, `configuration.environment.dipoles[${index}]`)),
    gaussian_anomalies: array(data.gaussian_anomalies, "configuration.environment.gaussian_anomalies").map((item, index) => parseGaussian(item, `configuration.environment.gaussian_anomalies[${index}]`)),
    periodic_fields: array(data.periodic_fields, "configuration.environment.periodic_fields").map((item, index) => parsePeriodic(item, `configuration.environment.periodic_fields[${index}]`)),
  };
}

function parseDipole(value: unknown, path: string): DipoleSourceConfiguration {
  const data = record(value, path);
  exactKeys(data, ["source_id", "initial_position_m", "velocity_m_per_s", "moment_A_m2", "minimum_distance_m", "enabled"], path);
  return {
    source_id: nonemptyString(data.source_id, `${path}.source_id`),
    initial_position_m: vector3(data.initial_position_m, `${path}.initial_position_m`),
    velocity_m_per_s: vector3(data.velocity_m_per_s, `${path}.velocity_m_per_s`),
    moment_A_m2: vector3(data.moment_A_m2, `${path}.moment_A_m2`),
    minimum_distance_m: positiveNumber(data.minimum_distance_m, `${path}.minimum_distance_m`),
    enabled: boolean(data.enabled, `${path}.enabled`),
  };
}

function parseGaussian(value: unknown, path: string): GaussianAnomalyConfiguration {
  const data = record(value, path);
  exactKeys(data, ["anomaly_id", "peak_amplitude_T", "direction_world", "center_position_m", "spatial_scale_m", "enabled"], path);
  const direction = vector3(data.direction_world, `${path}.direction_world`);
  unitVector(direction, `${path}.direction_world`);
  return {
    anomaly_id: nonemptyString(data.anomaly_id, `${path}.anomaly_id`),
    peak_amplitude_T: positiveNumber(data.peak_amplitude_T, `${path}.peak_amplitude_T`),
    direction_world: direction,
    center_position_m: vector3(data.center_position_m, `${path}.center_position_m`),
    spatial_scale_m: positiveNumber(data.spatial_scale_m, `${path}.spatial_scale_m`),
    enabled: boolean(data.enabled, `${path}.enabled`),
  };
}

function parsePeriodic(value: unknown, path: string): PeriodicFieldConfiguration {
  const data = record(value, path);
  exactKeys(data, ["source_id", "amplitude_world_T", "frequency_Hz", "phase_rad", "enabled"], path);
  return {
    source_id: nonemptyString(data.source_id, `${path}.source_id`),
    amplitude_world_T: vector3(data.amplitude_world_T, `${path}.amplitude_world_T`),
    frequency_Hz: positiveNumber(data.frequency_Hz, `${path}.frequency_Hz`, 1_000),
    phase_rad: finiteNumber(data.phase_rad, `${path}.phase_rad`),
    enabled: boolean(data.enabled, `${path}.enabled`),
  };
}

function parseNode(value: unknown, path: string): SensorNodeConfiguration {
  const data = record(value, path);
  exactKeys(data, [
    "sensor_id",
    "role",
    "profile",
    "measurement_mode",
    "position_m",
    "orientation_world_to_sensor_wxyz",
    "monoaxial_axis_sensor",
    "calibration_version",
    "motion",
    "errors",
  ], path);
  const orientation = quaternion(data.orientation_world_to_sensor_wxyz, `${path}.orientation_world_to_sensor_wxyz`);
  unitVector(orientation, `${path}.orientation_world_to_sensor_wxyz`);
  if (orientation[0] < 0) fail(`${path}.orientation_world_to_sensor_wxyz must use canonical sign w >= 0.`);
  const axis = vector3(data.monoaxial_axis_sensor, `${path}.monoaxial_axis_sensor`);
  unitVector(axis, `${path}.monoaxial_axis_sensor`);
  const profile = enumValue(data.profile, ["synthetic_magnetometer"] as const, `${path}.profile`);
  return {
    sensor_id: nonemptyString(data.sensor_id, `${path}.sensor_id`),
    role: enumValue(data.role, NODE_ROLES, `${path}.role`),
    profile,
    measurement_mode: enumValue(data.measurement_mode, MEASUREMENT_MODES, `${path}.measurement_mode`),
    position_m: vector3(data.position_m, `${path}.position_m`),
    orientation_world_to_sensor_wxyz: orientation,
    monoaxial_axis_sensor: axis,
    calibration_version: nonemptyString(data.calibration_version, `${path}.calibration_version`),
    motion: parseMotion(data.motion, `${path}.motion`),
    errors: parseErrors(data.errors, `${path}.errors`),
  };
}

function parseMotion(value: unknown, path: string): NodeMotionConfiguration {
  const data = record(value, path);
  exactKeys(data, ["kind", "velocity_m_per_s", "angular_rate_rad_per_s", "modulation_frequency_Hz"], path);
  return {
    kind: enumValue(data.kind, MOTION_KINDS, `${path}.kind`),
    velocity_m_per_s: vector3(data.velocity_m_per_s, `${path}.velocity_m_per_s`),
    angular_rate_rad_per_s: vector3(data.angular_rate_rad_per_s, `${path}.angular_rate_rad_per_s`),
    modulation_frequency_Hz: nonnegativeVector(data.modulation_frequency_Hz, `${path}.modulation_frequency_Hz`),
  };
}

function parseErrors(value: unknown, path: string): NodeErrorConfiguration {
  const data: Record<string, unknown> = {
    temperature_driver: null,
    saturation_limits_T: null,
    ...record(value, path),
  };
  exactKeys(data, [
    "gain_matrix", "soft_iron_matrix", "cross_axis_matrix", "bias_T",
    "deterministic_drift_T_per_s", "white_noise_std_T_per_sample",
    "random_walk_q_T2_per_s", "ou_sigma_T", "ou_tau_s",
    "initial_temperature_K", "ambient_temperature_K", "temperature_driver", "thermal_time_constant_s",
    "thermal_bias_T_per_K", "reference_temperature_K", "bandwidth_Hz",
    "saturation_limit_T", "saturation_limits_T", "clock_offset_s", "clock_drift_ppm",
    "dropout_rate_per_s", "dropout_duration_s", "stuck_rate_per_s", "stuck_duration_s",
  ], path);
  const bandwidth = data.bandwidth_Hz === null
    ? null
    : positiveNumber(data.bandwidth_Hz, `${path}.bandwidth_Hz`);
  const temperatureDriver = data.temperature_driver === null
    ? null
    : parseTemperatureDriver(data.temperature_driver, `${path}.temperature_driver`);
  const saturationLimits = data.saturation_limits_T === null
    ? null
    : positiveVector(data.saturation_limits_T, `${path}.saturation_limits_T`);
  if (saturationLimits?.some((value) => value > 10)) fail(`${path}.saturation_limits_T cannot exceed 10 T.`);
  const ambientTemperature = positiveNumber(data.ambient_temperature_K, `${path}.ambient_temperature_K`, 5_000);
  if (temperatureDriver?.kind === "ramp") {
    const finalTemperature = ambientTemperature + temperatureDriver.ramp_rate_K_per_s * temperatureDriver.ramp_duration_s;
    if (finalTemperature <= 0 || finalTemperature > 5_000) fail(`${path}.temperature_driver ramp must remain within (0, 5000] K.`);
  }
  if (temperatureDriver?.kind === "sinusoidal") {
    if (ambientTemperature - temperatureDriver.sinusoidal_amplitude_K <= 0 || ambientTemperature + temperatureDriver.sinusoidal_amplitude_K > 5_000) {
      fail(`${path}.temperature_driver sinusoidal range must remain within (0, 5000] K.`);
    }
  }
  return {
    gain_matrix: matrix3(data.gain_matrix, `${path}.gain_matrix`),
    soft_iron_matrix: matrix3(data.soft_iron_matrix, `${path}.soft_iron_matrix`),
    cross_axis_matrix: matrix3(data.cross_axis_matrix, `${path}.cross_axis_matrix`),
    bias_T: vector3(data.bias_T, `${path}.bias_T`),
    deterministic_drift_T_per_s: vector3(data.deterministic_drift_T_per_s, `${path}.deterministic_drift_T_per_s`),
    white_noise_std_T_per_sample: nonnegativeVector(data.white_noise_std_T_per_sample, `${path}.white_noise_std_T_per_sample`),
    random_walk_q_T2_per_s: nonnegativeVector(data.random_walk_q_T2_per_s, `${path}.random_walk_q_T2_per_s`),
    ou_sigma_T: nonnegativeVector(data.ou_sigma_T, `${path}.ou_sigma_T`),
    ou_tau_s: positiveNumber(data.ou_tau_s, `${path}.ou_tau_s`),
    initial_temperature_K: positiveNumber(data.initial_temperature_K, `${path}.initial_temperature_K`, 5_000),
    ambient_temperature_K: ambientTemperature,
    temperature_driver: temperatureDriver,
    thermal_time_constant_s: positiveNumber(data.thermal_time_constant_s, `${path}.thermal_time_constant_s`),
    thermal_bias_T_per_K: vector3(data.thermal_bias_T_per_K, `${path}.thermal_bias_T_per_K`),
    reference_temperature_K: positiveNumber(data.reference_temperature_K, `${path}.reference_temperature_K`, 5_000),
    bandwidth_Hz: bandwidth,
    saturation_limit_T: positiveNumber(data.saturation_limit_T, `${path}.saturation_limit_T`, 10),
    saturation_limits_T: saturationLimits,
    clock_offset_s: finiteNumber(data.clock_offset_s, `${path}.clock_offset_s`),
    clock_drift_ppm: finiteNumber(data.clock_drift_ppm, `${path}.clock_drift_ppm`),
    dropout_rate_per_s: nonnegativeNumber(data.dropout_rate_per_s, `${path}.dropout_rate_per_s`),
    dropout_duration_s: positiveNumber(data.dropout_duration_s, `${path}.dropout_duration_s`),
    stuck_rate_per_s: nonnegativeNumber(data.stuck_rate_per_s, `${path}.stuck_rate_per_s`),
    stuck_duration_s: positiveNumber(data.stuck_duration_s, `${path}.stuck_duration_s`),
  };
}

function parseTemperatureDriver(value: unknown, path: string): TemperatureDriverConfiguration {
  const data = record(value, path);
  exactKeys(data, [
    "kind",
    "ramp_rate_K_per_s",
    "ramp_duration_s",
    "sinusoidal_amplitude_K",
    "sinusoidal_frequency_Hz",
    "sinusoidal_phase_rad",
  ], path);
  const rampRate = finiteNumber(data.ramp_rate_K_per_s, `${path}.ramp_rate_K_per_s`);
  const amplitude = nonnegativeNumber(data.sinusoidal_amplitude_K, `${path}.sinusoidal_amplitude_K`);
  const phase = finiteNumber(data.sinusoidal_phase_rad, `${path}.sinusoidal_phase_rad`);
  if (Math.abs(rampRate) > 1_000) fail(`${path}.ramp_rate_K_per_s must be between -1000 and 1000.`);
  if (amplitude > 2_500) fail(`${path}.sinusoidal_amplitude_K cannot exceed 2500.`);
  if (Math.abs(phase) > 1_000_000) fail(`${path}.sinusoidal_phase_rad must be between -1000000 and 1000000.`);
  return {
    kind: enumValue(data.kind, TEMPERATURE_DRIVER_KINDS, `${path}.kind`),
    ramp_rate_K_per_s: rampRate,
    ramp_duration_s: positiveNumber(data.ramp_duration_s, `${path}.ramp_duration_s`, 1_000_000_000),
    sinusoidal_amplitude_K: amplitude,
    sinusoidal_frequency_Hz: positiveNumber(data.sinusoidal_frequency_Hz, `${path}.sinusoidal_frequency_Hz`, 1_000),
    sinusoidal_phase_rad: phase,
  };
}

function parseEvent(value: unknown, path: string, knownSensors: Set<string>): NetworkEventConfiguration {
  const data = record(value, path);
  exactKeys(data, ["event_id", "kind", "start_time_s", "duration_s", "target_sensor_ids", "field_offset_T", "drift_rate_T_per_s", "noise_multiplier"], path);
  const kind = enumValue(data.kind, EVENT_KINDS, `${path}.kind`);
  const targets = array(data.target_sensor_ids, `${path}.target_sensor_ids`).map((target, index) => nonemptyString(target, `${path}.target_sensor_ids[${index}]`));
  if (new Set(targets).size !== targets.length) fail(`${path}.target_sensor_ids contains duplicates.`);
  if (targets.some((target) => !knownSensors.has(target))) fail(`${path}.target_sensor_ids contains an unknown sensor.`);
  if (kind === "world_field_offset" && targets.length > 0) fail(`${path}.target_sensor_ids must be empty for world_field_offset.`);
  if (kind !== "world_field_offset" && targets.length === 0) fail(`${path}.target_sensor_ids must identify at least one sensor.`);
  if (kind === "shared_instrument_offset" && targets.length < 2) fail(`${path}.target_sensor_ids must identify at least two sensors for shared_instrument_offset.`);
  const noiseMultiplier = positiveNumber(data.noise_multiplier, `${path}.noise_multiplier`);
  if (kind === "node_noise_burst" && noiseMultiplier <= 1) fail(`${path}.noise_multiplier must exceed 1 for node_noise_burst.`);
  return {
    event_id: nonemptyString(data.event_id, `${path}.event_id`),
    kind,
    start_time_s: nonnegativeNumber(data.start_time_s, `${path}.start_time_s`),
    duration_s: positiveNumber(data.duration_s, `${path}.duration_s`),
    target_sensor_ids: targets,
    field_offset_T: vector3(data.field_offset_T, `${path}.field_offset_T`),
    drift_rate_T_per_s: vector3(data.drift_rate_T_per_s, `${path}.drift_rate_T_per_s`),
    noise_multiplier: noiseMultiplier,
  };
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${path} must be a JSON object.`);
  return value as Record<string, unknown>;
}

function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) fail(`${path} must be a JSON array.`);
  return value;
}

function exactKeys(value: Record<string, unknown>, keys: string[], path: string): void {
  const allowed = new Set(keys);
  const missing = keys.filter((key) => !(key in value));
  const unexpected = Object.keys(value).filter((key) => !allowed.has(key));
  if (missing.length) fail(`${path} is missing required field(s): ${missing.join(", ")}.`);
  if (unexpected.length) fail(`${path} contains unsupported field(s): ${unexpected.join(", ")}.`);
}

function nonemptyString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.trim().length === 0) fail(`${path} must be a non-empty string.`);
  return value;
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") fail(`${path} must be a boolean.`);
  return value;
}

function finiteNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(`${path} must be a finite number.`);
  return value;
}

function positiveNumber(value: unknown, path: string, maximum = Number.POSITIVE_INFINITY): number {
  const parsed = finiteNumber(value, path);
  if (parsed <= 0 || parsed > maximum) fail(`${path} must be greater than 0 and at most ${maximum}.`);
  return parsed;
}

function nonnegativeNumber(value: unknown, path: string): number {
  const parsed = finiteNumber(value, path);
  if (parsed < 0) fail(`${path} cannot be negative.`);
  return parsed;
}

function integer(value: unknown, path: string, minimum: number, maximum: number): number {
  const parsed = finiteNumber(value, path);
  if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) fail(`${path} must be an integer from ${minimum} to ${maximum}.`);
  return parsed;
}

function vector3(value: unknown, path: string): Vector3 {
  const values = array(value, path);
  if (values.length !== 3) fail(`${path} must contain exactly 3 numbers.`);
  return [
    finiteNumber(values[0], `${path}[0]`),
    finiteNumber(values[1], `${path}[1]`),
    finiteNumber(values[2], `${path}[2]`),
  ];
}

function nonnegativeVector(value: unknown, path: string): Vector3 {
  const vector = vector3(value, path);
  if (vector.some((entry) => entry < 0)) fail(`${path} cannot contain negative values.`);
  return vector;
}

function positiveVector(value: unknown, path: string): Vector3 {
  const vector = vector3(value, path);
  if (vector.some((entry) => entry <= 0)) fail(`${path} must contain only positive values.`);
  return vector;
}

function quaternion(value: unknown, path: string): Quaternion {
  const values = array(value, path);
  if (values.length !== 4) fail(`${path} must contain exactly 4 numbers in wxyz order.`);
  return [
    finiteNumber(values[0], `${path}[0]`),
    finiteNumber(values[1], `${path}[1]`),
    finiteNumber(values[2], `${path}[2]`),
    finiteNumber(values[3], `${path}[3]`),
  ];
}

function matrix3(value: unknown, path: string): Matrix3 {
  const rows = array(value, path);
  if (rows.length !== 3) fail(`${path} must contain exactly 3 rows.`);
  return [
    vector3(rows[0], `${path}[0]`),
    vector3(rows[1], `${path}[1]`),
    vector3(rows[2], `${path}[2]`),
  ];
}

function unitVector(value: readonly number[], path: string): void {
  const norm = Math.hypot(...value);
  if (Math.abs(norm - 1) > 1e-9) fail(`${path} must be normalized.`);
}

function enumValue<const Values extends readonly string[]>(
  value: unknown,
  values: Values,
  path: string,
): Values[number] {
  if (typeof value !== "string" || !values.includes(value)) {
    fail(`${path} must be one of: ${values.join(", ")}.`);
  }
  return value as Values[number];
}

function fail(message: string): never {
  throw new ConfigurationImportError(message);
}
