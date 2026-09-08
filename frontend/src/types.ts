export type HealthResponse = {
  status: string;
  service: string;
};

export type QuantumHealthResponse = {
  status: string;
  engine: string;
  adapter: string;
  qiskit: string;
  qiskit_version: string;
  numpy_reference: string;
  qubits: number;
  features: number;
  trainable_parameters: number;
  simulation: string;
  circuit_metadata: string;
  entangling_edges: number;
};

export type QuantumDiagnosticsResponse = {
  status: string;
  engine: string;
  qiskit: string;
  numpy_reference: string;
  qubits: number;
  features: number;
  trainable_parameters: number;
  simulation: string;
  state_comparison: string;
  execution_duration_ms: number;
};

export type MagnetometerConfiguration = {
  sensor_id: string;
  duration: number;
  sampling_rate: number;
  background_field: number;
  amplitude: number;
  frequency: number;
  phase: number;
  drift_rate: number;
  noise_std: number;
  temperature: number;
  anomaly_enabled: boolean;
  anomaly_time: number | null;
  anomaly_amplitude: number;
  random_seed: number;
};

export type SensorAcquisition = {
  sensor_id: string;
  sensor_type: string;
  timestamp: string;
  sampling_rate: number;
  time: number[];
  signal: number[];
  time_unit: string;
  physical_unit: string;
  configuration: MagnetometerConfiguration;
  metadata: Record<string, string | number | boolean | null>;
  sample_count: number;
};

export type FrequencySpectrum = {
  frequencies: number[];
  power_spectral_density: number[];
  frequency_unit: string;
  power_unit: string;
};

export type FeatureVector = {
  names: string[];
  values: number[];
  units: string[];
};

export type MagnetometerSimulationResponse = {
  acquisition: SensorAcquisition;
  spectrum: FrequencySpectrum;
  features: FeatureVector;
};
