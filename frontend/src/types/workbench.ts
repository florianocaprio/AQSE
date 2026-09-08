import type { MagnetometerConfiguration } from "../types";
import type { FeatureWindowConfiguration as ApiFeatureWindowConfiguration } from "./features";
import type { NetworkSessionConfiguration } from "./network";
import type { PreviewMode, QuantumBackend } from "./quantum";

export const WORKSHEET_IDS = [
  "overview",
  "sensors",
  "features",
  "quantum",
  "qng",
  "afse",
  "neural",
  "experiments",
] as const;

export type WorksheetId = (typeof WORKSHEET_IDS)[number];

export type OperationStatus =
  | "idle"
  | "loading"
  | "ready"
  | "running"
  | "paused"
  | "stopped"
  | "failed"
  | "unavailable";

export type CapabilityStatus =
  | "implemented"
  | "available_not_connected"
  | "architecture_defined"
  | "not_implemented";

export type WarningSeverity = "info" | "warning" | "error";

export type WarningRecord = {
  id: string;
  source: "system" | "sensor" | "features" | "quantum";
  severity: WarningSeverity;
  message: string;
  created_at: string;
};

export type RevisionedDraft<T> = {
  value: T;
  revision: number;
};

export type ExecutedSnapshot<T> = {
  id: string;
  value: Readonly<T>;
  source_revision: number;
  executed_at: string;
};

export type ArtifactProvenance = {
  experiment_id: string;
  artifact_id: string;
  created_at: string;
  input_artifact_ids: string[];
  configuration_snapshot_id?: string;
  feature_profile_id?: string;
  theta_snapshot_id?: string;
  scaler_version?: string;
};

export type ComputedArtifact<T> = {
  data: T;
  provenance: ArtifactProvenance;
  duration_ms?: number;
  warnings: WarningRecord[];
};

export type Vector3 = readonly [number, number, number];
export type Quaternion = readonly [number, number, number, number];
export type ThetaVector = readonly [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

export type SignalChannel = "x" | "y" | "z" | "magnitude";

export type FeatureWindowConfiguration = {
  window_duration_s: number;
  hop_duration_s: number;
  selected_channel: SignalChannel;
  feature_profile_id: string;
};

export type FeatureQuality = {
  valid: boolean;
  flags: string[];
};

export type FeatureWindow = {
  window_id: string;
  sensor_id: string;
  start_time_s: number;
  end_time_s: number;
  names: string[];
  values: number[];
  units: string[];
  quality: FeatureQuality;
};

export type FeatureBatch = {
  schema_version: string;
  feature_profile_id: string;
  source_session_id: string;
  windows: FeatureWindow[];
  valid_window_count: number;
  invalid_window_count: number;
};

export type FittedAngleScaler = {
  version: string;
  reference_dataset_id: string;
  feature_profile_id: string;
  mean: number[];
  scale: number[];
};

export type KernelDiagnostics = {
  dimensions: readonly [number, number];
  minimum: number;
  maximum: number;
  mean_off_diagonal_similarity: number;
  diagonal_deviation: number;
};

export type QuantumPreview = {
  preview_id: string;
  backend: string;
  feature_profile_id: string;
  theta: number[];
  encoded_angles: number[][];
  kernel: number[][];
  diagnostics: KernelDiagnostics;
  scaler: FittedAngleScaler;
  execution_duration_ms: number;
};

export type ExperimentRequestSnapshot =
  | {
      domain: "network";
      operation: "create" | "start" | "pause" | "resume" | "stop" | "reset" | "replay" | "step" | "delete";
      session_id: string | null;
      requested_frames?: number;
      configuration: NetworkSessionConfiguration | null;
    }
  | {
      domain: "features";
      source_session_id: string;
      source_data_revision: number;
      sensor_id: string;
      channel: SignalChannel;
      configuration: ApiFeatureWindowConfiguration;
    }
  | {
      domain: "quantum";
      mode: PreviewMode;
      backend: QuantumBackend;
      preview_size: number;
      reference_dataset_id: string;
      reference_window_ids: string[];
      theta: number[];
    }
  | {
      domain: "scalar_simulation";
      configuration: MagnetometerConfiguration;
    };

export type ExperimentRecord = {
  id: string;
  created_at: string;
  kind: "scalar_simulation" | "network_session" | "feature_extraction" | "quantum_preview";
  status: "completed" | "failed" | "stopped";
  title: string;
  sensor_type?: string;
  scenario?: string;
  seed?: number;
  sample_count?: number;
  feature_profile_id?: string;
  valid_window_count?: number;
  theta_snapshot_id?: string;
  scaler_version?: string;
  request_snapshot: ExperimentRequestSnapshot;
  warnings: string[];
  timings_ms: Record<string, number>;
};

export type ServiceHealth = {
  status: "checking" | "ready" | "unavailable";
  detail?: string;
  checked_at?: string;
};
