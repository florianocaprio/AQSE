import type {
  FeatureProfile,
  WindowFeatureRecord,
} from "./features";

export type PreviewMode = "self_reference" | "reference_query";
export type QuantumBackend = "qiskit" | "numpy";

export type QuantumPreviewRequest = {
  mode: PreviewMode;
  backend: QuantumBackend;
  feature_profile: FeatureProfile;
  reference_dataset_id: string;
  reference_windows: WindowFeatureRecord[];
  query_windows: WindowFeatureRecord[];
  theta: number[];
};
export type AngleScalerSnapshot = {
  algorithm: string;
  version: string;
  reference_dataset_id: string;
  feature_profile_id: string;
  mean: number[];
  scale: number[];
};

export type KernelDiagnostics = {
  minimum: number;
  maximum: number;
  mean_off_diagonal: number;
  maximum_diagonal_deviation: number;
  maximum_symmetry_deviation: number;
  minimum_eigenvalue: number;
};

export type QuantumPreviewResponse = {
  preview_id: string;
  mode: PreviewMode;
  backend: string;
  feature_profile: FeatureProfile;
  reference_window_ids: string[];
  query_window_ids: string[];
  raw_reference_features: number[][];
  encoded_reference_angles: number[][];
  raw_query_features: number[][];
  encoded_query_angles: number[][];
  executed_theta: number[];
  scaler: AngleScalerSnapshot;
  reference_kernel: number[][];
  query_reference_kernel: number[][] | null;
  diagnostics: KernelDiagnostics;
  execution_duration_ms: number;
  scientific_scope: string;
};

export type CircuitOperation = {
  order: number;
  stage:
    | "feature_upload_1"
    | "alpha"
    | "entanglement"
    | "beta"
    | "feature_upload_2"
    | "unclassified";
  gate: string;
  qubits: number[];
  parameters: string[];
};

export type CircuitDescription = {
  name: string;
  qubits: number;
  input_features: number;
  trainable_parameters: number;
  feature_uploads_per_feature: number;
  trainable_groups: Record<string, string[]>;
  cz_edges: Array<readonly [number, number]>;
  operations: CircuitOperation[];
};
