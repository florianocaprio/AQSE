import type { CapabilityStatus } from "./workbench";

export type Capability = {
  status: CapabilityStatus;
  detail: string;
};
export type WorkbenchCapabilities = {
  vector_sensor: Capability;
  continuous_network: Capability;
  quantum_preview: Capability;
  qng_training: Capability;
  local_embedding_afse: Capability;
  neural_model: Capability;
  physical_qpu: Capability;
};
