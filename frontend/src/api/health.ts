import type {
  HealthResponse,
  QuantumDiagnosticsResponse,
  QuantumHealthResponse,
} from "../types";
import { requestJson } from "./client";

export const POLLED_HEALTH_ENDPOINTS = {
  backend: "/api/health",
  quantum: "/api/quantum/health",
} as const;

export function getBackendHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return requestJson<HealthResponse>(POLLED_HEALTH_ENDPOINTS.backend, { signal });
}

export function getQuantumHealth(signal?: AbortSignal): Promise<QuantumHealthResponse> {
  return requestJson<QuantumHealthResponse>(POLLED_HEALTH_ENDPOINTS.quantum, { signal });
}

export function runQuantumDiagnostics(
  signal?: AbortSignal,
): Promise<QuantumDiagnosticsResponse> {
  return requestJson<QuantumDiagnosticsResponse>("/api/quantum/diagnostics", {
    method: "POST",
    signal,
  });
}
