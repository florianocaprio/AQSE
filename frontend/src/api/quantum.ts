import type {
  CircuitDescription,
  QuantumPreviewRequest,
  QuantumPreviewResponse,
} from "../types/quantum";
import { requestJson } from "./client";

export function getCircuitDescription(
  signal?: AbortSignal,
): Promise<CircuitDescription> {
  return requestJson<CircuitDescription>("/api/quantum/circuit", { signal });
}
export function runQuantumPreview(
  request: QuantumPreviewRequest,
  signal?: AbortSignal,
): Promise<QuantumPreviewResponse> {
  return requestJson<QuantumPreviewResponse>("/api/quantum/preview", {
    method: "POST",
    body: request,
    signal,
  });
}
