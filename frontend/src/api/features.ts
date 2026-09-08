import type {
  FeatureExtractionRequest,
  FeatureExtractionResponse,
} from "../types/features";
import { requestJson } from "./client";

export function extractVectorMagnetometerFeatures(
  request: FeatureExtractionRequest,
  signal?: AbortSignal,
): Promise<FeatureExtractionResponse> {
  return requestJson<FeatureExtractionResponse>(
    "/api/features/vector-magnetometer/extract",
    {
      method: "POST",
      body: request,
      signal,
    },
  );
}
