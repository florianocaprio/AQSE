import type { WorkbenchCapabilities } from "../types/capabilities";
import { requestJson } from "./client";

export function getWorkbenchCapabilities(
  signal?: AbortSignal,
): Promise<WorkbenchCapabilities> {
  return requestJson<WorkbenchCapabilities>("/api/workbench/capabilities", {
    signal,
  });
}
