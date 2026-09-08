import type {
  FrameBatch,
  FieldProviderCatalog,
  NetworkEventConfiguration,
  NetworkHealth,
  NetworkPresetCatalog,
  NetworkSessionConfiguration,
  ObservationSnapshot,
  ScheduledEventResponse,
  SessionView,
  TruthFrameBatch,
  TruthSnapshot,
} from "../types/network";
import { requestJson } from "./client";

const NETWORK_BASE = "/api/network";

export function getNetworkHealth(signal?: AbortSignal): Promise<NetworkHealth> {
  return requestJson<NetworkHealth>(`${NETWORK_BASE}/health`, { signal });
}

export function getNetworkDefaults(
  nodeCount: number,
  signal?: AbortSignal,
): Promise<NetworkSessionConfiguration> {
  const count = Math.min(8, Math.max(1, Math.round(nodeCount)));
  return requestJson<NetworkSessionConfiguration>(
    `${NETWORK_BASE}/defaults?node_count=${count}`,
    { signal },
  );
}

export function getNetworkPresets(
  signal?: AbortSignal,
): Promise<NetworkPresetCatalog> {
  return requestJson<NetworkPresetCatalog>(`${NETWORK_BASE}/presets`, { signal });
}

export function getFieldProviders(signal?: AbortSignal): Promise<FieldProviderCatalog> {
  return requestJson<FieldProviderCatalog>(`${NETWORK_BASE}/field-providers`, { signal });
}

export function getNetworkPresetConfiguration(
  presetId: string,
  signal?: AbortSignal,
): Promise<NetworkSessionConfiguration> {
  return requestJson<NetworkSessionConfiguration>(
    `${NETWORK_BASE}/presets/${encodeURIComponent(presetId)}`,
    { signal },
  );
}

export function createNetworkSession(
  configuration: NetworkSessionConfiguration,
  signal?: AbortSignal,
): Promise<SessionView> {
  return requestJson<SessionView>(`${NETWORK_BASE}/sessions`, {
    method: "POST",
    body: configuration,
    signal,
  });
}

export function getNetworkSession(
  sessionId: string,
  signal?: AbortSignal,
): Promise<SessionView> {
  return requestJson<SessionView>(sessionPath(sessionId), { signal });
}

export function controlNetworkSession(
  sessionId: string,
  action: "start" | "pause" | "resume" | "stop" | "reset" | "replay",
  signal?: AbortSignal,
): Promise<SessionView> {
  return requestJson<SessionView>(`${sessionPath(sessionId)}/${action}`, {
    method: "POST",
    signal,
  });
}

export function stepNetworkSession(
  sessionId: string,
  frames: number,
  signal?: AbortSignal,
): Promise<FrameBatch> {
  return requestJson<FrameBatch>(`${sessionPath(sessionId)}/step`, {
    method: "POST",
    body: { frames },
    signal,
  });
}

export function getObservationSnapshot(
  sessionId: string,
  signal?: AbortSignal,
): Promise<ObservationSnapshot> {
  return requestJson<ObservationSnapshot>(`${sessionPath(sessionId)}/snapshot`, {
    signal,
  });
}

export function getObservationFrames(
  sessionId: string,
  afterFrameId: number | null,
  limit = 1000,
  signal?: AbortSignal,
): Promise<FrameBatch> {
  return requestJson<FrameBatch>(
    framePath(sessionId, "frames", afterFrameId, limit),
    { signal },
  );
}

export function getTruthSnapshot(
  sessionId: string,
  signal?: AbortSignal,
): Promise<TruthSnapshot> {
  return requestJson<TruthSnapshot>(`${sessionPath(sessionId)}/truth/snapshot`, {
    signal,
  });
}

export function getTruthFrames(
  sessionId: string,
  afterFrameId: number | null,
  limit = 1000,
  signal?: AbortSignal,
): Promise<TruthFrameBatch> {
  return requestJson<TruthFrameBatch>(
    framePath(sessionId, "truth/frames", afterFrameId, limit),
    { signal },
  );
}

export function scheduleNetworkEvent(
  sessionId: string,
  event: NetworkEventConfiguration,
  signal?: AbortSignal,
): Promise<ScheduledEventResponse> {
  return requestJson<ScheduledEventResponse>(`${sessionPath(sessionId)}/events`, {
    method: "POST",
    body: event,
    signal,
  });
}

export function deleteNetworkSession(
  sessionId: string,
  signal?: AbortSignal,
): Promise<void> {
  return requestJson<void>(sessionPath(sessionId), { method: "DELETE", signal });
}

/**
 * Best-effort page-lifecycle cleanup. `keepalive` lets the browser transmit the
 * small DELETE request while the document is being discarded.
 */
export function deleteNetworkSessionKeepalive(sessionId: string): void {
  void fetch(sessionPath(sessionId), {
    method: "DELETE",
    keepalive: true,
  }).catch(() => undefined);
}

export function networkStreamUrl(sessionId: string, afterFrameId = 0): string {
  return `${sessionPath(sessionId)}/stream?after_frame_id=${Math.max(0, Math.round(afterFrameId))}`;
}

function sessionPath(sessionId: string): string {
  return `${NETWORK_BASE}/sessions/${encodeURIComponent(sessionId)}`;
}

function framePath(
  sessionId: string,
  suffix: string,
  afterFrameId: number | null,
  limit: number,
): string {
  const query = new URLSearchParams({ limit: String(limit) });
  if (afterFrameId !== null) query.set("after_frame_id", String(afterFrameId));
  return `${sessionPath(sessionId)}/${suffix}?${query.toString()}`;
}
