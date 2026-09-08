import { useCallback, useEffect, useRef } from "react";

import { ApiError, errorMessage, requestJson } from "../api/client";
import { extractVectorMagnetometerFeatures } from "../api/features";
import {
  controlNetworkSession,
  createNetworkSession,
  deleteNetworkSession,
  deleteNetworkSessionKeepalive,
  getNetworkDefaults,
  getFieldProviders,
  getNetworkHealth,
  getNetworkPresets,
  getNetworkPresetConfiguration,
  getNetworkSession,
  getTruthFrames,
  networkStreamUrl,
  stepNetworkSession,
} from "../api/network";
import { getCircuitDescription, runQuantumPreview } from "../api/quantum";
import { getWorkbenchCapabilities } from "../api/workbench";
import type { QuantumHealthResponse, HealthResponse } from "../types";
import type {
  FrameBatch,
  NetworkSessionConfiguration,
  SessionLifecycle,
  SessionView,
} from "../types/network";
import type {
  ExperimentRecord,
  ExperimentRequestSnapshot,
  ThetaVector,
} from "../types/workbench";
import { latestContiguousVectorSeries } from "./selectors";
import { useWorkbench } from "./workbench";

type SessionControl = "start" | "pause" | "resume" | "stop" | "reset" | "replay";
type NetworkRequestSnapshot = Extract<ExperimentRequestSnapshot, { domain: "network" }>;
type NetworkExperimentInput = {
  id: string;
  created_at: string;
  operation: NetworkRequestSnapshot["operation"];
  session_id: string | null;
  configuration: NetworkSessionConfiguration | Readonly<NetworkSessionConfiguration> | null;
  status: ExperimentRecord["status"];
  title: string;
  sample_count?: number;
  requested_frames?: number;
  warnings?: string[];
};
type StreamIdentity = {
  session_id: string;
  stream_revision: number;
};
const HEALTH_POLL_INTERVAL_MS = 15_000;

let pendingUnmountCleanup: { sessionId: string; timer: number } | null = null;

export function useWorkbenchBootstrap() {
  const { dispatch } = useWorkbench();

  useEffect(() => {
    let controller: AbortController | null = null;
    const pollHealth = () => {
      controller?.abort();
      controller = new AbortController();
      const signal = controller.signal;
      const checkedAt = () => new Date().toISOString();

      void requestJson<HealthResponse>("/api/health", { signal })
        .then((response) => {
          dispatch({
            type: "BACKEND_HEALTH",
            health: {
              status: response.status === "ok" ? "ready" : "unavailable",
              detail: response.service,
              checked_at: checkedAt(),
            },
          });
        })
        .catch((error: unknown) => {
          if (!isAbort(error)) {
            dispatch({
              type: "BACKEND_HEALTH",
              health: {
                status: "unavailable",
                detail: errorMessage(error, "Backend health check failed."),
                checked_at: checkedAt(),
              },
            });
          }
        });

      void requestJson<QuantumHealthResponse>("/api/quantum/health", { signal })
        .then((response) => {
          dispatch({
            type: "QUANTUM_HEALTH",
            health: {
              status: response.status === "ok" ? "ready" : "unavailable",
              detail: `${response.engine} · ${response.simulation}`,
              checked_at: checkedAt(),
            },
          });
        })
        .catch((error: unknown) => {
          if (!isAbort(error)) {
            dispatch({
              type: "QUANTUM_HEALTH",
              health: {
                status: "unavailable",
                detail: errorMessage(error, "Quantum infrastructure check failed."),
                checked_at: checkedAt(),
              },
            });
          }
        });

      void getNetworkHealth(signal)
        .then((detail) => {
          dispatch({
            type: "NETWORK_HEALTH",
            health: {
              status: "ready",
              detail: `${detail.streaming} · ${detail.session_store}`,
              checked_at: checkedAt(),
            },
            detail,
          });
        })
        .catch((error: unknown) => {
          if (!isAbort(error)) {
            dispatch({
              type: "NETWORK_HEALTH",
              health: {
                status: "unavailable",
                detail: errorMessage(error, "Continuous network service unavailable."),
                checked_at: checkedAt(),
              },
            });
          }
        });
    };

    pollHealth();
    const interval = window.setInterval(pollHealth, HEALTH_POLL_INTERVAL_MS);
    return () => {
      window.clearInterval(interval);
      controller?.abort();
    };
  }, [dispatch]);

  useEffect(() => {
    const controller = new AbortController();

    void getWorkbenchCapabilities(controller.signal)
      .then((capabilities) => dispatch({ type: "CAPABILITIES_READY", capabilities }))
      .catch((error: unknown) => {
        if (!isAbort(error)) {
          dispatch({
            type: "CAPABILITIES_FAILED",
            message: errorMessage(error, "Capability registry unavailable."),
          });
        }
      });

    void getNetworkDefaults(4, controller.signal)
      .then((configuration) => dispatch({ type: "NETWORK_DEFAULTS", configuration }))
      .catch((error: unknown) => {
        if (!isAbort(error)) {
          dispatch({
            type: "NETWORK_REQUEST",
            status: "failed",
            error: errorMessage(error, "Could not load network defaults."),
          });
        }
      });

    void getNetworkPresets(controller.signal)
      .then(({ presets }) => dispatch({ type: "NETWORK_PRESETS", presets }))
      .catch(() => undefined);

    void getFieldProviders(controller.signal)
      .then(({ providers }) => dispatch({ type: "NETWORK_FIELD_PROVIDERS", providers }))
      .catch(() => undefined);

    void getCircuitDescription(controller.signal)
      .then((circuit) => dispatch({ type: "QUANTUM_CIRCUIT", circuit }))
      .catch(() => undefined);

    return () => controller.abort();
  }, [dispatch]);
}

export function useNetworkStream() {
  const { state, dispatch } = useWorkbench();
  const sessionId = state.network.session?.session_id ?? null;
  const streamRevision = state.network.stream_revision;
  const blindModeRef = useRef(state.network.blind_mode);
  const lastObservationId = useRef(0);
  const lastStreamIdentity = useRef<StreamIdentity | null>(null);
  blindModeRef.current = state.network.blind_mode;
  lastObservationId.current = state.network.observations.at(-1)?.frame_id ?? 0;
  useSessionLifecycleCleanup(sessionId);

  useEffect(() => {
    if (!sessionId) return;

    const streamIdentity = {
      session_id: sessionId,
      stream_revision: streamRevision,
    };
    const streamCursor = networkStreamCursor(
      lastStreamIdentity.current,
      streamIdentity,
      lastObservationId.current,
    );
    lastStreamIdentity.current = streamIdentity;
    dispatch({
      type: "NETWORK_STREAM",
      status: "connecting",
      stream_revision: streamRevision,
    });
    const stream = new EventSource(networkStreamUrl(sessionId, streamCursor));
    let closed = false;
    let sessionCheckInFlight = false;

    const closeStream = () => {
      if (closed) return;
      closed = true;
      stream.close();
      dispatch({
        type: "NETWORK_STREAM",
        status: "disconnected",
        stream_revision: streamRevision,
      });
    };

    const onOpen = () => {
      if (!closed) {
        dispatch({
          type: "NETWORK_STREAM",
          status: "connected",
          stream_revision: streamRevision,
        });
      }
    };
    const onObservations = (event: MessageEvent<string>) => {
      try {
        const batch = JSON.parse(event.data) as FrameBatch;
        dispatch({
          type: "NETWORK_SESSION",
          status: batch.status,
          stream_revision: streamRevision,
        });
        dispatch({
          type: "NETWORK_OBSERVATIONS",
          frames: batch.frames,
          gap_detected: batch.gap_detected,
          stream_revision: streamRevision,
        });

        if (!blindModeRef.current && batch.from_frame_id !== null) {
          const beforeFirst = Math.max(0, batch.from_frame_id - 1);
          void getTruthFrames(
            sessionId,
            beforeFirst,
            Math.max(1, batch.frames.length),
          )
            .then((truth) => {
              dispatch({
                type: "NETWORK_TRUTH",
                frames: truth.frames,
                gap_detected: truth.gap_detected,
                stream_revision: streamRevision,
              });
            })
            .catch((error: unknown) => {
              dispatch({
                type: "WARNING_ADD",
                warning: warning("sensor", errorMessage(error, "Truth channel read failed.")),
              });
            });
        }
      } catch {
        dispatch({
          type: "WARNING_ADD",
          warning: warning("sensor", "A malformed sensor stream event was ignored."),
        });
      }
    };
    const onSessionStatus = (event: MessageEvent<string>) => {
      try {
        const view = JSON.parse(event.data) as SessionView["status"];
        dispatch({
          type: "NETWORK_SESSION",
          status: view,
          stream_revision: streamRevision,
        });
        if (view.state === "stopped") {
          closeStream();
        }
      } catch {
        // An invalid terminal notification must not corrupt collected observations.
      }
    };
    const onError = () => {
      if (closed) return;
      dispatch({
        type: "NETWORK_STREAM",
        status: "error",
        stream_revision: streamRevision,
      });
      if (sessionCheckInFlight) return;
      sessionCheckInFlight = true;
      void getNetworkSession(sessionId)
        .then((view) => {
          if (closed) return;
          dispatch({
            type: "NETWORK_SESSION",
            status: view.status,
            stream_revision: streamRevision,
          });
          if (view.status.state === "stopped") closeStream();
        })
        .catch((error: unknown) => {
          if (closed) return;
          if (isNotFound(error)) {
            closeStream();
            dispatch({ type: "NETWORK_SESSION_CLEAR" });
          }
        })
        .finally(() => {
          sessionCheckInFlight = false;
        });
    };

    stream.addEventListener("open", onOpen);
    stream.addEventListener("observations", onObservations as EventListener);
    stream.addEventListener("session_status", onSessionStatus as EventListener);
    stream.addEventListener("error", onError);

    return () => {
      closeStream();
    };
  }, [dispatch, sessionId, streamRevision]);
}

export function useWorkbenchActions() {
  const { state, dispatch } = useWorkbench();

  const loadNetworkDefaults = useCallback(
    async (nodeCount: number) => {
      dispatch({ type: "NETWORK_REQUEST", status: "loading" });
      try {
        const configuration = await getNetworkDefaults(nodeCount);
        dispatch({ type: "NETWORK_DEFAULTS", configuration });
        dispatch({ type: "NETWORK_REQUEST", status: "idle" });
      } catch (error: unknown) {
        dispatch({
          type: "NETWORK_REQUEST",
          status: "failed",
          error: errorMessage(error, "Could not load network defaults."),
        });
      }
    },
    [dispatch],
  );

  const loadNetworkPreset = useCallback(
    async (presetId: string) => {
      dispatch({ type: "NETWORK_REQUEST", status: "loading" });
      try {
        const configuration = await getNetworkPresetConfiguration(presetId);
        dispatch({ type: "NETWORK_DEFAULTS", configuration });
        dispatch({ type: "NETWORK_REQUEST", status: "idle" });
      } catch (error: unknown) {
        dispatch({
          type: "NETWORK_REQUEST",
          status: "failed",
          error: errorMessage(error, "Could not load the selected network preset."),
        });
      }
    },
    [dispatch],
  );

  const createSession = useCallback(async () => {
    const draft = state.network.draft;
    if (!draft) return;
    dispatch({ type: "NETWORK_REQUEST", status: "loading" });
    dispatch({ type: "NETWORK_CLEAR_DATA" });
    try {
      const view = await createNetworkSession(draft.value);
      dispatch({
        type: "NETWORK_SESSION",
        status: view.status,
        snapshot: snapshot(view.configuration, draft.revision, "network"),
      });
      dispatch(
        createNetworkExperimentAction({
          id: uniqueId("experiment"),
          created_at: new Date().toISOString(),
          operation: "create",
          session_id: view.status.session_id,
          configuration: draft.value,
          status: "completed",
          title: `Session created · ${view.configuration.session_name}`,
          sample_count: view.status.latest_frame_id,
        }),
      );
    } catch (error: unknown) {
      dispatch({
        type: "NETWORK_REQUEST",
        status: "failed",
        error: errorMessage(error, "Network session creation failed."),
      });
    }
  }, [dispatch, state.network.draft]);

  const controlSession = useCallback(
    async (action: SessionControl) => {
      const current = state.network.session;
      if (!current) return;
      dispatch({ type: "NETWORK_REQUEST", status: "loading" });
      try {
        const view = await controlNetworkSession(current.session_id, action);
        dispatch({ type: "NETWORK_SESSION", status: view.status });
        if (shouldReconnectAfterControl(action, current.state)) {
          dispatch({ type: "NETWORK_STREAM_RECONNECT" });
          dispatch({ type: "NETWORK_CLEAR_DATA" });
        }
        if (action === "reset" || action === "replay") {
          emitNetworkExperiment(dispatch, {
            id: uniqueId("experiment"),
            created_at: new Date().toISOString(),
            operation: action,
            session_id: current.session_id,
            configuration: view.configuration,
            status: "completed",
            title: `${action === "reset" ? "Session reset" : "Same realization replayed"} · ${view.configuration.session_name}`,
            sample_count: view.status.latest_frame_id,
          });
        }
        if (action === "stop") {
          dispatch(
            createNetworkExperimentAction({
              id: uniqueId("experiment"),
              created_at: new Date().toISOString(),
              operation: "stop",
              session_id: current.session_id,
              configuration: view.configuration,
              status: "stopped",
              title: view.configuration.session_name,
              sample_count: view.status.latest_frame_id,
              warnings: state.network.gap_detected ? ["A retained-buffer gap was detected."] : [],
            }),
          );
        }
      } catch (error: unknown) {
        dispatch({
          type: "NETWORK_REQUEST",
          status: "failed",
          error: errorMessage(error, `Network ${action} request failed.`),
        });
      }
    },
    [dispatch, state.network.gap_detected, state.network.session],
  );

  const discardSession = useCallback(async () => {
    const current = state.network.session;
    if (!current || current.state === "running" || current.state === "paused") return;
    const configuration = state.network.executed?.value ?? state.network.draft?.value ?? null;
    dispatch({ type: "NETWORK_REQUEST", status: "loading" });
    try {
      await deleteNetworkSession(current.session_id);
      emitNetworkExperiment(dispatch, {
        id: uniqueId("experiment"),
        created_at: new Date().toISOString(),
        operation: "delete",
        session_id: current.session_id,
        configuration,
        status: "completed",
        title: `Session deleted · ${configuration?.session_name ?? current.session_id}`,
        sample_count: current.latest_frame_id,
      });
      dispatch({ type: "NETWORK_SESSION_CLEAR" });
    } catch (error: unknown) {
      if (isNotFound(error)) {
        emitNetworkExperiment(dispatch, {
          id: uniqueId("experiment"),
          created_at: new Date().toISOString(),
          operation: "delete",
          session_id: current.session_id,
          configuration,
          status: "completed",
          title: `Session already absent · ${configuration?.session_name ?? current.session_id}`,
          sample_count: current.latest_frame_id,
          warnings: ["The backend returned 404; stale local session state was cleared."],
        });
        dispatch({ type: "NETWORK_SESSION_CLEAR" });
        return;
      }
      dispatch({
        type: "NETWORK_REQUEST",
        status: "failed",
        error: errorMessage(error, "Session deletion failed."),
      });
    }
  }, [dispatch, state.network.draft, state.network.executed, state.network.session]);

  const stepSession = useCallback(async () => {
    const current = state.network.session;
    if (!current) return;
    dispatch({ type: "NETWORK_REQUEST", status: "loading" });
    try {
      const batch = await stepNetworkSession(current.session_id, 1);
      dispatch({ type: "NETWORK_SESSION", status: batch.status });
      dispatch({
        type: "NETWORK_OBSERVATIONS",
        frames: batch.frames,
        gap_detected: batch.gap_detected,
      });
      const configuration = state.network.executed?.value ?? state.network.draft?.value ?? null;
      dispatch(
        createNetworkExperimentAction({
          id: uniqueId("experiment"),
          created_at: new Date().toISOString(),
          operation: "step",
          session_id: current.session_id,
          configuration,
          status: "completed",
          title: `Single step · ${configuration?.session_name ?? current.session_id}`,
          sample_count: batch.frames.length,
          requested_frames: 1,
          warnings: batch.gap_detected ? ["The bounded buffer reported a frame gap."] : [],
        }),
      );
      if (!state.network.blind_mode && batch.from_frame_id !== null) {
        try {
          const truth = await getTruthFrames(
            current.session_id,
            Math.max(0, batch.from_frame_id - 1),
            Math.max(1, batch.frames.length),
          );
          dispatch({
            type: "NETWORK_TRUTH",
            frames: truth.frames,
            gap_detected: truth.gap_detected,
          });
        } catch (error: unknown) {
          dispatch({
            type: "WARNING_ADD",
            warning: warning("sensor", errorMessage(error, "Truth channel read failed.")),
          });
        }
      }
    } catch (error: unknown) {
      dispatch({
        type: "NETWORK_REQUEST",
        status: "failed",
        error: errorMessage(error, "Single-step request failed."),
      });
    }
  }, [
    dispatch,
    state.network.blind_mode,
    state.network.draft,
    state.network.executed,
    state.network.session,
  ]);

  const setBlindMode = useCallback(
    async (enabled: boolean) => {
      dispatch({ type: "NETWORK_BLIND_MODE", enabled });
      const session = state.network.session;
      if (enabled || !session || session.latest_frame_id === 0) return;
      try {
        const startAfter = Math.max(0, session.latest_frame_id - 1_000);
        const truth = await getTruthFrames(session.session_id, startAfter, 1_000);
        dispatch({
          type: "NETWORK_TRUTH",
          frames: truth.frames,
          gap_detected: truth.gap_detected,
        });
      } catch (error: unknown) {
        dispatch({
          type: "WARNING_ADD",
          warning: warning("sensor", errorMessage(error, "Truth channel read failed.")),
        });
      }
    },
    [dispatch, state.network.session],
  );

  const extractFeatures = useCallback(async () => {
    const sensorId = state.network.selected_node_id;
    const config = state.network.executed?.value ?? state.network.draft?.value;
    const sessionId = state.network.session?.session_id;
    if (!sensorId || !config || !sessionId) return;

    const series = latestContiguousVectorSeries(
      state.network.observations,
      sensorId,
      config.sampling_rate_Hz,
    );
    if (!series) {
      dispatch({
        type: "FEATURE_REQUEST",
        status: "failed",
        error: "At least 16 contiguous valid vector observations are required.",
      });
      return;
    }

    const draft = state.features.draft;
    const sourceDataRevision = state.network.data_revision;
    const requiredSamples = Math.max(
      16,
      Math.round(draft.value.duration_s * config.sampling_rate_Hz),
    );
    if (series.time_s.length < requiredSamples) {
      dispatch({
        type: "FEATURE_REQUEST",
        status: "failed",
        error: `${requiredSamples} contiguous samples are required by the current window duration.`,
      });
      return;
    }
    dispatch({ type: "FEATURE_REQUEST", status: "loading" });
    try {
      const result = await extractVectorMagnetometerFeatures({
        series,
        channel: state.features.channel,
        window: draft.value,
      });
      dispatch({
        type: "FEATURE_RESULT",
        result,
        source_session_id: sessionId,
        source_data_revision: sourceDataRevision,
        snapshot: snapshot(draft.value, draft.revision, "features"),
      });
      dispatch({
        type: "EXPERIMENT_ADD",
        experiment: {
          id: uniqueId("experiment"),
          created_at: new Date().toISOString(),
          kind: "feature_extraction",
          status: "completed",
          title: `Feature extraction · ${sensorId}`,
          sensor_type: "synthetic_magnetometer",
          sample_count: series.time_s.length,
          feature_profile_id: result.profile.profile_id,
          valid_window_count: result.valid_window_count,
          request_snapshot: {
            domain: "features",
            source_session_id: sessionId,
            source_data_revision: sourceDataRevision,
            sensor_id: sensorId,
            channel: state.features.channel,
            configuration: structuredClone(draft.value),
          },
          warnings: result.warnings,
          timings_ms: {},
        },
      });
    } catch (error: unknown) {
      dispatch({
        type: "FEATURE_REQUEST",
        status: "failed",
        error: errorMessage(error, "Feature extraction failed."),
      });
    }
  }, [dispatch, state.features.channel, state.features.draft, state.network]);

  const previewQuantum = useCallback(async () => {
    const result = state.features.result;
    if (!result || !state.features.source_session_id) return;
    const referenceWindows = result.windows
      .filter(({ quality }) => quality.valid_for_quantum)
      .slice(0, state.quantum.preview_size);
    if (referenceWindows.length < 2) {
      dispatch({
        type: "QUANTUM_REQUEST",
        status: "failed",
        error: "At least two feature windows marked valid_for_quantum are required.",
      });
      return;
    }

    const thetaDraft = state.quantum.theta_draft;
    dispatch({ type: "QUANTUM_REQUEST", status: "loading" });
    try {
      const preview = await runQuantumPreview({
        mode: "self_reference",
        backend: state.quantum.backend,
        feature_profile: result.profile,
        reference_dataset_id: state.features.source_session_id,
        reference_windows: referenceWindows,
        query_windows: [],
        theta: [...thetaDraft.value],
      });
      const thetaSnapshot = snapshot(
        thetaDraft.value,
        thetaDraft.revision,
        "theta",
      );
      dispatch({ type: "QUANTUM_RESULT", preview, theta_snapshot: thetaSnapshot });
      dispatch({
        type: "EXPERIMENT_ADD",
        experiment: {
          id: uniqueId("experiment"),
          created_at: new Date().toISOString(),
          kind: "quantum_preview",
          status: "completed",
          title: `Fixed-theta TQK preview · ${preview.reference_window_ids.length} windows`,
          feature_profile_id: preview.feature_profile.profile_id,
          valid_window_count: preview.reference_window_ids.length,
          theta_snapshot_id: thetaSnapshot.id,
          scaler_version: preview.scaler.version,
          request_snapshot: {
            domain: "quantum",
            mode: "self_reference",
            backend: state.quantum.backend,
            preview_size: state.quantum.preview_size,
            reference_dataset_id: state.features.source_session_id,
            reference_window_ids: referenceWindows.map(({ window_id }) => window_id),
            theta: [...thetaDraft.value],
          },
          warnings: [],
          timings_ms: { quantum_preview: preview.execution_duration_ms },
        },
      });
    } catch (error: unknown) {
      dispatch({
        type: "QUANTUM_REQUEST",
        status: "failed",
        error: errorMessage(error, "Quantum preview failed."),
      });
    }
  }, [dispatch, state.features, state.quantum]);

  return {
    loadNetworkDefaults,
    loadNetworkPreset,
    createSession,
    discardSession,
    controlSession,
    stepSession,
    setBlindMode,
    extractFeatures,
    previewQuantum,
  };
}

function snapshot<T>(value: T, sourceRevision: number, prefix: string) {
  return {
    id: uniqueId(prefix),
    value,
    source_revision: sourceRevision,
    executed_at: new Date().toISOString(),
  };
}

export function createNetworkRequestSnapshot(
  operation: NetworkRequestSnapshot["operation"],
  sessionId: string | null,
  configuration: NetworkSessionConfiguration | Readonly<NetworkSessionConfiguration> | null,
  requestedFrames?: number,
): NetworkRequestSnapshot {
  return {
    domain: "network",
    operation,
    session_id: sessionId,
    ...(requestedFrames === undefined ? {} : { requested_frames: requestedFrames }),
    configuration: configuration ? structuredClone(configuration) : null,
  };
}

export function createNetworkExperimentAction(
  input: NetworkExperimentInput,
): { type: "EXPERIMENT_ADD"; experiment: ExperimentRecord } {
  return {
    type: "EXPERIMENT_ADD",
    experiment: {
      id: input.id,
      created_at: input.created_at,
      kind: "network_session",
      status: input.status,
      title: input.title,
      sensor_type: "synthetic_magnetometer_network",
      seed: input.configuration?.random_seed,
      sample_count: input.sample_count,
      request_snapshot: createNetworkRequestSnapshot(
        input.operation,
        input.session_id,
        input.configuration,
        input.requested_frames,
      ),
      warnings: input.warnings ?? [],
      timings_ms: {},
    },
  };
}

export function emitNetworkExperiment(
  dispatch: (action: ReturnType<typeof createNetworkExperimentAction>) => void,
  input: NetworkExperimentInput,
): void {
  dispatch(createNetworkExperimentAction(input));
}

function warning(source: "sensor" | "features" | "quantum", message: string) {
  return {
    id: uniqueId("warning"),
    source,
    severity: "warning" as const,
    message,
    created_at: new Date().toISOString(),
  };
}

function uniqueId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function isNotFound(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404;
}

function useSessionLifecycleCleanup(sessionId: string | null): void {
  const latestSessionId = useRef(sessionId);
  latestSessionId.current = sessionId;

  useEffect(() => {
    if (
      pendingUnmountCleanup &&
      pendingUnmountCleanup.sessionId === latestSessionId.current
    ) {
      window.clearTimeout(pendingUnmountCleanup.timer);
      pendingUnmountCleanup = null;
    }

    let pageHideHandled = false;
    const onPageHide = () => {
      pageHideHandled = true;
      const currentSessionId = latestSessionId.current;
      if (currentSessionId) deleteNetworkSessionKeepalive(currentSessionId);
    };
    window.addEventListener("pagehide", onPageHide);

    return () => {
      window.removeEventListener("pagehide", onPageHide);
      if (pageHideHandled) return;
      const currentSessionId = latestSessionId.current;
      if (!currentSessionId) return;

      const timer = window.setTimeout(() => {
        if (pendingUnmountCleanup?.timer !== timer) return;
        pendingUnmountCleanup = null;
        deleteNetworkSessionKeepalive(currentSessionId);
      }, 0);
      pendingUnmountCleanup = { sessionId: currentSessionId, timer };
    };
  }, []);
}

export function thetaAsVector(values: number[]): ThetaVector {
  if (values.length !== 16) throw new Error("Theta must contain exactly 16 values.");
  return values as unknown as ThetaVector;
}

export function shouldReconnectAfterControl(
  action: SessionControl,
  previousState: SessionLifecycle,
): boolean {
  return (
    (action === "reset" || action === "replay") &&
    ["created", "running", "paused", "stopped"].includes(previousState)
  );
}

export function networkStreamCursor(
  previousIdentity: StreamIdentity | null,
  nextIdentity: StreamIdentity,
  lastObservationFrameId: number,
): number {
  if (
    !previousIdentity ||
    previousIdentity.session_id !== nextIdentity.session_id ||
    previousIdentity.stream_revision !== nextIdentity.stream_revision
  ) {
    return 0;
  }
  return Math.max(0, Math.round(lastObservationFrameId));
}
