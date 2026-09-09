import {
  createContext,
  useContext,
  useMemo,
  useReducer,
  type Dispatch,
  type ReactNode,
} from "react";

import type { WorkbenchCapabilities } from "../types/capabilities";
import type {
  FeatureExtractionResponse,
  FeatureWindowConfiguration,
} from "../types/features";
import type {
  NetworkHealth,
  FieldProviderStatus,
  NetworkPresetDescriptor,
  NetworkSessionConfiguration,
  ObservationFrame,
  SessionStatus,
  TruthFrame,
} from "../types/network";
import type {
  CircuitDescription,
  QuantumBackend,
  QuantumPreviewResponse,
} from "../types/quantum";
import type {
  ExecutedSnapshot,
  ComputedArtifact,
  ExperimentRecord,
  RevisionedDraft,
  ServiceHealth,
  ThetaVector,
  SignalChannel,
  WarningRecord,
  WorksheetId,
} from "../types/workbench";

const MAX_EXPERIMENT_HISTORY = 50;
const DEFAULT_THETA = Array.from({ length: 16 }, () => 0) as unknown as ThetaVector;

export type RequestStatus = "idle" | "loading" | "running" | "ready" | "failed";

type NetworkState = {
  health: ServiceHealth;
  detail: NetworkHealth | null;
  presets: NetworkPresetDescriptor[];
  field_providers: FieldProviderStatus[];
  draft: RevisionedDraft<NetworkSessionConfiguration> | null;
  executed: ExecutedSnapshot<NetworkSessionConfiguration> | null;
  session: SessionStatus | null;
  observations: ObservationFrame[];
  truth: TruthFrame[];
  selected_node_id: string | null;
  blind_mode: boolean;
  request_status: RequestStatus;
  stream_status: "disconnected" | "connecting" | "connected" | "error";
  stream_revision: number;
  error: string | null;
  gap_detected: boolean;
  data_revision: number;
};

type FeatureState = {
  channel: SignalChannel;
  draft: RevisionedDraft<FeatureWindowConfiguration>;
  executed: ExecutedSnapshot<FeatureWindowConfiguration> | null;
  result: FeatureExtractionResponse | null;
  computed: ComputedArtifact<FeatureExtractionResponse> | null;
  source_session_id: string | null;
  source_data_revision: number | null;
  request_status: RequestStatus;
  active_request_id: string | null;
  error: string | null;
};

type QuantumState = {
  theta_draft: RevisionedDraft<ThetaVector>;
  theta_executed: ExecutedSnapshot<ThetaVector> | null;
  backend: QuantumBackend;
  preview_size: number;
  circuit: CircuitDescription | null;
  preview: QuantumPreviewResponse | null;
  computed: ComputedArtifact<QuantumPreviewResponse> | null;
  request_status: RequestStatus;
  active_request_id: string | null;
  error: string | null;
};

export type WorkbenchState = {
  active_worksheet: WorksheetId;
  backend_health: ServiceHealth;
  quantum_health: ServiceHealth;
  capabilities: WorkbenchCapabilities | null;
  capability_error: string | null;
  network: NetworkState;
  features: FeatureState;
  quantum: QuantumState;
  warnings: WarningRecord[];
  experiments: ExperimentRecord[];
};

export type WorkbenchAction =
  | { type: "NAVIGATE"; worksheet: WorksheetId }
  | { type: "BACKEND_HEALTH"; health: ServiceHealth }
  | { type: "QUANTUM_HEALTH"; health: ServiceHealth }
  | { type: "CAPABILITIES_READY"; capabilities: WorkbenchCapabilities }
  | { type: "CAPABILITIES_FAILED"; message: string }
  | { type: "NETWORK_HEALTH"; health: ServiceHealth; detail?: NetworkHealth }
  | { type: "NETWORK_PRESETS"; presets: NetworkPresetDescriptor[] }
  | { type: "NETWORK_FIELD_PROVIDERS"; providers: FieldProviderStatus[] }
  | { type: "NETWORK_DEFAULTS"; configuration: NetworkSessionConfiguration }
  | { type: "NETWORK_DRAFT"; configuration: NetworkSessionConfiguration }
  | { type: "NETWORK_CONFIGURATION_IMPORTED"; configuration: NetworkSessionConfiguration }
  | { type: "NETWORK_NODE_SELECTED"; sensor_id: string }
  | { type: "NETWORK_BLIND_MODE"; enabled: boolean }
  | { type: "NETWORK_REQUEST"; status: RequestStatus; error?: string }
  | { type: "NETWORK_SESSION"; status: SessionStatus; snapshot?: ExecutedSnapshot<NetworkSessionConfiguration>; stream_revision?: number }
  | { type: "NETWORK_SESSION_CLEAR" }
  | { type: "NETWORK_STREAM"; status: NetworkState["stream_status"]; stream_revision: number }
  | { type: "NETWORK_STREAM_RECONNECT" }
  | { type: "NETWORK_OBSERVATIONS"; frames: ObservationFrame[]; gap_detected?: boolean; stream_revision?: number }
  | { type: "NETWORK_TRUTH"; frames: TruthFrame[]; gap_detected?: boolean; stream_revision?: number }
  | { type: "NETWORK_CLEAR_DATA" }
  | { type: "FEATURE_DRAFT"; configuration: FeatureWindowConfiguration }
  | { type: "FEATURE_CHANNEL"; channel: SignalChannel }
  | { type: "FEATURE_REQUEST"; status: RequestStatus; request_id?: string; error?: string }
  | {
      type: "FEATURE_RESULT";
      result: FeatureExtractionResponse;
      source_session_id: string;
      source_data_revision: number;
      snapshot: ExecutedSnapshot<FeatureWindowConfiguration>;
      request_id: string;
      input_artifact_ids: string[];
    }
  | { type: "THETA_VALUE"; index: number; value: number }
  | { type: "THETA_RESET" }
  | { type: "QUANTUM_BACKEND"; backend: QuantumBackend }
  | { type: "QUANTUM_PREVIEW_SIZE"; size: number }
  | { type: "QUANTUM_CIRCUIT"; circuit: CircuitDescription }
  | { type: "QUANTUM_REQUEST"; status: RequestStatus; request_id?: string; error?: string }
  | {
      type: "QUANTUM_RESULT";
      preview: QuantumPreviewResponse;
      theta_snapshot: ExecutedSnapshot<ThetaVector>;
      request_id: string;
      input_artifact_ids: string[];
    }
  | { type: "WARNING_ADD"; warning: WarningRecord }
  | { type: "WARNING_DISMISS"; id: string }
  | { type: "EXPERIMENT_ADD"; experiment: ExperimentRecord };

export function createInitialWorkbenchState(): WorkbenchState {
  return {
    active_worksheet: worksheetFromHash(),
    backend_health: { status: "checking" },
    quantum_health: { status: "checking" },
    capabilities: null,
    capability_error: null,
    network: {
      health: { status: "checking" },
      detail: null,
      presets: [],
      field_providers: [],
      draft: null,
      executed: null,
      session: null,
      observations: [],
      truth: [],
      selected_node_id: null,
      blind_mode: true,
      request_status: "idle",
      stream_status: "disconnected",
      stream_revision: 0,
      error: null,
      gap_detected: false,
      data_revision: 0,
    },
    features: {
      channel: "magnitude",
      draft: {
        value: {
          duration_s: 1,
          overlap_fraction: 0.5,
          minimum_cycles: 2,
          minimum_peak_prominence_db: 6,
          minimum_snr_db: 0,
          reject_clipped: true,
        },
        revision: 0,
      },
      executed: null,
      result: null,
      computed: null,
      source_session_id: null,
      source_data_revision: null,
      request_status: "idle",
      active_request_id: null,
      error: null,
    },
    quantum: {
      theta_draft: { value: DEFAULT_THETA, revision: 0 },
      theta_executed: null,
      backend: "qiskit",
      preview_size: 16,
      circuit: null,
      preview: null,
      computed: null,
      request_status: "idle",
      active_request_id: null,
      error: null,
    },
    warnings: [],
    experiments: [],
  };
}

export function workbenchReducer(
  state: WorkbenchState,
  action: WorkbenchAction,
): WorkbenchState {
  switch (action.type) {
    case "NAVIGATE":
      return { ...state, active_worksheet: action.worksheet };
    case "BACKEND_HEALTH":
      return { ...state, backend_health: action.health };
    case "QUANTUM_HEALTH":
      return { ...state, quantum_health: action.health };
    case "CAPABILITIES_READY":
      return { ...state, capabilities: action.capabilities, capability_error: null };
    case "CAPABILITIES_FAILED":
      return { ...state, capability_error: action.message };
    case "NETWORK_HEALTH":
      return {
        ...state,
        network: {
          ...state.network,
          health: action.health,
          detail: action.detail ?? state.network.detail,
        },
      };
    case "NETWORK_PRESETS":
      return {
        ...state,
        network: { ...state.network, presets: action.presets },
      };
    case "NETWORK_FIELD_PROVIDERS":
      return {
        ...state,
        network: { ...state.network, field_providers: action.providers },
      };
    case "NETWORK_DEFAULTS":
      return {
        ...state,
        network: {
          ...state.network,
          draft: {
            value: action.configuration,
            revision: state.network.draft ? state.network.draft.revision + 1 : 0,
          },
          selected_node_id: action.configuration.nodes[0]?.sensor_id ?? null,
          error: null,
        },
      };
    case "NETWORK_DRAFT":
      return {
        ...state,
        network: {
          ...state.network,
          draft: {
            value: action.configuration,
            revision: (state.network.draft?.revision ?? 0) + 1,
          },
        },
      };
    case "NETWORK_CONFIGURATION_IMPORTED":
      return {
        ...state,
        network: {
          ...state.network,
          draft: {
            value: action.configuration,
            revision: (state.network.draft?.revision ?? 0) + 1,
          },
          selected_node_id: action.configuration.nodes[0]?.sensor_id ?? null,
          error: null,
        },
      };
    case "NETWORK_NODE_SELECTED":
      return {
        ...state,
        network: { ...state.network, selected_node_id: action.sensor_id },
      };
    case "NETWORK_BLIND_MODE":
      return {
        ...state,
        network: {
          ...state.network,
          blind_mode: action.enabled,
          truth: action.enabled ? [] : state.network.truth,
        },
      };
    case "NETWORK_REQUEST":
      return {
        ...state,
        network: {
          ...state.network,
          request_status: action.status,
          error: action.error ?? null,
        },
      };
    case "NETWORK_SESSION":
      if (isObsoleteStreamAction(state, action.stream_revision)) return state;
      return {
        ...state,
        network: {
          ...state.network,
          session: action.status,
          executed: action.snapshot ?? state.network.executed,
          request_status: "ready",
          error: null,
        },
      };
    case "NETWORK_SESSION_CLEAR":
      return {
        ...state,
        network: {
          ...state.network,
          session: null,
          executed: null,
          observations: [],
          truth: [],
          request_status: "idle",
          stream_status: "disconnected",
          error: null,
          gap_detected: false,
          data_revision: state.network.data_revision + 1,
        },
      };
    case "NETWORK_STREAM":
      if (isObsoleteStreamAction(state, action.stream_revision)) return state;
      return {
        ...state,
        network: { ...state.network, stream_status: action.status },
      };
    case "NETWORK_STREAM_RECONNECT":
      return {
        ...state,
        network: {
          ...state.network,
          stream_revision: state.network.stream_revision + 1,
        },
      };
    case "NETWORK_OBSERVATIONS":
      if (isObsoleteStreamAction(state, action.stream_revision)) return state;
      return {
        ...state,
        network: {
          ...state.network,
          observations: mergeFrames(
            state.network.observations,
            action.frames,
            networkFrameLimit(state),
          ),
          data_revision:
            state.network.data_revision + (action.frames.length > 0 ? 1 : 0),
          gap_detected: state.network.gap_detected || Boolean(action.gap_detected),
        },
      };
    case "NETWORK_TRUTH":
      if (isObsoleteStreamAction(state, action.stream_revision)) return state;
      if (state.network.blind_mode) return state;
      return {
        ...state,
        network: {
          ...state.network,
          truth: mergeFrames(
            state.network.truth,
            action.frames,
            networkFrameLimit(state),
          ),
          gap_detected: state.network.gap_detected || Boolean(action.gap_detected),
        },
      };
    case "NETWORK_CLEAR_DATA":
      return {
        ...state,
        network: {
          ...state.network,
          observations: [],
          truth: [],
          gap_detected: false,
          data_revision: state.network.data_revision + 1,
        },
      };
    case "FEATURE_DRAFT":
      return {
        ...state,
        features: {
          ...state.features,
          draft: {
            value: action.configuration,
            revision: state.features.draft.revision + 1,
          },
        },
      };
    case "FEATURE_CHANNEL":
      return {
        ...state,
        features: {
          ...state.features,
          channel: action.channel,
          draft: {
            ...state.features.draft,
            revision: state.features.draft.revision + 1,
          },
        },
      };
    case "FEATURE_REQUEST":
      if (
        action.status !== "loading" &&
        action.request_id !== undefined &&
        action.request_id !== state.features.active_request_id
      ) return state;
      return {
        ...state,
        features: {
          ...state.features,
          request_status: action.status,
          active_request_id:
            action.status === "loading"
              ? action.request_id ?? null
              : null,
          error: action.error ?? null,
        },
      };
    case "FEATURE_RESULT":
      if (action.request_id !== state.features.active_request_id) return state;
      return {
        ...state,
        features: {
          ...state.features,
          result: action.result,
          computed: {
            data: action.result,
            provenance: {
              experiment_id: action.snapshot.id,
              artifact_id: `feature-artifact-${action.snapshot.id}`,
              created_at: action.snapshot.executed_at,
              input_artifact_ids: action.input_artifact_ids,
              configuration_snapshot_id: action.snapshot.id,
              feature_profile_id: action.result.profile.profile_id,
            },
            warnings: action.result.warnings.map((message, index) => ({
              id: `${action.snapshot.id}-warning-${index}`,
              source: "features" as const,
              severity: "warning" as const,
              message,
              created_at: action.snapshot.executed_at,
            })),
          },
          source_session_id: action.source_session_id,
          source_data_revision: action.source_data_revision,
          executed: action.snapshot,
          request_status: "ready",
          active_request_id: null,
          error: null,
        },
      };
    case "THETA_VALUE": {
      if (action.index < 0 || action.index >= 16 || !Number.isFinite(action.value)) {
        return state;
      }
      const theta = state.quantum.theta_draft.value.map((value, index) =>
        index === action.index ? action.value : value,
      ) as unknown as ThetaVector;
      return {
        ...state,
        quantum: {
          ...state.quantum,
          theta_draft: {
            value: theta,
            revision: state.quantum.theta_draft.revision + 1,
          },
        },
      };
    }
    case "THETA_RESET":
      return {
        ...state,
        quantum: {
          ...state.quantum,
          theta_draft: {
            value: DEFAULT_THETA,
            revision: state.quantum.theta_draft.revision + 1,
          },
        },
      };
    case "QUANTUM_BACKEND":
      return { ...state, quantum: { ...state.quantum, backend: action.backend } };
    case "QUANTUM_PREVIEW_SIZE":
      return {
        ...state,
        quantum: {
          ...state.quantum,
          preview_size: Math.min(128, Math.max(2, Math.round(action.size))),
        },
      };
    case "QUANTUM_CIRCUIT":
      return { ...state, quantum: { ...state.quantum, circuit: action.circuit } };
    case "QUANTUM_REQUEST":
      if (
        action.status !== "loading" &&
        action.request_id !== undefined &&
        action.request_id !== state.quantum.active_request_id
      ) return state;
      return {
        ...state,
        quantum: {
          ...state.quantum,
          request_status: action.status,
          active_request_id:
            action.status === "loading"
              ? action.request_id ?? null
              : null,
          error: action.error ?? null,
        },
      };
    case "QUANTUM_RESULT":
      if (action.request_id !== state.quantum.active_request_id) return state;
      return {
        ...state,
        quantum: {
          ...state.quantum,
          preview: action.preview,
          computed: {
            data: action.preview,
            provenance: {
              experiment_id: action.theta_snapshot.id,
              artifact_id: `quantum-artifact-${action.preview.preview_id}`,
              created_at: action.theta_snapshot.executed_at,
              input_artifact_ids: action.input_artifact_ids,
              theta_snapshot_id: action.theta_snapshot.id,
              feature_profile_id: action.preview.feature_profile.profile_id,
              scaler_version: action.preview.scaler.version,
            },
            duration_ms: action.preview.execution_duration_ms,
            warnings: [],
          },
          theta_executed: action.theta_snapshot,
          request_status: "ready",
          active_request_id: null,
          error: null,
        },
      };
    case "WARNING_ADD":
      return {
        ...state,
        warnings: [...state.warnings.filter(({ id }) => id !== action.warning.id), action.warning],
      };
    case "WARNING_DISMISS":
      return { ...state, warnings: state.warnings.filter(({ id }) => id !== action.id) };
    case "EXPERIMENT_ADD":
      return {
        ...state,
        experiments: [action.experiment, ...state.experiments].slice(
          0,
          MAX_EXPERIMENT_HISTORY,
        ),
      };
  }
}

function isObsoleteStreamAction(
  state: WorkbenchState,
  streamRevision: number | undefined,
): boolean {
  return (
    streamRevision !== undefined &&
    streamRevision !== state.network.stream_revision
  );
}

export function networkResultIsStale(state: WorkbenchState): boolean {
  return Boolean(
    state.network.draft &&
      state.network.executed &&
      state.network.draft.revision !== state.network.executed.source_revision,
  );
}

export function featureResultIsStale(state: WorkbenchState): boolean {
  const resultSensorId = state.features.result?.windows[0]?.sensor_id ?? null;
  return Boolean(
    state.features.result &&
      (!state.features.executed ||
        networkResultIsStale(state) ||
        state.features.draft.revision !== state.features.executed.source_revision ||
        state.features.source_session_id !== state.network.session?.session_id ||
        state.features.source_data_revision !== state.network.data_revision ||
        resultSensorId !== state.network.selected_node_id),
  );
}

export function quantumResultIsStale(state: WorkbenchState): boolean {
  if (!state.quantum.preview || !state.quantum.theta_executed) return false;
  const expectedBackend = `${state.quantum.backend}_statevector`;
  const eligibleWindowCount = state.features.result?.windows.filter(
    ({ quality }) => quality.valid_for_quantum,
  ).length ?? 0;
  const selectedWindowCount = Math.min(
    eligibleWindowCount,
    state.quantum.preview_size,
  );
  const expectedWindowIds = state.features.result?.windows
    .filter(({ quality }) => quality.valid_for_quantum)
    .slice(0, selectedWindowCount)
    .map(({ window_id }) => window_id) ?? [];
  const executedWindowIds = state.quantum.preview.reference_window_ids;
  return (
    state.quantum.theta_draft.revision !== state.quantum.theta_executed.source_revision ||
    featureResultIsStale(state) ||
    state.quantum.preview.backend !== expectedBackend ||
    executedWindowIds.length !== selectedWindowCount ||
    executedWindowIds.some((windowId, index) => windowId !== expectedWindowIds[index]) ||
    state.quantum.preview.feature_profile.profile_id !== state.features.result?.profile.profile_id ||
    state.quantum.preview.scaler.reference_dataset_id !== state.features.source_session_id
  );
}

const WorkbenchContext = createContext<
  { state: WorkbenchState; dispatch: Dispatch<WorkbenchAction> } | undefined
>(undefined);

export function WorkbenchProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(
    workbenchReducer,
    undefined,
    createInitialWorkbenchState,
  );
  const value = useMemo(() => ({ state, dispatch }), [state]);
  return <WorkbenchContext.Provider value={value}>{children}</WorkbenchContext.Provider>;
}

export function useWorkbench() {
  const context = useContext(WorkbenchContext);
  if (!context) throw new Error("useWorkbench must be used inside WorkbenchProvider");
  return context;
}

function worksheetFromHash(): WorksheetId {
  if (typeof window === "undefined") return "overview";
  const candidate = window.location.hash.replace(/^#\/?/, "") as WorksheetId;
  const valid: WorksheetId[] = [
    "overview",
    "sensors",
    "features",
    "quantum",
    "qng",
    "afse",
    "neural",
    "experiments",
  ];
  return valid.includes(candidate) ? candidate : "overview";
}

function networkFrameLimit(state: WorkbenchState): number {
  const configuration = state.network.executed?.value ?? state.network.draft?.value;
  if (!configuration) return 600;
  return Math.min(
    20_000,
    Math.max(1, Math.round(configuration.sampling_rate_Hz * configuration.buffer_duration_s)),
  );
}

function mergeFrames<T extends { frame_id: number; session_id: string }>(
  current: T[],
  incoming: T[],
  limit: number,
): T[] {
  if (incoming.length === 0) return current;
  const sessionId = incoming[incoming.length - 1]?.session_id;
  const byId = new Map<number, T>();
  for (const frame of current) {
    if (frame.session_id === sessionId) byId.set(frame.frame_id, frame);
  }
  for (const frame of incoming) byId.set(frame.frame_id, frame);
  return [...byId.values()]
    .sort((left, right) => left.frame_id - right.frame_id)
    .slice(-limit);
}
