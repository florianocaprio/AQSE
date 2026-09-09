import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";

import {
  applyDemoBundles,
  cancelDemoTrainingJob,
  getDemoAnalysis,
  getDemoRegistry,
  getDemoTrainingJob,
  startDemoAnalysis,
  startDemoTraining,
  stopDemoAnalysis,
} from "../api/demo";
import { errorMessage } from "../api/client";
import type {
  DemoAnalysisView,
  DemoRegistryView,
  DemoTrainingJobView,
} from "../types/demo";

type RequestState = "idle" | "loading" | "ready" | "failed";

export type DemoState = {
  registry: DemoRegistryView | null;
  analysis: DemoAnalysisView | null;
  training: DemoTrainingJobView | null;
  registry_request: RequestState;
  analysis_request: RequestState;
  training_request: RequestState;
  registry_error: string | null;
  analysis_error: string | null;
  training_error: string | null;
  active_registry_request_id: string | null;
  active_analysis_request_id: string | null;
  active_training_request_id: string | null;
};

type DemoAction =
  | { type: "REGISTRY_REQUEST"; request_id: string }
  | { type: "REGISTRY_READY"; request_id: string; registry: DemoRegistryView }
  | { type: "REGISTRY_FAILED"; request_id: string; message: string }
  | { type: "ANALYSIS_REQUEST"; request_id: string }
  | { type: "ANALYSIS_READY"; request_id: string; analysis: DemoAnalysisView }
  | { type: "ANALYSIS_POLL_READY"; analysis: DemoAnalysisView }
  | { type: "ANALYSIS_FAILED"; request_id: string; message: string }
  | { type: "ANALYSIS_CLEAR" }
  | { type: "TRAINING_REQUEST"; request_id: string }
  | { type: "TRAINING_READY"; request_id: string; training: DemoTrainingJobView }
  | { type: "TRAINING_FAILED"; request_id: string; message: string };

const initialDemoState: DemoState = {
  registry: null,
  analysis: null,
  training: null,
  registry_request: "idle",
  analysis_request: "idle",
  training_request: "idle",
  registry_error: null,
  analysis_error: null,
  training_error: null,
  active_registry_request_id: null,
  active_analysis_request_id: null,
  active_training_request_id: null,
};

function reducer(state: DemoState, action: DemoAction): DemoState {
  switch (action.type) {
    case "REGISTRY_REQUEST":
      return {
        ...state,
        registry_request: "loading",
        registry_error: null,
        active_registry_request_id: action.request_id,
      };
    case "REGISTRY_READY":
      if (state.active_registry_request_id !== action.request_id) return state;
      return {
        ...state,
        registry: action.registry,
        registry_request: "ready",
        registry_error: null,
      };
    case "REGISTRY_FAILED":
      if (state.active_registry_request_id !== action.request_id) return state;
      return {
        ...state,
        registry_request: "failed",
        registry_error: action.message,
      };
    case "ANALYSIS_REQUEST":
      return {
        ...state,
        analysis_request: "loading",
        analysis_error: null,
        active_analysis_request_id: action.request_id,
      };
    case "ANALYSIS_READY":
      if (state.active_analysis_request_id !== action.request_id) return state;
      return {
        ...state,
        analysis: action.analysis,
        analysis_request: "ready",
        analysis_error: null,
      };
    case "ANALYSIS_POLL_READY":
      if (
        state.analysis?.session_id !== action.analysis.session_id ||
        action.analysis.worker_epoch < state.analysis.worker_epoch
      ) {
        return state;
      }
      return {
        ...state,
        analysis: action.analysis,
        analysis_request: "ready",
        analysis_error: null,
      };
    case "ANALYSIS_FAILED":
      if (state.active_analysis_request_id !== action.request_id) return state;
      return {
        ...state,
        analysis_request: "failed",
        analysis_error: action.message,
      };
    case "ANALYSIS_CLEAR":
      return {
        ...state,
        analysis: null,
        analysis_request: "idle",
        analysis_error: null,
        active_analysis_request_id: null,
      };
    case "TRAINING_REQUEST":
      return {
        ...state,
        training_request: "loading",
        training_error: null,
        active_training_request_id: action.request_id,
      };
    case "TRAINING_READY":
      if (state.active_training_request_id !== action.request_id) return state;
      return {
        ...state,
        training: action.training,
        training_request: "ready",
        training_error: null,
      };
    case "TRAINING_FAILED":
      if (state.active_training_request_id !== action.request_id) return state;
      return {
        ...state,
        training_request: "failed",
        training_error: action.message,
      };
  }
}

type DemoContextValue = {
  state: DemoState;
  refreshRegistry: () => Promise<void>;
  startAnalysis: (sessionId: string) => Promise<void>;
  stopAnalysis: (sessionId: string) => Promise<void>;
  startTraining: () => Promise<void>;
  cancelTraining: () => Promise<void>;
  applyBundlePair: (
    localBundleId: string,
    networkBundleId: string,
    selectionFreezeId: string,
  ) => Promise<void>;
  clearAnalysis: () => void;
  pollAnalysis: (sessionId: string, signal?: AbortSignal) => Promise<void>;
  pollTraining: (jobId: string, signal?: AbortSignal) => Promise<void>;
};

const DemoContext = createContext<DemoContextValue | null>(null);

export function DemoProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialDemoState);
  const registrySequence = useRef(0);
  const analysisSequence = useRef(0);
  const trainingSequence = useRef(0);
  const currentIntentId = useRef<string | null>(null);

  const refreshRegistry = useCallback(async () => {
    const requestId = requestIdentity("registry", ++registrySequence.current);
    dispatch({ type: "REGISTRY_REQUEST", request_id: requestId });
    try {
      const registry = await getDemoRegistry();
      dispatch({ type: "REGISTRY_READY", request_id: requestId, registry });
    } catch (error: unknown) {
      dispatch({
        type: "REGISTRY_FAILED",
        request_id: requestId,
        message: errorMessage(error, "AQSE demo registry is unavailable."),
      });
    }
  }, []);

  const startAnalysis = useCallback(async (sessionId: string) => {
    const requestId = requestIdentity("analysis", ++analysisSequence.current);
    dispatch({ type: "ANALYSIS_REQUEST", request_id: requestId });
    try {
      const analysis = await startDemoAnalysis(sessionId);
      dispatch({ type: "ANALYSIS_READY", request_id: requestId, analysis });
    } catch (error: unknown) {
      dispatch({
        type: "ANALYSIS_FAILED",
        request_id: requestId,
        message: errorMessage(error, "Could not start continuous AQSE analysis."),
      });
    }
  }, []);

  const stopAnalysis = useCallback(async (sessionId: string) => {
    const requestId = requestIdentity("analysis", ++analysisSequence.current);
    dispatch({ type: "ANALYSIS_REQUEST", request_id: requestId });
    try {
      const analysis = await stopDemoAnalysis(sessionId);
      dispatch({ type: "ANALYSIS_READY", request_id: requestId, analysis });
    } catch (error: unknown) {
      dispatch({
        type: "ANALYSIS_FAILED",
        request_id: requestId,
        message: errorMessage(error, "Could not stop continuous AQSE analysis."),
      });
    }
  }, []);

  const clearAnalysis = useCallback(() => {
    analysisSequence.current += 1;
    dispatch({ type: "ANALYSIS_CLEAR" });
  }, []);

  const pollAnalysis = useCallback(
    async (sessionId: string, signal?: AbortSignal) => {
      try {
        const analysis = await getDemoAnalysis(sessionId, signal);
        dispatch({ type: "ANALYSIS_POLL_READY", analysis });
      } catch (error: unknown) {
        if (!isAbort(error)) {
          const requestId = requestIdentity(
            "analysis-poll-failure",
            ++analysisSequence.current,
          );
          dispatch({ type: "ANALYSIS_REQUEST", request_id: requestId });
          dispatch({
            type: "ANALYSIS_FAILED",
            request_id: requestId,
            message: errorMessage(error, "Continuous AQSE analysis is unavailable."),
          });
        }
      }
    },
    [],
  );

  const startTraining = useCallback(async () => {
    const studyId = state.registry?.study_artifact_id;
    if (!studyId) {
      const requestId = requestIdentity("training", ++trainingSequence.current);
      dispatch({ type: "TRAINING_REQUEST", request_id: requestId });
      dispatch({
        type: "TRAINING_FAILED",
        request_id: requestId,
        message: "Prepare the bounded demo study before starting training.",
      });
      return;
    }
    if (!currentIntentId.current || isTerminal(state.training?.state)) {
      currentIntentId.current = `aqse-demo-intent-${crypto.randomUUID()}`;
    }
    const requestId = requestIdentity("training", ++trainingSequence.current);
    dispatch({ type: "TRAINING_REQUEST", request_id: requestId });
    try {
      const training = await startDemoTraining({
        intent_id: currentIntentId.current,
        study_artifact_id: studyId,
        requested_action: "fit-two-candidate-local-and-network-bundles",
      });
      dispatch({ type: "TRAINING_READY", request_id: requestId, training });
    } catch (error: unknown) {
      dispatch({
        type: "TRAINING_FAILED",
        request_id: requestId,
        message: errorMessage(error, "Could not start bounded QNG training."),
      });
    }
  }, [state.registry?.study_artifact_id, state.training?.state]);

  const cancelTraining = useCallback(async () => {
    const jobId = state.training?.job_id;
    if (!jobId) return;
    const requestId = requestIdentity("training", ++trainingSequence.current);
    dispatch({ type: "TRAINING_REQUEST", request_id: requestId });
    try {
      const training = await cancelDemoTrainingJob(jobId);
      dispatch({ type: "TRAINING_READY", request_id: requestId, training });
    } catch (error: unknown) {
      dispatch({
        type: "TRAINING_FAILED",
        request_id: requestId,
        message: errorMessage(error, "Could not cancel bounded QNG training."),
      });
    }
  }, [state.training?.job_id]);

  const pollTraining = useCallback(
    async (jobId: string, signal?: AbortSignal) => {
      try {
        const training = await getDemoTrainingJob(jobId, signal);
        const requestId = requestIdentity(
          "training-poll",
          ++trainingSequence.current,
        );
        dispatch({ type: "TRAINING_REQUEST", request_id: requestId });
        dispatch({ type: "TRAINING_READY", request_id: requestId, training });
      } catch (error: unknown) {
        if (!isAbort(error)) {
          const requestId = requestIdentity(
            "training-poll-failure",
            ++trainingSequence.current,
          );
          dispatch({ type: "TRAINING_REQUEST", request_id: requestId });
          dispatch({
            type: "TRAINING_FAILED",
            request_id: requestId,
            message: errorMessage(error, "Could not refresh bounded QNG training."),
          });
        }
      }
    },
    [],
  );

  const applyBundlePair = useCallback(
    async (
      localBundleId: string,
      networkBundleId: string,
      selectionFreezeId: string,
    ) => {
      const requestId = requestIdentity("registry", ++registrySequence.current);
      dispatch({ type: "REGISTRY_REQUEST", request_id: requestId });
      try {
        await applyDemoBundles({
          local_bundle_id: localBundleId,
          network_bundle_id: networkBundleId,
          selection_freeze_id: selectionFreezeId,
        });
        const registry = await getDemoRegistry();
        dispatch({ type: "REGISTRY_READY", request_id: requestId, registry });
      } catch (error: unknown) {
        dispatch({
          type: "REGISTRY_FAILED",
          request_id: requestId,
          message: errorMessage(error, "Could not apply the compatible bundle pair."),
        });
      }
    },
    [],
  );

  const value = useMemo(
    () => ({
      state,
      refreshRegistry,
      startAnalysis,
      stopAnalysis,
      startTraining,
      cancelTraining,
      applyBundlePair,
      clearAnalysis,
      pollAnalysis,
      pollTraining,
    }),
    [
      state,
      refreshRegistry,
      startAnalysis,
      stopAnalysis,
      startTraining,
      cancelTraining,
      applyBundlePair,
      clearAnalysis,
      pollAnalysis,
      pollTraining,
    ],
  );
  return <DemoContext.Provider value={value}>{children}</DemoContext.Provider>;
}

export function useDemo(): DemoContextValue {
  const value = useContext(DemoContext);
  if (!value) throw new Error("useDemo must be used inside DemoProvider");
  return value;
}

export function useDemoPolling(sessionId: string | null) {
  const {
    state,
    refreshRegistry,
    clearAnalysis,
    pollAnalysis,
    pollTraining,
  } = useDemo();
  const priorSession = useRef<string | null>(null);
  const analysisSessionId = state.analysis?.session_id ?? null;
  const analysisState = state.analysis?.state ?? null;
  const trainingJobId = state.training?.job_id ?? null;
  const trainingState = state.training?.state ?? null;

  useEffect(() => {
    void refreshRegistry();
    const interval = window.setInterval(() => void refreshRegistry(), 15_000);
    return () => window.clearInterval(interval);
  }, [refreshRegistry]);

  useEffect(() => {
    if (priorSession.current !== sessionId) {
      priorSession.current = sessionId;
      clearAnalysis();
    }
  }, [clearAnalysis, sessionId]);

  useEffect(() => {
    if (
      !sessionId ||
      analysisSessionId !== sessionId ||
      analysisState === "stopped"
    ) {
      return;
    }
    let controller: AbortController | null = null;
    const poll = () => {
      controller?.abort();
      controller = new AbortController();
      void pollAnalysis(sessionId, controller.signal);
    };
    const interval = window.setInterval(poll, 1_000);
    return () => {
      window.clearInterval(interval);
      controller?.abort();
    };
  }, [analysisSessionId, analysisState, pollAnalysis, sessionId]);

  useEffect(() => {
    if (!trainingJobId || isTerminal(trainingState ?? undefined)) return;
    let controller: AbortController | null = null;
    const poll = () => {
      controller?.abort();
      controller = new AbortController();
      void pollTraining(trainingJobId, controller.signal);
    };
    const interval = window.setInterval(poll, 750);
    return () => {
      window.clearInterval(interval);
      controller?.abort();
    };
  }, [pollTraining, trainingJobId, trainingState]);
}

function requestIdentity(domain: string, sequence: number): string {
  return `${domain}-${sequence}-${Date.now()}`;
}

function isTerminal(state: DemoTrainingJobView["state"] | undefined): boolean {
  return state === "completed" || state === "cancelled" || state === "failed";
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
