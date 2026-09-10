import { useState } from "react";

import { errorMessage } from "../api/client";
import { runQuantumDiagnostics } from "../api/health";
import { CapabilityBadge } from "../components/CapabilityBadge";
import {
  activeBundleForTask,
  compactId,
  formatDemoNumber,
  latestResultForSensor,
} from "../components/DemoReadouts";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { PipelineDiagram } from "../diagrams/PipelineDiagram";
import { TrainingLoopDiagram } from "../diagrams/TrainingLoopDiagram";
import { readingForSensor } from "../state/selectors";
import { useDemo } from "../state/demo";
import { useWorkbench } from "../state/workbench";
import type { ServiceHealth } from "../types/workbench";

export function OverviewWorksheet() {
  const { state, dispatch } = useWorkbench();
  const demo = useDemo();
  const session = state.network.session;
  const networkConfiguration = state.network.executed?.value ?? state.network.draft?.value;
  const latestFrame = state.network.observations.at(-1);
  const latestReading = latestFrame && state.network.selected_node_id
    ? readingForSensor(latestFrame, state.network.selected_node_id)
    : null;
  const registry = demo.state.registry;
  const analysis = demo.state.analysis?.session_id === session?.session_id
    ? demo.state.analysis
    : null;
  const liveResult = latestResultForSensor(
    analysis,
    state.network.selected_node_id,
    session?.session_id ?? null,
  );
  const localBundle = activeBundleForTask(registry, "aqse.local-change.v1");
  const networkBundle = activeBundleForTask(registry, "aqse.network-pattern.v1");
  const analysisActive = analysis && !["stopped", "failed"].includes(analysis.state);
  const analysisCanStart = Boolean(
    session && registry?.prepared && registry.active && !analysisActive,
  );
  const pipelineReady = Boolean(registry?.active && localBundle && networkBundle);
  const wmmProvider = state.network.field_providers.find(({ provider_id }) =>
    provider_id.toLowerCase().includes("wmm") || provider_id.toLowerCase().includes("world"),
  );

  return (
    <section className="worksheet" aria-labelledby="overview-title">
      <WorksheetHeader
        titleId="overview-title"
        index="01"
        eyebrow="AQSE RESEARCH WORKBOOK"
        title="Overview"
        description="Operational boundary and provenance-aware status for the local sensor-to-quantum demonstrator. Availability never implies scientific validation."
      />

      <div className="overview-status-grid" aria-live="polite">
        <HealthCard title="Backend API" health={state.backend_health} />
        <HealthCard title="Sensor network service" health={state.network.health} />
        <QuantumHealthCard health={state.quantum_health} />
        <article className="status-card">
          <p className="panel-kicker">SIMULATION PROCESS</p>
          <strong className={`large-status ${session?.state ?? "idle"}`}>
            {session?.state.toUpperCase() ?? "NOT CREATED"}
          </strong>
          <small>
            {session
              ? `Backend time ${session.sim_time_s.toFixed(3)} s · frame ${session.latest_frame_id}`
              : "The network service can be healthy while no session is running."}
          </small>
        </article>
        <article className="status-card">
          <p className="panel-kicker">CONTINUOUS ANALYSIS</p>
          <strong className={`large-status ${analysis?.state ?? "idle"}`}>
            {analysis?.state.replaceAll("_", " ").toUpperCase() ?? "NOT STARTED"}
          </strong>
          <small>
            {analysis
              ? `${analysis.completed_window_count} windows · age ${formatAge(analysis.result_age_ms)}`
              : "Start analysis explicitly after a prepared bundle and simulator session are available."}
          </small>
          <div className="compact-action-row">
            <button
              type="button"
              className="button button-primary"
              disabled={!analysisCanStart || demo.state.analysis_request === "loading"}
              onClick={() => session && void demo.startAnalysis(session.session_id)}
            >
              Start analysis
            </button>
            <button
              type="button"
              className="button button-secondary"
              disabled={!session || !analysisActive || demo.state.analysis_request === "loading"}
              onClick={() => session && void demo.stopAnalysis(session.session_id)}
            >
              Stop
            </button>
          </div>
        </article>
      </div>

      {demo.state.registry_error && (
        <p className="inline-message error-message" role="alert">{demo.state.registry_error}</p>
      )}
      {demo.state.analysis_error && (
        <p className="inline-message error-message" role="alert">{demo.state.analysis_error}</p>
      )}
      {registry && !registry.prepared && (
        <p className="inline-message warning-message" role="status">
          End-to-end artifacts are not prepared: {registry.preparation_detail}
        </p>
      )}

      <div className="worksheet-grid three-summary-columns">
        <article className="workbench-panel">
          <p className="panel-kicker">SENSOR SUMMARY</p><h2>{networkConfiguration?.session_name ?? "Not configured"}</h2>
          <dl className="quality-grid"><div><dt>Nodes</dt><dd>{networkConfiguration?.nodes.length ?? "—"}</dd></div><div><dt>Sampling</dt><dd>{networkConfiguration ? `${networkConfiguration.sampling_rate_Hz} Hz` : "—"}</dd></div><div><dt>Seed</dt><dd>{networkConfiguration?.random_seed ?? "—"}</dd></div><div><dt>Frames</dt><dd>{session?.latest_frame_id ?? "—"}</dd></div><div><dt>Selected</dt><dd>{state.network.selected_node_id ?? "—"}</dd></div><div><dt>Latest observed value</dt><dd>{formatReading(latestReading)}</dd></div><div className="span-two"><dt>Quality flags</dt><dd>{latestReading?.quality_flags.length ? latestReading.quality_flags.join(", ") : latestReading ? "none" : "—"}</dd></div><div className="span-two"><dt>World Magnetic Model</dt><dd>{wmmProvider ? wmmProvider.configured ? "configured" : "not configured" : "provider registry unavailable"}</dd></div></dl>
        </article>
        <article className="workbench-panel">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">LIVE STATE8 WINDOW</p><h2>{liveResult?.profile_id ?? "No live result"}</h2></div>
            {liveResult && <span className={liveResult.feature_valid ? "result-badge" : "result-badge stale"}>{liveResult.feature_valid ? "VALID" : "ABSTAIN"}</span>}
          </div>
          <dl className="quality-grid"><div><dt>Sensor</dt><dd>{liveResult?.sensor_id ?? "—"}</dd></div><div><dt>Context</dt><dd>{liveResult?.context_mode.replaceAll("_", " ") ?? "—"}</dd></div><div><dt>Window</dt><dd>{liveResult ? `${liveResult.window_start_s.toFixed(1)}–${liveResult.window_end_exclusive_s.toFixed(1)} s` : "—"}</dd></div><div><dt>Reference</dt><dd title={liveResult?.reference_id}>{compactId(liveResult?.reference_id)}</dd></div><div className="span-two"><dt>Quality</dt><dd>{liveResult?.quality_flags.length ? liveResult.quality_flags.join(", ") : liveResult ? "eligible" : "—"}</dd></div></dl>
        </article>
        <article className="workbench-panel">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">MODEL OUTPUT</p><h2>{liveResult?.displayed_class ?? "No prediction"}</h2></div>
            {liveResult && <span className={liveResult.uncertain ? "result-badge stale" : "result-badge"}>{liveResult.uncertain ? "UNCERTAIN" : "SCORED"}</span>}
          </div>
          <dl className="quality-grid"><div><dt>Task</dt><dd>{liveResult?.task_id ?? "—"}</dd></div><div><dt>Top score</dt><dd>{formatDemoNumber(liveResult?.top_score ?? null)}</dd></div><div><dt>Margin</dt><dd>{formatDemoNumber(liveResult?.top_two_margin ?? null)}</dd></div><div><dt>Processing</dt><dd>{liveResult ? `${liveResult.processing_duration_ms.toFixed(2)} ms` : "—"}</dd></div><div className="span-two"><dt>Conditional attribution</dt><dd>{liveResult?.attribution_note ?? "—"}</dd></div></dl>
        </article>
      </div>

      <article className="workbench-panel demo-bundle-panel">
        <div className="panel-heading-row">
          <div><p className="panel-kicker">ACTIVE COMPATIBLE BUNDLE</p><h2>{registry?.active ? `Application generation ${registry.active.generation}` : "No bundle applied"}</h2></div>
          <span className={pipelineReady ? "result-badge" : "result-badge stale"}>{pipelineReady ? "COHESIVE PAIR" : "UNAVAILABLE"}</span>
        </div>
        <dl className="quality-grid">
          <div><dt>Application</dt><dd title={registry?.active?.application_id ?? undefined}>{compactId(registry?.active?.application_id)}</dd></div>
          <div><dt>Active selection freeze</dt><dd title={registry?.active?.selection_freeze_id ?? undefined}>{compactId(registry?.active?.selection_freeze_id)}</dd></div>
          <div><dt>Local bundle</dt><dd title={localBundle?.bundle_id}>{compactId(localBundle?.bundle_id)}</dd></div>
          <div><dt>Network bundle</dt><dd title={networkBundle?.bundle_id}>{compactId(networkBundle?.bundle_id)}</dd></div>
          <div><dt>Analysis epoch</dt><dd>{analysis?.worker_epoch ?? "—"}</dd></div>
          <div><dt>Reference progress</dt><dd>{analysis ? `${(analysis.reference_progress * 100).toFixed(0)}%` : "—"}</dd></div>
          <div><dt>Data / result age</dt><dd>{liveResult ? `${formatAge(liveResult.data_age_ms)} / ${formatAge(analysis?.result_age_ms ?? null)}` : "—"}</dd></div>
          <div><dt>Queue / skipped</dt><dd>{analysis ? `${analysis.queue_depth} / ${analysis.skipped_window_count}` : "—"}</dd></div>
          <div><dt>Latency p50 / p95</dt><dd>{analysis && analysis.latency_p50_ms !== null && analysis.latency_p95_ms !== null ? `${analysis.latency_p50_ms.toFixed(2)} / ${analysis.latency_p95_ms.toFixed(2)} ms` : "—"}</dd></div>
        </dl>
        <p className="boundary-note">{registry?.scientific_label ?? "Registry status has not been received."}</p>
      </article>

      <div className="worksheet-grid two-columns">
        <article className="workbench-panel span-two">
          <div className="panel-heading-row">
            <div>
              <p className="panel-kicker">CANONICAL INFERENCE PATH</p>
              <h2>System boundary</h2>
            </div>
            <span className="result-badge">OBSERVATIONS ONLY</span>
          </div>
          <PipelineDiagram stages={[
            { label: "Sensor", detail: "observations", status: session ? "implemented" : "architecture_defined" },
            { label: "State8", detail: "causal features", status: liveResult ? "implemented" : "architecture_defined" },
            { label: "Encoder", detail: "frozen scaler", status: liveResult?.encoded_angles ? "implemented" : "architecture_defined" },
            { label: "VQC / TQK", detail: "exact state", status: pipelineReady ? "implemented" : "architecture_defined" },
            { label: "AFSE", detail: "Nyström vector", status: liveResult?.afse_vector ? "implemented" : "architecture_defined" },
            { label: "Neural", detail: "model scores", status: liveResult?.class_scores ? "implemented" : "architecture_defined" },
            { label: "Output", detail: "conditional", status: liveResult?.displayed_class ? "implemented" : "architecture_defined" },
          ]} />
          <p className="boundary-note">
            Ground truth is available through a separate simulator validation channel and is never
            forwarded to State8 extraction, the protected quantum representation, AFSE, or the fitted model.
          </p>
        </article>

        <article className="workbench-panel">
          <div className="panel-heading-row">
            <div>
              <p className="panel-kicker">CAPABILITY REGISTER</p>
              <h2>Implemented versus reserved</h2>
            </div>
          </div>
          {state.capability_error && <p className="inline-message warning-message">{state.capability_error}</p>}
          <div className="capability-list">
            {state.capabilities ? (
              Object.entries(state.capabilities).map(([name, capability]) => (
                <div className="capability-row" key={name}>
                  <span>
                    <strong>{humanize(name)}</strong>
                    <small>{capability.detail}</small>
                  </span>
                  <CapabilityBadge status={capability.status} compact />
                </div>
              ))
            ) : (
              <p className="empty-copy">Capability registry has not been received.</p>
            )}
          </div>
        </article>

        <article className="workbench-panel">
          <div className="panel-heading-row">
            <div>
              <p className="panel-kicker">TRAINING ARCHITECTURE</p>
              <h2>Bounded QNG training</h2>
            </div>
            <span className={demo.state.training?.state === "completed" ? "result-badge" : "result-badge stale"}>
              {demo.state.training?.state.replaceAll("_", " ").toUpperCase() ?? "IDLE"}
            </span>
          </div>
          <TrainingLoopDiagram />
          <p className="boundary-note">
            QNG runs only from an explicit bounded training job. Completion creates candidate
            artifacts; it never promotes them automatically. Live inference never runs QNG.
          </p>
        </article>
      </div>

      {state.warnings.length > 0 && (
        <aside className="warning-tray" aria-label="Workbench warnings">
          <div>
            <p className="panel-kicker">DIAGNOSTICS</p>
            <h2>Warnings</h2>
          </div>
          {state.warnings.map((warning) => (
            <div className={`warning-entry ${warning.severity}`} key={warning.id}>
              <span><strong>{warning.source}</strong> · {warning.message}</span>
              <button type="button" className="text-button" onClick={() => dispatch({ type: "WARNING_DISMISS", id: warning.id })}>Dismiss</button>
            </div>
          ))}
        </aside>
      )}
    </section>
  );
}

function HealthCard({ title, health }: { title: string; health: ServiceHealth }) {
  return (
    <article className="status-card">
      <p className="panel-kicker">SERVICE HEALTH</p>
      <h3>{title}</h3>
      <strong className={`large-status ${health.status}`}>{health.status.toUpperCase()}</strong>
      <small>{health.detail ?? "Awaiting health response."}</small>
    </article>
  );
}

function QuantumHealthCard({ health }: { health: ServiceHealth }) {
  const [requestStatus, setRequestStatus] = useState<"idle" | "running" | "complete" | "failed">("idle");
  const [result, setResult] = useState<string | null>(null);

  const runDiagnostics = async () => {
    if (requestStatus === "running") return;
    setRequestStatus("running");
    setResult("Running the explicit Qiskit/NumPy statevector comparison…");
    try {
      const diagnostics = await runQuantumDiagnostics();
      setRequestStatus("complete");
      setResult(
        `State comparison ${diagnostics.state_comparison} · ${diagnostics.execution_duration_ms.toFixed(1)} ms`,
      );
    } catch (error: unknown) {
      setRequestStatus("failed");
      setResult(errorMessage(error, "Quantum diagnostics failed."));
    }
  };

  return (
    <article className="status-card">
      <p className="panel-kicker">QUANTUM READINESS</p>
      <h3>Quantum adapter</h3>
      <strong className={`large-status ${health.status}`}>{health.status.toUpperCase()}</strong>
      <small>{health.detail ?? "Awaiting lightweight readiness response."}</small>
      <button
        type="button"
        className="button button-secondary"
        disabled={requestStatus === "running" || health.status !== "ready"}
        onClick={() => void runDiagnostics()}
      >
        {requestStatus === "running" ? "RUNNING DIAGNOSTICS…" : "RUN QUANTUM DIAGNOSTICS"}
      </button>
      {result && (
        <small aria-live="polite" className={requestStatus === "failed" ? "inline-message" : undefined}>
          {result}
        </small>
      )}
    </article>
  );
}

function humanize(value: string): string {
  return value.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

function formatReading(reading: ReturnType<typeof readingForSensor>): string {
  if (!reading) return "—";
  if (reading.components_T) {
    return `[${reading.components_T.map((value) => `${(value * 1e9).toFixed(2)} nT`).join(", ")}]`;
  }
  return reading.value_T === null ? "missing" : `${(reading.value_T * 1e9).toFixed(2)} nT`;
}

function formatAge(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return value < 1_000 ? `${value.toFixed(0)} ms` : `${(value / 1_000).toFixed(2)} s`;
}
