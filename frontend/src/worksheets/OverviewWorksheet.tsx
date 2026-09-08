import { useState } from "react";

import { errorMessage } from "../api/client";
import { runQuantumDiagnostics } from "../api/health";
import { CapabilityBadge } from "../components/CapabilityBadge";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { PipelineDiagram } from "../diagrams/PipelineDiagram";
import { TrainingLoopDiagram } from "../diagrams/TrainingLoopDiagram";
import { readingForSensor } from "../state/selectors";
import {
  featureResultIsStale,
  quantumResultIsStale,
  useWorkbench,
} from "../state/workbench";
import type { ServiceHealth } from "../types/workbench";

export function OverviewWorksheet() {
  const { state, dispatch } = useWorkbench();
  const session = state.network.session;
  const networkConfiguration = state.network.executed?.value ?? state.network.draft?.value;
  const latestFrame = state.network.observations.at(-1);
  const latestReading = latestFrame && state.network.selected_node_id
    ? readingForSensor(latestFrame, state.network.selected_node_id)
    : null;
  const preview = state.quantum.preview;
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
      </div>

      <div className="worksheet-grid three-summary-columns">
        <article className="workbench-panel">
          <p className="panel-kicker">SENSOR SUMMARY</p><h2>{networkConfiguration?.session_name ?? "Not configured"}</h2>
          <dl className="quality-grid"><div><dt>Nodes</dt><dd>{networkConfiguration?.nodes.length ?? "—"}</dd></div><div><dt>Sampling</dt><dd>{networkConfiguration ? `${networkConfiguration.sampling_rate_Hz} Hz` : "—"}</dd></div><div><dt>Seed</dt><dd>{networkConfiguration?.random_seed ?? "—"}</dd></div><div><dt>Frames</dt><dd>{session?.latest_frame_id ?? "—"}</dd></div><div><dt>Selected</dt><dd>{state.network.selected_node_id ?? "—"}</dd></div><div><dt>Latest observed value</dt><dd>{formatReading(latestReading)}</dd></div><div className="span-two"><dt>Quality flags</dt><dd>{latestReading?.quality_flags.length ? latestReading.quality_flags.join(", ") : latestReading ? "none" : "—"}</dd></div><div className="span-two"><dt>World Magnetic Model</dt><dd>{wmmProvider ? wmmProvider.configured ? "configured" : "not configured" : "provider registry unavailable"}</dd></div></dl>
        </article>
        <article className="workbench-panel">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">FEATURE SUMMARY</p><h2>{state.features.result?.profile.profile_id ?? "Not executed"}</h2></div>
            {state.features.result && <span className={featureResultIsStale(state) ? "result-badge stale" : "result-badge"}>{featureResultIsStale(state) ? "STALE" : "CURRENT"}</span>}
          </div>
          <dl className="quality-grid"><div><dt>Valid windows</dt><dd>{state.features.result?.valid_window_count ?? "—"}</dd></div><div><dt>Invalid windows</dt><dd>{state.features.result?.invalid_window_count ?? "—"}</dd></div><div><dt>Channel</dt><dd>{state.features.result?.profile.channel ?? state.features.channel}</dd></div><div><dt>Profile</dt><dd>{state.features.result?.profile.extractor_version ?? "—"}</dd></div></dl>
        </article>
        <article className="workbench-panel">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">QUANTUM SUMMARY</p><h2>{preview?.backend ?? "Not executed"}</h2></div>
            {preview && <span className={quantumResultIsStale(state) ? "result-badge stale" : "result-badge"}>{quantumResultIsStale(state) ? "STALE" : "CURRENT"}</span>}
          </div>
          <dl className="quality-grid"><div><dt>Kernel</dt><dd>{preview ? `${preview.reference_kernel.length} × ${preview.reference_kernel[0]?.length ?? 0}` : "—"}</dd></div><div><dt>Execution</dt><dd>{preview ? `${preview.execution_duration_ms.toFixed(3)} ms` : "—"}</dd></div><div><dt>Scaler</dt><dd>{preview?.scaler.version ?? "—"}</dd></div><div><dt>θ snapshot</dt><dd>{state.quantum.theta_executed?.id ?? "—"}</dd></div></dl>
        </article>
      </div>

      <div className="worksheet-grid two-columns">
        <article className="workbench-panel span-two">
          <div className="panel-heading-row">
            <div>
              <p className="panel-kicker">CANONICAL INFERENCE PATH</p>
              <h2>System boundary</h2>
            </div>
            <span className="result-badge">OBSERVATIONS ONLY</span>
          </div>
          <PipelineDiagram />
          <p className="boundary-note">
            Ground truth is available through a separate simulator validation channel and is never
            forwarded to feature extraction, the quantum preview, AFSE, or any future model input.
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
              <h2>QNG boundary</h2>
            </div>
            <CapabilityBadge status="available_not_connected" compact />
          </div>
          <TrainingLoopDiagram />
          <p className="boundary-note">
            Sensor-integrated QNG training is pending. The explicit quantum preview uses fixed θ
            and cannot update parameters.
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
