import { CapabilityBadge } from "../components/CapabilityBadge";
import { KernelHeatmap } from "../components/KernelHeatmap";
import { QuantumCircuitControls } from "../components/QuantumCircuitControls";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { VqcCircuitDiagram } from "../diagrams/VqcCircuitDiagram";
import {
  featureResultIsStale,
  quantumResultIsStale,
  useWorkbench,
} from "../state/workbench";
import { useWorkbenchActions } from "../state/useWorkbenchController";
import type { QuantumBackend } from "../types/quantum";

export function QuantumWorksheet() {
  const { state, dispatch } = useWorkbench();
  const { previewQuantum } = useWorkbenchActions();
  const validWindows = state.features.result?.windows.filter(
    ({ quality }) => quality.valid_for_quantum,
  ) ?? [];
  const selectedCount = Math.min(validWindows.length, state.quantum.preview_size);
  const canPreview =
    selectedCount >= 2 &&
    !featureResultIsStale(state) &&
    state.quantum_health.status === "ready";
  const preview = state.quantum.preview;

  return (
    <section className="worksheet" aria-labelledby="quantum-title">
      <WorksheetHeader
        titleId="quantum-title"
        index="04"
        eyebrow="EXPLICIT FIXED-THETA EXECUTION"
        title="Quantum Engine"
        description="Inspect the author-provided eight-qubit circuit and run a bounded TQK kernel preview. This worksheet never trains θ and never invokes QNG."
        actions={<CapabilityBadge status={state.capabilities?.quantum_preview.status ?? "available_not_connected"} />}
      />

      <div className="quantum-safety-banner" role="note">
        <strong>Scientific boundary</strong>
        <span>Feature windows and their signed provenance tokens are forwarded intact. AngleScaler is fitted by the backend on the selected reference set. No browser-side feature reconstruction occurs.</span>
      </div>

      <VqcCircuitDiagram circuit={state.quantum.circuit} />

      <div className="worksheet-grid quantum-layout-grid">
        <aside className="workbench-panel configuration-panel">
          <div className="panel-heading-row"><div><p className="panel-kicker">PREVIEW REQUEST</p><h2>Execution controls</h2></div><span className="result-badge">MANUAL ONLY</span></div>
          <label className="input-control full"><span className="input-label">Backend</span>
            <select value={state.quantum.backend} onChange={(event) => dispatch({ type: "QUANTUM_BACKEND", backend: event.target.value as QuantumBackend })}>
              <option value="qiskit">Qiskit exact statevector</option>
              <option value="numpy">NumPy reference</option>
            </select>
          </label>
          <label className="input-control full"><span className="input-label">Maximum reference windows <small>2–128</small></span><input type="number" min={2} max={128} step={1} value={state.quantum.preview_size} onChange={(event) => { if (Number.isFinite(event.target.valueAsNumber)) dispatch({ type: "QUANTUM_PREVIEW_SIZE", size: event.target.valueAsNumber }); }} /></label>
          <dl className="request-ledger">
            <div><dt>Feature result</dt><dd>{state.features.result ? featureResultIsStale(state) ? "stale" : "current" : "not executed"}</dd></div>
            <div><dt>Eligible windows</dt><dd>{validWindows.length}</dd></div>
            <div><dt>Selected reference rows</dt><dd>{selectedCount}</dd></div>
            <div><dt>Mode</dt><dd>SELF-REFERENCE — EXPLORATORY</dd></div>
            <div><dt>θ source revision</dt><dd>{state.quantum.theta_draft.revision}</dd></div>
            <div><dt>QNG</dt><dd>not invoked</dd></div>
          </dl>
          {state.quantum.error && <p className="inline-message error-message" role="alert">{state.quantum.error}</p>}
          {!canPreview && <p className="field-help">Preview requires a current extraction result, at least two quantum-eligible windows, and healthy quantum infrastructure.</p>}
          <button type="button" className="button button-primary full-button" disabled={!canPreview || state.quantum.request_status === "loading"} onClick={() => void previewQuantum()}>{state.quantum.request_status === "loading" ? "Running preview…" : "Run fixed-θ preview"}</button>
        </aside>

        <article className="workbench-panel quantum-result-panel">
          <div className="panel-heading-row"><div><p className="panel-kicker">EXECUTED ARTIFACT</p><h2>Kernel diagnostics</h2></div>{preview && <span className={quantumResultIsStale(state) ? "result-badge stale" : "result-badge"}>{quantumResultIsStale(state) ? "STALE" : "EXECUTED"}</span>}</div>
          {preview ? (
            <>
              <dl className="diagnostic-grid">
                <Metric label="Minimum" value={preview.diagnostics.minimum} />
                <Metric label="Maximum" value={preview.diagnostics.maximum} />
                <Metric label="Mean off-diagonal" value={preview.diagnostics.mean_off_diagonal} />
                <Metric label="Max diagonal deviation" value={preview.diagnostics.maximum_diagonal_deviation} />
                <Metric label="Max symmetry deviation" value={preview.diagnostics.maximum_symmetry_deviation} />
                <Metric label="Minimum eigenvalue" value={preview.diagnostics.minimum_eigenvalue} />
                <Metric label="Execution" value={preview.execution_duration_ms} unit="ms" />
                <Metric label="Rows" value={preview.reference_window_ids.length} />
              </dl>
              <div className="scope-note"><strong>{preview.backend}</strong><span>{preview.scientific_scope}</span></div>
            </>
          ) : <p className="empty-copy">No quantum preview has been executed. Diagnostics are intentionally blank.</p>}
        </article>
      </div>

      <section aria-label="Quantum theta configuration">
        <QuantumCircuitControls
          values={state.quantum.theta_draft.value}
          onChange={(index, value) => dispatch({ type: "THETA_VALUE", index, value })}
          onReset={() => dispatch({ type: "THETA_RESET" })}
          connectedToPreview
        />
      </section>

      {preview && (
        <div className="worksheet-grid two-columns quantum-artifacts">
          <KernelHeatmap matrix={preview.reference_kernel} label="Reference fidelity kernel" />
          <article className="workbench-panel">
            <div className="panel-heading-row"><div><p className="panel-kicker">ANGLE SCALER SNAPSHOT</p><h2>{preview.scaler.version}</h2></div></div>
            <dl className="quality-grid"><div><dt>Algorithm</dt><dd>{preview.scaler.algorithm}</dd></div><div><dt>Reference dataset</dt><dd>{preview.scaler.reference_dataset_id}</dd></div><div><dt>Feature profile</dt><dd>{preview.scaler.feature_profile_id}</dd></div><div><dt>Dimension</dt><dd>{preview.scaler.mean.length}</dd></div></dl>
            <div className="array-ledger"><span>mean</span><code>[{preview.scaler.mean.map(formatNumber).join(", ")}]</code><span>scale</span><code>[{preview.scaler.scale.map(formatNumber).join(", ")}]</code></div>
          </article>
          <article className="workbench-panel span-two">
            <div className="panel-heading-row"><div><p className="panel-kicker">ENCODING TRACE</p><h2>Raw feature value → encoded angle</h2></div><span className="chart-unit">first reference row · rad</span></div>
            <div className="table-scroll"><table className="data-table"><thead><tr><th>Index</th><th>Feature</th><th>Unit</th><th>Raw value</th><th>Encoded angle</th></tr></thead><tbody>{preview.feature_profile.feature_names.map((name, index) => <tr key={name}><td>F{String(index).padStart(2, "0")}</td><td>{name}</td><td>{preview.feature_profile.feature_units[index]}</td><td><code>{formatNumber(preview.raw_reference_features[0]?.[index] ?? Number.NaN)}</code></td><td><code>{formatNumber(preview.encoded_reference_angles[0]?.[index] ?? Number.NaN)} rad</code></td></tr>)}</tbody></table></div>
            <p className="boundary-note">Values are displayed directly from the immutable preview response. The browser does not recompute AngleScaler output.</p>
          </article>
        </div>
      )}
    </section>
  );
}

function Metric({ label, value, unit }: { label: string; value: number; unit?: string }) {
  return <div><dt>{label}</dt><dd>{formatNumber(value)}{unit ? ` ${unit}` : ""}</dd></div>;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return Math.abs(value) < 1e-3 || Math.abs(value) >= 1e4
    ? value.toExponential(4)
    : value.toPrecision(6);
}
