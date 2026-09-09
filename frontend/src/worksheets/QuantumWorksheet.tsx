import { CapabilityBadge } from "../components/CapabilityBadge";
import {
  activeBundleForTask,
  bundleForResult,
  compactId,
  latestResultForSensor,
} from "../components/DemoReadouts";
import { KernelHeatmap } from "../components/KernelHeatmap";
import { QuantumCircuitControls } from "../components/QuantumCircuitControls";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { VqcCircuitDiagram } from "../diagrams/VqcCircuitDiagram";
import { useDemo } from "../state/demo";
import {
  featureResultIsStale,
  quantumResultIsStale,
  useWorkbench,
} from "../state/workbench";
import { useWorkbenchActions } from "../state/useWorkbenchController";
import type { QuantumBackend } from "../types/quantum";

export function QuantumWorksheet() {
  const { state, dispatch } = useWorkbench();
  const { state: demoState } = useDemo();
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
  const liveResult = latestResultForSensor(
    demoState.analysis,
    state.network.selected_node_id,
    state.network.session?.session_id ?? null,
  );
  const activeTaskId = liveResult?.task_id ?? (
    (state.network.executed?.value.nodes.length ?? 0) >= 3
      ? "aqse.network-pattern.v1"
      : "aqse.local-change.v1"
  );
  const activeBundle = bundleForResult(demoState.registry, liveResult)
    ?? activeBundleForTask(demoState.registry, activeTaskId);

  return (
    <section className="worksheet" aria-labelledby="quantum-title">
      <WorksheetHeader
        titleId="quantum-title"
        index="04"
        eyebrow="EXPLICIT FIXED-THETA EXECUTION"
        title="Quantum Engine"
        description="Inspect the author-provided eight-qubit circuit, the frozen live inference bundle, and an isolated manual kernel preview. Inference never runs QNG."
        actions={<CapabilityBadge status={state.capabilities?.quantum_preview.status ?? "available_not_connected"} />}
      />

      <div className="quantum-safety-banner" role="note">
        <strong>Scientific boundary</strong>
        <span>The connected path uses exact local state simulation, not a physical QPU. Frozen bundle θ and the editable manual-preview θ draft are separate identities; the browser never substitutes one for the other.</span>
      </div>

      <article className="workbench-panel demo-bundle-panel">
        <div className="panel-heading-row">
          <div><p className="panel-kicker">LIVE QUANTUM REPRESENTATION</p><h2>{activeBundle?.bundle_id ?? "No compatible bundle applied"}</h2></div>
          {activeBundle && <span className="result-badge">FROZEN θ</span>}
        </div>
        <dl className="quality-grid">
          <div><dt>Task</dt><dd>{activeBundle?.task_id ?? "—"}</dd></div>
          <div><dt>State8 profile</dt><dd>{activeBundle?.profile_id ?? "—"}</dd></div>
          <div><dt>θ candidate</dt><dd>{activeBundle?.theta_candidate_name ?? "—"}</dd></div>
          <div><dt>θ identity</dt><dd title={activeBundle?.theta_id}>{compactId(activeBundle?.theta_id)}</dd></div>
          <div><dt>Accepted QNG updates</dt><dd>{activeBundle?.accepted_qng_updates ?? "—"}</dd></div>
          <div><dt>TQK reference rows</dt><dd>{activeBundle?.afse_reference_size ?? "—"}</dd></div>
          <div><dt>Encoded live window</dt><dd>{liveResult?.encoded_angles ? `${liveResult.encoded_angles.length} angles` : "—"}</dd></div>
          <div><dt>Backend mode</dt><dd>exact local statevector</dd></div>
        </dl>
        <p className="boundary-note">
          TQK fidelity comparisons feed the frozen AFSE reference map. They do not represent one qubit per sensor and do not expose physical hardware execution.
        </p>
      </article>

      <VqcCircuitDiagram circuit={state.quantum.circuit} />

      <article className="workbench-panel compact-panel demo-circuit-metadata">
        <div className="panel-heading-row"><div><p className="panel-kicker">PROTECTED CIRCUIT METADATA</p><h2>{state.quantum.circuit?.name ?? "Awaiting circuit metadata"}</h2></div></div>
        <dl className="quality-grid">
          <div><dt>Qubits</dt><dd>{state.quantum.circuit?.qubits ?? "—"}</dd></div>
          <div><dt>Input features</dt><dd>{state.quantum.circuit?.input_features ?? "—"}</dd></div>
          <div><dt>Trainable parameters</dt><dd>{state.quantum.circuit?.trainable_parameters ?? "—"}</dd></div>
          <div><dt>Ordered operations</dt><dd>{state.quantum.circuit?.operations.length ?? "—"}</dd></div>
        </dl>
      </article>

      <div className="worksheet-grid quantum-layout-grid demo-preview-boundary">
        <aside className="workbench-panel configuration-panel">
          <div className="panel-heading-row"><div><p className="panel-kicker">LEGACY / MANUAL PREVIEW</p><h2>Isolated execution controls</h2></div><span className="result-badge stale">NOT THE LIVE BUNDLE</span></div>
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

      <section aria-label="Manual preview theta configuration" className="demo-preview-boundary">
        <p className="boundary-note">These 16 controls belong only to the manual preview draft. They cannot mutate the frozen θ used by continuous inference.</p>
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
