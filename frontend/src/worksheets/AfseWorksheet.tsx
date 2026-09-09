import { CapabilityBadge } from "../components/CapabilityBadge";
import {
  activeBundleForTask,
  bundleForResult,
  compactId,
  formatDemoNumber,
  latestResultForSensor,
} from "../components/DemoReadouts";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { useDemo } from "../state/demo";
import { useWorkbench } from "../state/workbench";

export function AfseWorksheet() {
  const { state: workbenchState } = useWorkbench();
  const { state } = useDemo();
  const result = latestResultForSensor(
    state.analysis,
    workbenchState.network.selected_node_id,
    workbenchState.network.session?.session_id ?? null,
  );
  const fallbackTask = (workbenchState.network.executed?.value.nodes.length ?? 0) >= 3
    ? "aqse.network-pattern.v1"
    : "aqse.local-change.v1";
  const bundle = bundleForResult(state.registry, result)
    ?? activeBundleForTask(state.registry, fallbackTask);
  const fitted = Boolean(bundle);

  return (
    <section className="worksheet" aria-labelledby="afse-title">
      <WorksheetHeader
        titleId="afse-title"
        index="06"
        eyebrow="FIXED LOCAL EMBEDDING"
        title="Local Embedding / AFSE"
        description="Inspect the actual fixed-dimensional regularized Nyström embedding produced from the frozen quantum reference. The visible query batch never refits or reorders the reference space."
        actions={<CapabilityBadge status={fitted ? "implemented" : workbenchState.capabilities?.local_embedding_afse.status ?? "architecture_defined"} />}
      />

      {state.registry_error && <p className="inline-message error-message" role="alert">{state.registry_error}</p>}
      {state.analysis_error && <p className="inline-message error-message" role="alert">{state.analysis_error}</p>}

      <div className="worksheet-grid two-columns">
        <article className="workbench-panel">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">FROZEN MAP</p><h2>{bundle?.afse_method_id ?? "No fitted AFSE bundle"}</h2></div>
            {bundle && <span className="result-badge">VERSIONED</span>}
          </div>
          <dl className="quality-grid">
            <div><dt>Bundle</dt><dd title={bundle?.bundle_id}>{compactId(bundle?.bundle_id)}</dd></div>
            <div><dt>Task</dt><dd>{bundle?.task_id ?? "—"}</dd></div>
            <div><dt>Reference size M</dt><dd>{bundle?.afse_reference_size ?? "—"}</dd></div>
            <div><dt>Embedding dimension</dt><dd>{result?.afse_vector?.length ?? bundle?.afse_reference_size ?? "—"}</dd></div>
            <div><dt>Ridge λ</dt><dd>{bundle ? formatDemoNumber(bundle.afse_ridge_lambda) : "—"}</dd></div>
            <div><dt>θ identity</dt><dd title={bundle?.theta_id}>{compactId(bundle?.theta_id)}</dd></div>
            <div><dt>Profile</dt><dd>{bundle?.profile_id ?? "—"}</dd></div>
            <div><dt>Live reference size</dt><dd>{result?.afse_reference_size ?? "—"}</dd></div>
          </dl>
          {!bundle && <p className="empty-copy">Prepare and explicitly apply a compatible bundle pair before AFSE inference can run.</p>}
        </article>

        <article className="workbench-panel">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">RECONSTRUCTION DIAGNOSTIC</p><h2>{result ? `Window ${compactId(result.window_id)}` : "No live embedding"}</h2></div>
            {result && result.heuristic_ood !== null && (
              <span className={result.heuristic_ood ? "result-badge stale" : "result-badge"}>
                {result.heuristic_ood ? "HEURISTIC OOD" : "WITHIN HEURISTIC"}
              </span>
            )}
          </div>
          <dl className="quality-grid">
            <div><dt>Sensor</dt><dd>{result?.sensor_id ?? "—"}</dd></div>
            <div><dt>Context</dt><dd>{result?.context_mode.replaceAll("_", " ") ?? "—"}</dd></div>
            <div><dt>Residual</dt><dd>{formatDemoNumber(result?.reconstruction_residual ?? null)}</dd></div>
            <div><dt>Feature eligibility</dt><dd>{result ? result.feature_valid ? "eligible" : "abstained" : "—"}</dd></div>
          </dl>
          <p className="boundary-note">The residual/OOD flag is a fixed engineering diagnostic. It is not calibrated uncertainty, a causal finding, or a physical anomaly proof.</p>
        </article>

        <article className="workbench-panel span-two">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">LIVE AFSE VECTOR z(x)</p><h2>{result?.afse_vector ? `${result.afse_vector.length} fixed coordinates` : "Awaiting an eligible analyzed window"}</h2></div>
            {result?.afse_vector && <span className="chart-unit">query only · frozen reference</span>}
          </div>
          {result?.afse_vector ? (
            <div className="embedding-vector-grid">
              {result.afse_vector.map((value, index) => (
                <div key={`${result.result_id}-z-${index}`}>
                  <span>z{index}</span>
                  <output>{formatDemoNumber(value)}</output>
                </div>
              ))}
            </div>
          ) : <p className="empty-copy">No AFSE coordinates are shown until the backend returns a real connected-path result.</p>}
        </article>
      </div>
    </section>
  );
}
