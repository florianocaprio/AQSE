import { CapabilityBadge } from "../components/CapabilityBadge";
import {
  DemoMetricSummaryTable,
  activeBundleForTask,
  bundleForResult,
  compactId,
  formatDemoNumber,
  latestResultForSensor,
} from "../components/DemoReadouts";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { useDemo } from "../state/demo";
import { useWorkbench } from "../state/workbench";

export function NeuralWorksheet() {
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

  return (
    <section className="worksheet" aria-labelledby="neural-title">
      <WorksheetHeader
        titleId="neural-title"
        index="07"
        eyebrow="FITTED CLASSICAL OUTPUT"
        title="Neural Model"
        description="Inspect the fitted classical model downstream of AFSE, its uncalibrated class scores, conditional diagnosis, quality/context gates, and same-feature raw baseline."
        actions={<CapabilityBadge status={bundle ? "implemented" : workbenchState.capabilities?.neural_model.status ?? "not_implemented"} />}
      />

      {state.analysis_error && <p className="inline-message error-message" role="alert">{state.analysis_error}</p>}

      <div className="worksheet-grid two-columns">
        <article className="workbench-panel">
          <div className="panel-heading-row"><div><p className="panel-kicker">FITTED MODEL CONTRACT</p><h2>{bundle?.classifier_model_id ?? "No active classifier"}</h2></div>{bundle && <span className="result-badge">FROZEN BUNDLE</span>}</div>
          <dl className="quality-grid">
            <div><dt>Bundle</dt><dd title={bundle?.bundle_id}>{compactId(bundle?.bundle_id)}</dd></div>
            <div><dt>Task</dt><dd>{bundle?.task_id ?? "—"}</dd></div>
            <div><dt>Input</dt><dd>{bundle ? `standardized AFSE z${bundle.afse_reference_size}` : "—"}</dd></div>
            <div><dt>Hidden layers</dt><dd>{bundle ? "32 → 16" : "—"}</dd></div>
            <div><dt>Activation</dt><dd>{bundle ? "tanh" : "—"}</dd></div>
            <div><dt>Solver</dt><dd>{bundle ? "LBFGS" : "—"}</dd></div>
            <div><dt>Classes</dt><dd>{bundle?.class_order.join(", ") ?? "—"}</dd></div>
            <div><dt>Score semantics</dt><dd>not probability-calibrated</dd></div>
          </dl>
          {!bundle && <p className="empty-copy">Prepare and explicitly apply a compatible bundle pair before model inference can run.</p>}
        </article>

        <article className="workbench-panel model-output-card" aria-live="polite">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">CONDITIONAL DIAGNOSIS</p><h2>{result?.displayed_class ?? "No prediction"}</h2></div>
            {result && <span className={result.uncertain || !result.feature_valid ? "result-badge stale" : "result-badge"}>{result.uncertain ? "UNCERTAIN" : result.feature_valid ? "SCORED" : "ABSTAIN"}</span>}
          </div>
          <dl className="quality-grid">
            <div><dt>Raw predicted class</dt><dd>{result?.predicted_class ?? "—"}</dd></div>
            <div><dt>Top model score</dt><dd>{formatDemoNumber(result?.top_score ?? null)}</dd></div>
            <div><dt>Top-two margin</dt><dd>{formatDemoNumber(result?.top_two_margin ?? null)}</dd></div>
            <div><dt>Context mode</dt><dd>{result?.context_mode.replaceAll("_", " ") ?? "—"}</dd></div>
            <div><dt>Raw baseline class</dt><dd>{result?.raw_baseline_class ?? "—"}</dd></div>
            <div><dt>Heuristic OOD</dt><dd>{result?.heuristic_ood === null || !result ? "—" : result.heuristic_ood ? "flagged" : "not flagged"}</dd></div>
          </dl>
          <p className="boundary-note">{result?.attribution_note ?? "No conditional attribution is available until a real analyzed window is returned."}</p>
        </article>

        <article className="workbench-panel span-two">
          <div className="panel-heading-row"><div><p className="panel-kicker">LIVE CLASS SCORE COMPARISON</p><h2>Quantum → AFSE model versus raw-feature baseline</h2></div>{result && <span className="chart-unit">model scores · not calibrated probabilities</span>}</div>
          {result?.class_scores ? (
            <div className="table-scroll">
              <table className="data-table score-table">
                <thead><tr><th>Class</th><th>AFSE model score</th><th>Raw baseline score</th></tr></thead>
                <tbody>
                  {result.class_order.map((label, index) => (
                    <tr key={`${result.result_id}-${label}`}>
                      <td>{label}</td>
                      <td><ScoreBar value={result.class_scores?.[index] ?? null} /></td>
                      <td><ScoreBar value={result.raw_baseline_scores?.[index] ?? null} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p className="empty-copy">No real score vector is available. The UI does not synthesize classes or confidence values.</p>}
          {result && (!result.feature_valid || result.quality_flags.length > 0 || result.heuristic_ood) && (
            <p className="inline-message warning-message" role="note">
              <strong>Quality / rules warning — not a neural prediction:</strong>{" "}
              {!result.feature_valid ? "window rejected; " : ""}
              {result.heuristic_ood ? "heuristic OOD flag; " : ""}
              {result.quality_flags.length ? result.quality_flags.join(", ") : "no additional flags"}.
            </p>
          )}
        </article>

        {bundle && (
          <>
            <article className="workbench-panel">
              <div className="panel-heading-row"><div><p className="panel-kicker">AFSE MODEL VALIDATION</p><h2>{bundle.theta_candidate_name}</h2></div></div>
              <DemoMetricSummaryTable metrics={bundle.validation} label="AFSE model validation metrics" />
            </article>
            <article className="workbench-panel">
              <div className="panel-heading-row"><div><p className="panel-kicker">RAW BASELINE VALIDATION</p><h2>Same State8 features</h2></div></div>
              <DemoMetricSummaryTable metrics={bundle.raw_baseline_validation} label="Raw baseline validation metrics" />
            </article>
          </>
        )}
      </div>
    </section>
  );
}

function ScoreBar({ value }: { value: number | null }) {
  const width = value === null || !Number.isFinite(value)
    ? 0
    : Math.max(0, Math.min(100, value * 100));
  return (
    <div className="score-readout">
      <span className="score-track" aria-hidden="true"><i style={{ width: `${width}%` }} /></span>
      <output>{formatDemoNumber(value)}</output>
    </div>
  );
}
