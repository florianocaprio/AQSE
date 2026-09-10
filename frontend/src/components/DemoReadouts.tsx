import type {
  DemoAnalysisResult,
  DemoAnalysisView,
  DemoBundleSummary,
  DemoMetricSummary,
  DemoPairedModelComparison,
  DemoRegistryView,
  DemoTaskId,
} from "../types/demo";

export function latestResultForSensor(
  analysis: DemoAnalysisView | null,
  sensorId: string | null,
  expectedSessionId?: string | null,
): DemoAnalysisResult | null {
  if (expectedSessionId !== undefined && analysis?.session_id !== expectedSessionId) {
    return null;
  }
  const results = analysis?.latest_results ?? [];
  if (sensorId) {
    return results.find((result) => result.sensor_id === sensorId) ?? null;
  }
  return results[0] ?? null;
}

export function bundleForResult(
  registry: DemoRegistryView | null,
  result: DemoAnalysisResult | null,
): DemoBundleSummary | null {
  if (!result) return null;
  return registry?.bundles.find(({ bundle_id }) => bundle_id === result.bundle_id) ?? null;
}

export function activeBundleForTask(
  registry: DemoRegistryView | null,
  taskId: DemoTaskId,
): DemoBundleSummary | null {
  return registry?.bundles.find(
    ({ active, task_id }) => active && task_id === taskId,
  ) ?? null;
}

export function DemoMetricSummaryTable({
  metrics,
  label,
}: {
  metrics: DemoMetricSummary;
  label: string;
}) {
  const support = Object.entries(metrics.class_support)
    .map(([className, count]) => `${className}: ${count}`)
    .join(" · ");
  const recall = Object.entries(metrics.class_recall)
    .map(([className, value]) => `${className}: ${value === null ? "n/a" : formatPercent(value)}`)
    .join(" · ");
  return (
    <div className="metric-summary" aria-label={label}>
      <dl className="quality-grid">
        <div><dt>Task</dt><dd>{metrics.task_id}</dd></div>
        <div><dt>Partition / profile</dt><dd>{metrics.partition} · {metrics.profile_id}</dd></div>
        <div><dt>Balanced accuracy</dt><dd>{formatPercent(metrics.balanced_accuracy)}</dd></div>
        <div><dt>Macro F1</dt><dd>{formatPercent(metrics.macro_f1)}</dd></div>
        <div><dt>Coverage</dt><dd>{formatPercent(metrics.coverage)}</dd></div>
        <div><dt>Eligible / samples</dt><dd>{metrics.eligible_count} / {metrics.sample_count}</dd></div>
        <div><dt>Uncertain</dt><dd>{metrics.uncertain_count}</dd></div>
        <div><dt>Heuristic OOD</dt><dd>{metrics.heuristic_ood_count}</dd></div>
        <div className="span-two"><dt>Class support</dt><dd>{support || "—"}</dd></div>
        <div className="span-two"><dt>Class recall</dt><dd>{recall || "—"}</dd></div>
      </dl>
      {metrics.replay && (
        <dl className="quality-grid metric-detail-grid" aria-label={`${label} causal replay`}>
          <div><dt>Replay episodes / windows</dt><dd>{metrics.replay.episode_count} / {metrics.replay.window_count}</dd></div>
          <div><dt>Normal FP episodes</dt><dd>{formatPercent(metrics.replay.false_positive_episode_rate)}</dd></div>
          <div><dt>Normal FP windows</dt><dd>{formatPercent(metrics.replay.false_positive_window_rate)}</dd></div>
          <div><dt>Detected / changed</dt><dd>{metrics.replay.detected_episode_count} / {metrics.replay.changed_episode_count}</dd></div>
          <div><dt>Censored changes</dt><dd>{metrics.replay.censored_episode_count}</dd></div>
          <div><dt>Detection delay p50 / p95</dt><dd>{formatSeconds(metrics.replay.detection_delay_p50_s)} / {formatSeconds(metrics.replay.detection_delay_p95_s)}</dd></div>
        </dl>
      )}
      {metrics.by_node_count.length > 0 && (
        <details className="metric-node-details">
          <summary>Per-node-count metrics ({metrics.by_node_count.length})</summary>
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Nodes</th><th>Eligible / samples</th><th>Balanced accuracy</th><th>Macro F1</th><th>Coverage</th></tr></thead>
              <tbody>
                {metrics.by_node_count.map((item) => (
                  <tr key={`${metrics.task_id}-${metrics.partition}-${item.node_count}`}>
                    <td>{item.node_count}</td>
                    <td>{item.eligible_count} / {item.sample_count}</td>
                    <td>{formatPercent(item.balanced_accuracy)}</td>
                    <td>{formatPercent(item.macro_f1)}</td>
                    <td>{formatPercent(item.coverage)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}

export function DemoModelComparisonSummary({
  comparison,
  label,
}: {
  comparison: DemoPairedModelComparison;
  label: string;
}) {
  return (
    <div className="model-comparison-summary" aria-label={label}>
      <div className="panel-heading-row">
        <div><p className="panel-kicker">PAIRED AFSE / RAW COMPARISON</p><h3>{comparison.partition} · {comparison.task_id}</h3></div>
        <span className={comparison.outcome === "HELPED" ? "result-badge" : "result-badge stale"}>{comparison.outcome}</span>
      </div>
      <dl className="quality-grid">
        <div><dt>Balanced accuracy Δ</dt><dd>{formatPointDelta(comparison.balanced_accuracy_delta)}</dd></div>
        <div><dt>95% episode bootstrap interval</dt><dd>{formatInterval(comparison.balanced_accuracy_delta_interval)}</dd></div>
        <div><dt>Macro F1 Δ</dt><dd>{formatPointDelta(comparison.macro_f1_delta)}</dd></div>
        <div><dt>95% episode bootstrap interval</dt><dd>{formatInterval(comparison.macro_f1_delta_interval)}</dd></div>
      </dl>
      <p className="boundary-note">Outcome compares the quantum-AFSE representation with the same-architecture raw State8 baseline. It is not selected from TEST.</p>
    </div>
  );
}

export function formatDemoNumber(value: number | null, digits = 5): string {
  if (value === null || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  return magnitude < 1e-3 || magnitude >= 1e4
    ? value.toExponential(Math.max(1, digits - 1))
    : value.toPrecision(digits);
}

export function formatPercent(value: number): string {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

function formatSeconds(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "—" : `${value.toFixed(2)} s`;
}

function formatPointDelta(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)} pp`;
}

function formatInterval(interval: DemoPairedModelComparison["balanced_accuracy_delta_interval"]): string {
  return `[${formatPointDelta(interval.lower)}, ${formatPointDelta(interval.upper)}]`;
}

export function compactId(value: string | null | undefined): string {
  if (!value) return "—";
  return value.length > 30 ? `${value.slice(0, 14)}…${value.slice(-10)}` : value;
}
