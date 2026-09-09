import type {
  DemoAnalysisResult,
  DemoAnalysisView,
  DemoBundleSummary,
  DemoMetricSummary,
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
  return (
    <dl className="quality-grid" aria-label={label}>
      <div><dt>Task</dt><dd>{metrics.task_id}</dd></div>
      <div><dt>Partition</dt><dd>{metrics.partition}</dd></div>
      <div><dt>Balanced accuracy</dt><dd>{formatPercent(metrics.balanced_accuracy)}</dd></div>
      <div><dt>Macro F1</dt><dd>{formatPercent(metrics.macro_f1)}</dd></div>
      <div><dt>Coverage</dt><dd>{formatPercent(metrics.coverage)}</dd></div>
      <div><dt>Samples</dt><dd>{metrics.sample_count}</dd></div>
      <div><dt>Uncertain</dt><dd>{metrics.uncertain_count}</dd></div>
      <div><dt>Heuristic OOD</dt><dd>{metrics.heuristic_ood_count}</dd></div>
    </dl>
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

export function compactId(value: string | null | undefined): string {
  if (!value) return "—";
  return value.length > 30 ? `${value.slice(0, 14)}…${value.slice(-10)}` : value;
}
