import { useMemo } from "react";

import { CapabilityBadge } from "../components/CapabilityBadge";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { latestContiguousVectorSeries } from "../state/selectors";
import { featureResultIsStale, useWorkbench } from "../state/workbench";
import { useWorkbenchActions } from "../state/useWorkbenchController";
import type { FeatureWindowConfiguration } from "../types/features";

const EXPECTED_FEATURES = [
  "amplitude",
  "phase",
  "frequency",
  "variance",
  "drift",
  "snr",
  "spectral_peak",
  "temperature",
];

export function FeaturesWorksheet() {
  const { state, dispatch } = useWorkbench();
  const { extractFeatures } = useWorkbenchActions();
  const draft = state.features.draft.value;
  const result = state.features.result;
  const latestWindow = result?.windows.at(-1) ?? null;
  const selectedNode = state.network.selected_node_id;
  const networkConfiguration = state.network.executed?.value ?? state.network.draft?.value;
  const source = useMemo(
    () => selectedNode && networkConfiguration
      ? latestContiguousVectorSeries(
          state.network.observations,
          selectedNode,
          networkConfiguration.sampling_rate_Hz,
        )
      : null,
    [networkConfiguration, selectedNode, state.network.observations],
  );
  const selectedConfig = networkConfiguration?.nodes.find(({ sensor_id }) => sensor_id === selectedNode);
  const requiredSamples = Math.max(
    16,
    Math.round(draft.duration_s * (networkConfiguration?.sampling_rate_Hz ?? 0)),
  );
  const canExtract = Boolean(
    source &&
    source.time_s.length >= requiredSamples &&
    selectedConfig?.measurement_mode === "vector",
  );
  const update = (patch: Partial<FeatureWindowConfiguration>) => dispatch({
    type: "FEATURE_DRAFT",
    configuration: { ...draft, ...patch },
  });

  return (
    <section className="worksheet" aria-labelledby="features-title">
      <WorksheetHeader
        titleId="features-title"
        index="03"
        eyebrow="CAUSAL FEATURE PIPELINE"
        title="Features"
        description="Extract the canonical eight-dimensional harmonic profile from measured vector observations. Complete windows only; truth is excluded by schema."
        actions={<CapabilityBadge status={state.capabilities?.vector_sensor.status ?? "available_not_connected"} />}
      />

      <div className="worksheet-grid feature-layout-grid">
        <aside className="workbench-panel configuration-panel">
          <div className="panel-heading-row"><div><p className="panel-kicker">EXTRACTION DRAFT</p><h2>Window policy</h2></div><span className="result-badge">REV {state.features.draft.revision}</span></div>
          <label className="input-control full"><span className="input-label">Source channel</span>
            <select
              value={state.features.channel}
              onChange={(event) => dispatch({
                type: "FEATURE_CHANNEL",
                channel: event.target.value as "x" | "y" | "z" | "magnitude",
              })}
              aria-label="Feature source channel"
            >
              <option value="magnitude">Magnitude</option>
              <option value="x">X component</option>
              <option value="y">Y component</option>
              <option value="z">Z component</option>
            </select>
          </label>
          <div className="control-grid">
            <NumberInput label="Window duration" unit="s" value={draft.duration_s} min={0.01} max={120} onChange={(duration_s) => update({ duration_s })} />
            <NumberInput label="Overlap" unit="fraction" value={draft.overlap_fraction} min={0} max={0.99} step={0.05} onChange={(overlap_fraction) => update({ overlap_fraction })} />
            <NumberInput label="Minimum cycles" value={draft.minimum_cycles} min={2} max={2} disabled onChange={() => undefined} />
            <NumberInput label="Peak prominence" unit="dB" value={draft.minimum_peak_prominence_db} min={6} max={6} disabled onChange={() => undefined} />
            <NumberInput label="Minimum SNR" unit="dB" value={draft.minimum_snr_db} min={0} max={0} disabled onChange={() => undefined} />
          </div>
          <p className="field-help">The 2-cycle, 6 dB peak-prominence, 0 dB SNR, and zero-clipping gates are canonical and locked by the backend profile.</p>
          <label className="toggle-control locked-control"><input type="checkbox" checked={draft.reject_clipped} disabled /><span><strong>Reject clipped windows</strong><small>Mandatory in the current backend quality policy.</small></span></label>

          <div className="source-summary">
            <span>Observation source</span>
            <strong>{selectedNode ?? "No node selected"}</strong>
            <small>{source ? `${source.time_s.length} / ${requiredSamples} required contiguous valid vector samples` : `No eligible contiguous segment (${requiredSamples} samples required).`}</small>
          </div>
          {selectedConfig && selectedConfig.measurement_mode !== "vector" && <p className="inline-message warning-message">Feature extraction currently requires vector-mode observations. The selected node is {selectedConfig.measurement_mode}.</p>}
          {state.features.error && <p className="inline-message error-message" role="alert">{state.features.error}</p>}
          <button type="button" className="button button-primary full-button" onClick={() => void extractFeatures()} disabled={!canExtract || state.features.request_status === "loading"}>
            {state.features.request_status === "loading" ? "Extracting…" : "Extract measured features"}
          </button>
        </aside>

        <div className="feature-results-area">
          <article className="workbench-panel">
            <div className="panel-heading-row">
              <div><p className="panel-kicker">CANONICAL 8D VECTOR</p><h2>Latest feature window</h2></div>
              {result && <span className={featureResultIsStale(state) ? "result-badge stale" : "result-badge"}>{featureResultIsStale(state) ? "STALE" : "EXECUTED"}</span>}
            </div>
            <div className="canonical-feature-grid">
              {EXPECTED_FEATURES.map((expected, index) => {
                const recordIndex = latestWindow?.features.names.indexOf(expected) ?? -1;
                const value = recordIndex >= 0 ? latestWindow?.features.values[recordIndex] : undefined;
                const unit = recordIndex >= 0 ? latestWindow?.features.units[recordIndex] : "—";
                const valid = recordIndex >= 0 ? latestWindow?.quality.per_feature_valid[recordIndex] : undefined;
                return (
                  <article className={`canonical-feature-card${valid === false ? " invalid" : ""}`} key={expected}>
                    <span>F{String(index).padStart(2, "0")}</span>
                    <strong>{expected}</strong>
                    <output>{value === undefined ? "—" : formatValue(value)}</output>
                    <small>{unit}{valid === false ? " · invalid" : ""}</small>
                  </article>
                );
              })}
            </div>
            {!latestWindow && <p className="empty-copy">No feature result has been executed. Cards intentionally contain no placeholder values.</p>}
          </article>

          <article className="workbench-panel">
            <div className="panel-heading-row"><div><p className="panel-kicker">QUALITY GATE</p><h2>Window ledger</h2></div>{result && <span className="chart-unit">{result.valid_window_count} valid · {result.invalid_window_count} invalid</span>}</div>
            {result ? (
              <div className="table-scroll"><table className="data-table"><thead><tr><th>Window</th><th>Time</th><th>Status</th><th>Quantum</th><th>Samples</th><th>Flags</th></tr></thead><tbody>{result.windows.map((window) => <tr key={window.window_id}><td><code>{window.window_id}</code></td><td>{window.start_time_s.toFixed(3)}–{window.end_time_s.toFixed(3)} s</td><td><span className={`quality-status ${window.quality.status}`}>{window.quality.status}</span></td><td>{window.quality.valid_for_quantum ? "eligible" : "rejected"}</td><td>{window.quality.sample_count}</td><td>{window.quality.flags.length ? window.quality.flags.join(", ") : "none"}</td></tr>)}</tbody></table></div>
            ) : <p className="empty-copy">Execute extraction to populate the backend quality ledger.</p>}
            {result?.warnings.map((warning) => <p className="inline-message warning-message" key={warning}>{warning}</p>)}
          </article>

          <article className="workbench-panel provenance-panel">
            <div><p className="panel-kicker">PROVENANCE</p><h2>Executed profile</h2></div>
            {result ? <dl className="quality-grid"><div><dt>Profile</dt><dd>{result.profile.profile_id}</dd></div><div><dt>Extractor</dt><dd>{result.profile.extractor_version}</dd></div><div><dt>Sampling</dt><dd>{result.profile.sampling_rate_hz} Hz</dd></div><div><dt>Channel</dt><dd>{result.profile.channel}</dd></div><div><dt>Window / hop</dt><dd>{result.profile.window_samples} / {result.profile.hop_samples} samples</dd></div><div><dt>Discarded tail</dt><dd>{result.discarded_sample_count} samples</dd></div></dl> : <p className="empty-copy">No executed profile snapshot.</p>}
          </article>
        </div>
      </div>
    </section>
  );
}

function NumberInput({ label, unit, value, min, max, step = 0.01, disabled = false, onChange }: { label: string; unit?: string; value: number; min?: number; max?: number; step?: number; disabled?: boolean; onChange: (value: number) => void }) {
  return <label className="input-control"><span className="input-label">{label}{unit && <small>{unit}</small>}</span><input type="number" value={value} min={min} max={max} step={step} disabled={disabled} onChange={(event) => { if (Number.isFinite(event.target.valueAsNumber)) onChange(event.target.valueAsNumber); }} /></label>;
}

function formatValue(value: number): string {
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  return magnitude >= 1e4 || magnitude < 1e-3 ? value.toExponential(4) : value.toPrecision(6);
}
