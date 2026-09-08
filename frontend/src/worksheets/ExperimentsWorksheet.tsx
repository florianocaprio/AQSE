import { useId, useState, type ChangeEvent } from "react";

import { WorksheetHeader } from "../components/WorksheetHeader";
import { parseNetworkConfigurationJson } from "../state/configurationImport";
import { useWorkbench } from "../state/workbench";
import type { ExperimentRecord } from "../types/workbench";

const MAX_CONFIGURATION_FILE_BYTES = 2 * 1024 * 1024;
const BLIND_LEDGER_HELP =
  "Experiment request snapshots can contain latent scenario inputs. Disable blind mode to view or export the experiment ledger. Blind mode is a local UI barrier, not an API security control.";

export function experimentDisclosurePolicy(
  blindMode: boolean,
  experimentCount: number,
) {
  return {
    exposeRequestSnapshots: !blindMode,
    allowLedgerExport: !blindMode && experimentCount > 0,
  };
}

export function ExperimentsWorksheet() {
  const { state, dispatch } = useWorkbench();
  const configurationInputId = useId();
  const [importError, setImportError] = useState<string | null>(null);
  const [importNotice, setImportNotice] = useState<string | null>(null);
  const disclosure = experimentDisclosurePolicy(
    state.network.blind_mode,
    state.experiments.length,
  );

  const importConfiguration = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    setImportError(null);
    setImportNotice(null);
    if (!file) return;

    try {
      if (file.size > MAX_CONFIGURATION_FILE_BYTES) {
        throw new Error("The configuration file must be 2 MB or smaller.");
      }
      const configuration = parseNetworkConfigurationJson(await file.text());
      dispatch({ type: "NETWORK_CONFIGURATION_IMPORTED", configuration });
      setImportNotice(
        `Loaded “${configuration.session_name}” with ${configuration.nodes.length} sensor ${configuration.nodes.length === 1 ? "node" : "nodes"} as a new draft. No network session was created.`,
      );
    } catch (error) {
      setImportError(
        error instanceof Error
          ? error.message
          : "The configuration could not be imported.",
      );
    } finally {
      input.value = "";
    }
  };

  const exportExperiments = () => {
    if (!disclosure.allowLedgerExport) return;
    const payload = JSON.stringify({
      schema_version: "aqse.workbench.experiments.v1",
      exported_at: new Date().toISOString(),
      records: state.experiments,
    }, null, 2);
    download(payload, `aqse-experiments-${timestamp()}.json`, "application/json");
  };
  const exportConfiguration = () => {
    const configuration = state.network.draft?.value ?? state.network.executed?.value;
    if (!configuration) return;
    download(
      JSON.stringify(configuration, null, 2),
      `aqse-network-configuration-${timestamp()}.json`,
      "application/json",
    );
  };
  const exportFeatures = () => {
    const result = state.features.result;
    if (!result) return;
    const featureColumns = result.profile.feature_names.map((name, index) => `${name} [${result.profile.feature_units[index]}]`);
    const header = ["profile_id", "window_id", "acquisition_id", "sensor_id", "start_time_s", "end_time_s", "quality_status", "valid_for_quantum", "quality_flags", "provenance_token", ...featureColumns];
    const rows = result.windows.map((window) => [
      result.profile.profile_id,
      window.window_id,
      window.acquisition_id,
      window.sensor_id,
      window.start_time_s,
      window.end_time_s,
      window.quality.status,
      window.quality.valid_for_quantum,
      window.quality.flags.join("|"),
      window.provenance_token,
      ...window.features.values,
    ]);
    const csv = [header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n");
    download(csv, `aqse-features-${timestamp()}.csv`, "text/csv;charset=utf-8");
  };

  return (
    <section className="worksheet" aria-labelledby="experiments-title">
      <WorksheetHeader
        titleId="experiments-title"
        index="08"
        eyebrow="LOCAL PROVENANCE LEDGER"
        title="Experiments"
        description="Bounded in-memory history of actions completed in this browser session, including internal immutable request snapshots. Records describe actual requests and never synthesize missing metrics."
        actions={(
          <div className="export-actions">
            <button
              type="button"
              className="button button-secondary"
              onClick={exportConfiguration}
              disabled={state.network.blind_mode || (!state.network.draft && !state.network.executed)}
              title={state.network.blind_mode ? "Disable blind mode before exporting scenario inputs." : "Export the current draft scenario configuration."}
            >
              Export scenario input JSON
            </button>
            <button type="button" className="button button-secondary" onClick={exportFeatures} disabled={!state.features.result}>
              Export features CSV
            </button>
            <button
              type="button"
              className="button button-secondary"
              onClick={exportExperiments}
              disabled={!disclosure.allowLedgerExport}
              aria-describedby={state.network.blind_mode ? "experiment-blind-boundary" : undefined}
              title={state.network.blind_mode ? BLIND_LEDGER_HELP : "Export the local experiment ledger with its request snapshots."}
            >
              Export experiments JSON
            </button>
          </div>
        )}
      />
      <article className="workbench-panel configuration-import-panel" aria-labelledby="configuration-import-title">
        <div className="panel-heading-row">
          <div>
            <p className="panel-kicker">DRAFT CONFIGURATION</p>
            <h2 id="configuration-import-title">Import network configuration</h2>
          </div>
          <span className="chart-unit">JSON · 2 MB maximum</span>
        </div>
        <p className="configuration-import-copy">
          Select a complete AQSE network configuration JSON file. The file may contain scenario truth such as latent-source and scheduled-event definitions. A valid file replaces only the editable draft; it does not create, stop, or modify a network session.
        </p>
        <div className="configuration-import-control">
          <label htmlFor={configurationInputId}>Network configuration JSON</label>
          <input
            id={configurationInputId}
            type="file"
            accept="application/json,.json"
            onChange={importConfiguration}
            aria-describedby={`${configurationInputId}-help`}
          />
          <span id={`${configurationInputId}-help`}>
            Required fields and all 1–8 sensor nodes are validated locally before the draft is updated.
          </span>
        </div>
        {importError && <p className="inline-message error-message" role="alert">{importError}</p>}
        {importNotice && <p className="inline-message success-message" role="status">{importNotice}</p>}
        <p className="configuration-import-boundary">
          Existing feature and quantum results remain visible but become stale when they depend on the previous executed configuration. Create a session explicitly from the Sensors worksheet when ready.
        </p>
        <p className="configuration-import-boundary">
          Configuration exports use the current draft when available. They contain scenario inputs—including configured latent sources and scheduled events—but never realized simulator truth frames. Export is disabled in blind mode; disable blind mode explicitly to export. This local UI barrier is not an API security control.
        </p>
      </article>
      <article className="workbench-panel">
        <div className="panel-heading-row"><div><p className="panel-kicker">SESSION HISTORY</p><h2>{state.experiments.length} records</h2></div><span className="chart-unit">maximum 50 · newest first</span></div>
        {state.network.blind_mode && (
          <p id="experiment-blind-boundary" className="configuration-import-boundary" role="status">
            {BLIND_LEDGER_HELP} Records remain intact in local state.
          </p>
        )}
        {state.experiments.length === 0 ? <p className="empty-copy">No network session, feature extraction, or quantum preview has completed during this browser session.</p> : (
          <div className="experiment-ledger">{state.experiments.map((record) => <ExperimentCard record={record} exposeRequestSnapshot={disclosure.exposeRequestSnapshots} key={record.id} />)}</div>
        )}
      </article>
    </section>
  );
}

export function ExperimentCard({
  record,
  exposeRequestSnapshot,
}: {
  record: ExperimentRecord;
  exposeRequestSnapshot: boolean;
}) {
  return (
    <article className="experiment-card">
      <header><span className="panel-kicker">{record.kind.replaceAll("_", " ")}</span><strong>{record.title}</strong><time dateTime={record.created_at}>{new Date(record.created_at).toLocaleString()}</time></header>
      <dl>
        <Datum label="Status" value={record.status} />
        <Datum label="Seed" value={record.seed} />
        <Datum label="Samples / frames" value={record.sample_count} />
        <Datum label="Feature profile" value={record.feature_profile_id} />
        <Datum label="Valid windows" value={record.valid_window_count} />
        <Datum label="θ snapshot" value={record.theta_snapshot_id} />
        <Datum label="Scaler" value={record.scaler_version} />
      </dl>
      {exposeRequestSnapshot && (
        <details className="experiment-request-snapshot">
          <summary>Requested configuration snapshot</summary>
          <pre>{JSON.stringify(record.request_snapshot, null, 2)}</pre>
        </details>
      )}
      {Object.keys(record.timings_ms).length > 0 && <p className="experiment-timings">{Object.entries(record.timings_ms).map(([name, value]) => `${name}: ${value.toFixed(3)} ms`).join(" · ")}</p>}
      {record.warnings.length > 0 && <p className="inline-message warning-message">{record.warnings.join(" · ")}</p>}
    </article>
  );
}

function Datum({ label, value }: { label: string; value: string | number | undefined }) {
  return <div><dt>{label}</dt><dd>{value ?? "—"}</dd></div>;
}

function download(content: string, filename: string, mimeType: string) {
  const url = URL.createObjectURL(new Blob([content], { type: mimeType }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function timestamp(): string {
  return new Date().toISOString().replaceAll(":", "-");
}

function csvCell(value: unknown): string {
  const text = String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}
