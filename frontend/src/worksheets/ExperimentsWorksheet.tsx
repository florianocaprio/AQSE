import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useState,
  type ChangeEvent,
} from "react";

import {
  DemoMetricSummaryTable,
  DemoModelComparisonSummary,
  compactId,
  latestResultForSensor,
} from "../components/DemoReadouts";
import {
  addDemoKnowledgeLabel,
  approveDemoTrainCollection,
  captureDemoKnowledgeObservation,
  getDemoKnowledge,
} from "../api/demo";
import { errorMessage } from "../api/client";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { parseNetworkConfigurationJson } from "../state/configurationImport";
import { useDemo } from "../state/demo";
import { useWorkbench } from "../state/workbench";
import type {
  DemoBundleSummary,
  DemoSelectionFreezeSummary,
  KnowledgeRegistryView,
  KnowledgeTask,
} from "../types/demo";
import type { ExperimentRecord } from "../types/workbench";

const MAX_CONFIGURATION_FILE_BYTES = 2 * 1024 * 1024;
const BLIND_LEDGER_HELP =
  "Experiment request snapshots can contain latent scenario inputs. Disable blind mode to view or export the experiment ledger. Blind mode is a local UI barrier, not an API security control.";
const KNOWLEDGE_CLASS_ORDER: Record<KnowledgeTask, readonly string[]> = {
  local: ["NORMAL", "CHANGE_DETECTED"],
  network: [
    "NORMAL",
    "ENVIRONMENT_COMPATIBLE",
    "DEVICE_COMPATIBLE",
    "MIXED_OR_AMBIGUOUS",
  ],
};

export function experimentDisclosurePolicy(
  blindMode: boolean,
  experimentCount: number,
) {
  return {
    exposeRequestSnapshots: !blindMode,
    allowLedgerExport: !blindMode && experimentCount > 0,
  };
}

export function selectionFreezeForPair(
  freezes: readonly DemoSelectionFreezeSummary[],
  localBundleId: string,
  networkBundleId: string,
): DemoSelectionFreezeSummary | null {
  return freezes.find(
    (freeze) => freeze.local_bundle_id === localBundleId
      && freeze.network_bundle_id === networkBundleId,
  ) ?? null;
}

export function ExperimentsWorksheet() {
  const { state, dispatch } = useWorkbench();
  const {
    state: demoState,
    applyBundlePair,
    startTraining,
  } = useDemo();
  const configurationInputId = useId();
  const [importError, setImportError] = useState<string | null>(null);
  const [importNotice, setImportNotice] = useState<string | null>(null);
  const [localBundleId, setLocalBundleId] = useState("");
  const [networkBundleId, setNetworkBundleId] = useState("");
  const [knowledge, setKnowledge] = useState<KnowledgeRegistryView | null>(null);
  const [knowledgeBusy, setKnowledgeBusy] = useState(false);
  const [knowledgeError, setKnowledgeError] = useState<string | null>(null);
  const [knowledgeNotice, setKnowledgeNotice] = useState<string | null>(null);
  const [selectedObservationId, setSelectedObservationId] = useState("");
  const [reviewLabel, setReviewLabel] = useState("");
  const [reviewerId, setReviewerId] = useState("");
  const [collectionTask, setCollectionTask] = useState<KnowledgeTask>("local");
  const [selectedTrainIds, setSelectedTrainIds] = useState<string[]>([]);
  const [approverId, setApproverId] = useState("");
  const [localTrainCollectionId, setLocalTrainCollectionId] = useState("");
  const [networkTrainCollectionId, setNetworkTrainCollectionId] = useState("");
  const disclosure = experimentDisclosurePolicy(
    state.network.blind_mode,
    state.experiments.length,
  );
  const currentResult = latestResultForSensor(
    demoState.analysis,
    state.network.selected_node_id,
    state.network.session?.session_id ?? null,
  );
  const selectedObservation = knowledge?.observations.find(
    ({ observation_id }) => observation_id === selectedObservationId,
  ) ?? null;
  const reviewedObservationIds = useMemo(
    () => new Set(knowledge?.reviewed_labels.map(({ observation_id }) => observation_id)),
    [knowledge?.reviewed_labels],
  );
  const collectionCandidates = useMemo(
    () => knowledge?.observations.filter(
      (observation) => observation.task === collectionTask
        && reviewedObservationIds.has(observation.observation_id),
    ) ?? [],
    [collectionTask, knowledge?.observations, reviewedObservationIds],
  );
  const compatibleFreeze = useMemo(
    () => selectionFreezeForPair(
      demoState.registry?.selection_freezes ?? [],
      localBundleId,
      networkBundleId,
    ),
    [demoState.registry?.selection_freezes, localBundleId, networkBundleId],
  );

  useEffect(() => {
    if (localBundleId || networkBundleId) return;
    const active = demoState.registry?.active;
    const initial = active
      ? demoState.registry?.selection_freezes.find(
        ({ freeze_id }) => freeze_id === active.selection_freeze_id,
      )
      : demoState.registry?.selection_freezes[0];
    if (!initial) return;
    setLocalBundleId(initial.local_bundle_id);
    setNetworkBundleId(initial.network_bundle_id);
  }, [demoState.registry, localBundleId, networkBundleId]);

  const refreshKnowledge = useCallback(async (signal?: AbortSignal) => {
    setKnowledgeBusy(true);
    setKnowledgeError(null);
    try {
      setKnowledge(await getDemoKnowledge(signal));
    } catch (error: unknown) {
      if (!isAbort(error)) {
        setKnowledgeError(errorMessage(error, "The knowledge registry is unavailable."));
      }
    } finally {
      if (!signal?.aborted) setKnowledgeBusy(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refreshKnowledge(controller.signal);
    return () => controller.abort();
  }, [refreshKnowledge]);

  const captureCurrentObservation = async () => {
    if (!currentResult) return;
    setKnowledgeBusy(true);
    setKnowledgeError(null);
    setKnowledgeNotice(null);
    try {
      const response = await captureDemoKnowledgeObservation({
        schema_version: "aqse.network-demo.knowledge-capture-request.v1",
        session_id: currentResult.session_id,
        sensor_id: currentResult.sensor_id,
        task: currentResult.task_id === "aqse.local-change.v1" ? "local" : "network",
      });
      setSelectedObservationId(response.observation.observation_id);
      setReviewLabel("");
      setKnowledgeNotice(
        response.reused
          ? "The immutable observation already existed; no duplicate was written."
          : "Observation captured without prediction or scenario truth.",
      );
      await refreshKnowledge();
    } catch (error: unknown) {
      setKnowledgeError(errorMessage(error, "The observation could not be captured."));
      setKnowledgeBusy(false);
    }
  };

  const saveHumanReview = async () => {
    if (!selectedObservation || !reviewLabel || !reviewerId.trim()) return;
    setKnowledgeBusy(true);
    setKnowledgeError(null);
    setKnowledgeNotice(null);
    try {
      const response = await addDemoKnowledgeLabel({
        schema_version: "aqse.network-demo.reviewed-label-request.v1",
        observation_id: selectedObservation.observation_id,
        task: selectedObservation.task,
        label: reviewLabel,
        reviewer_id: reviewerId.trim(),
        reviewed_at_utc: new Date().toISOString(),
      });
      setKnowledgeNotice(
        response.reused
          ? "This immutable human review already existed."
          : "Human-reviewed label saved in its separate supervision channel.",
      );
      await refreshKnowledge();
    } catch (error: unknown) {
      setKnowledgeError(errorMessage(error, "The human-reviewed label could not be saved."));
      setKnowledgeBusy(false);
    }
  };

  const approveTrainCollection = async () => {
    if (!approverId.trim() || selectedTrainIds.length === 0) return;
    setKnowledgeBusy(true);
    setKnowledgeError(null);
    setKnowledgeNotice(null);
    try {
      const response = await approveDemoTrainCollection({
        schema_version: "aqse.network-demo.train-approval-request.v1",
        partition: "TRAIN",
        task: collectionTask,
        observation_ids: [...selectedTrainIds].sort(),
        approved_by: approverId.trim(),
        approved_at_utc: new Date().toISOString(),
        approval_declaration: "explicitly approved for bounded TRAIN-only retraining",
      });
      setKnowledgeNotice(
        response.reused
          ? "This immutable TRAIN collection already existed."
          : "TRAIN-only collection approved. No training or bundle application was started.",
      );
      await refreshKnowledge();
    } catch (error: unknown) {
      setKnowledgeError(errorMessage(error, "The TRAIN collection could not be approved."));
      setKnowledgeBusy(false);
    }
  };

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
  const exportDemoRegistry = () => {
    if (!demoState.registry) return;
    download(
      JSON.stringify(demoState.registry, null, 2),
      `aqse-demo-registry-${timestamp()}.json`,
      "application/json",
    );
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
        description="Inspect the persistent read-only demo study and bundle registry alongside bounded actions completed in this browser session. Opening the worksheet performs no scientific computation."
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
            <button type="button" className="button button-secondary" onClick={exportDemoRegistry} disabled={!demoState.registry}>
              Export demo registry JSON
            </button>
          </div>
        )}
      />

      {demoState.registry_error && <p className="inline-message error-message" role="alert">{demoState.registry_error}</p>}
      <div className="worksheet-grid two-columns demo-registry-layout">
        <article className="workbench-panel">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">BOUNDED DEMO STUDY</p><h2>{demoState.registry?.study_artifact_id ?? "Registry unavailable"}</h2></div>
            {demoState.registry && <span className={demoState.registry.prepared ? "result-badge" : "result-badge stale"}>{demoState.registry.prepared ? "PREPARED" : "NOT PREPARED"}</span>}
          </div>
          <dl className="quality-grid">
            <div><dt>Study digest</dt><dd title={demoState.registry?.study_content_digest ?? undefined}>{compactId(demoState.registry?.study_content_digest)}</dd></div>
            <div><dt>Selection freeze</dt><dd title={demoState.registry?.selection_freeze_id ?? undefined}>{compactId(demoState.registry?.selection_freeze_id)}</dd></div>
            <div><dt>Final evaluation</dt><dd title={demoState.registry?.final_evaluation_id ?? undefined}>{compactId(demoState.registry?.final_evaluation_id)}</dd></div>
            <div><dt>Saved bundles</dt><dd>{demoState.registry?.bundles.length ?? "—"}</dd></div>
            <div><dt>Compatible saved pairs</dt><dd>{demoState.registry?.selection_freezes.length ?? "—"}</dd></div>
            <div className="span-two"><dt>Historical TEST ledger hash</dt><dd title={demoState.registry?.historical_test_ledger_sha256}>{compactId(demoState.registry?.historical_test_ledger_sha256)}</dd></div>
          </dl>
          <p className="boundary-note">{demoState.registry?.scientific_label ?? "No scientific registry label received."} Opening this worksheet never re-runs TEST, training, or model selection.</p>
          {demoState.registry && !demoState.registry.prepared && <p className="inline-message warning-message">{demoState.registry.preparation_detail}</p>}
        </article>

        <article className="workbench-panel">
          <div className="panel-heading-row"><div><p className="panel-kicker">EXPLICIT APPLICATION</p><h2>Compatible bundle pair</h2></div><span className="result-badge stale">MANUAL ONLY</span></div>
          <label className="input-control full"><span className="input-label">Local task bundle</span>
            <select aria-label="Local task bundle" value={localBundleId} onChange={(event) => setLocalBundleId(event.target.value)}>
              <option value="">Select a saved local bundle</option>
              {demoState.registry?.bundles.filter(({ task_id }) => task_id === "aqse.local-change.v1").map((bundle) => <option value={bundle.bundle_id} key={bundle.bundle_id}>{bundle.bundle_id}{bundle.active ? " · active" : ""}</option>)}
            </select>
          </label>
          <label className="input-control full"><span className="input-label">Network task bundle</span>
            <select aria-label="Network task bundle" value={networkBundleId} onChange={(event) => setNetworkBundleId(event.target.value)}>
              <option value="">Select a saved network bundle</option>
              {demoState.registry?.bundles.filter(({ task_id }) => task_id === "aqse.network-pattern.v1").map((bundle) => <option value={bundle.bundle_id} key={bundle.bundle_id}>{bundle.bundle_id}{bundle.active ? " · active" : ""}</option>)}
            </select>
          </label>
          <button
            type="button"
            className="button button-primary full-button"
            disabled={!compatibleFreeze || demoState.registry_request === "loading"}
            onClick={() => {
              if (!compatibleFreeze) return;
              void applyBundlePair(localBundleId, networkBundleId, compatibleFreeze.freeze_id);
            }}
          >
            Apply selected compatible pair
          </button>
          <p className="field-help">
            {compatibleFreeze
              ? `Compatible immutable selection freeze: ${compatibleFreeze.freeze_id}.`
              : "Choose a local/network pair published by the same immutable selection freeze."}
            {" "}Selection is never inferred from TEST metrics. The backend validates compatibility and applies the pair atomically.
          </p>
        </article>
      </div>

      <article className="workbench-panel demo-saved-bundles-panel">
        <div className="panel-heading-row"><div><p className="panel-kicker">SAVED MODEL BUNDLES</p><h2>{demoState.registry?.bundles.length ?? 0} versioned artifacts</h2></div><span className="chart-unit">read-only registry</span></div>
        {demoState.registry?.bundles.length ? (
          <div className="demo-bundle-ledger">
            {demoState.registry.bundles.map((bundle) => <DemoBundleCard bundle={bundle} key={bundle.bundle_id} />)}
          </div>
        ) : <p className="empty-copy">No saved demo bundles are available. No placeholder metrics are shown.</p>}
      </article>

      <article className="workbench-panel demo-final-metrics-panel">
        <div className="panel-heading-row"><div><p className="panel-kicker">FROZEN FINAL EVALUATION</p><h2>{demoState.registry?.final_evaluation_id ?? "No final evaluation published"}</h2></div><span className="chart-unit">read-only · no rerun</span></div>
        {demoState.registry?.final_metrics.length ? (
          <>
            <div className="worksheet-grid two-columns demo-validation-grid">
              {demoState.registry.final_metrics.map((metrics, index) => (
                <article className="nested-result-card" key={`${metrics.task_id}-${metrics.partition}-afse`}>
                  <p className="panel-kicker">QUANTUM-AFSE MODEL · FINAL {index + 1}</p>
                  <DemoMetricSummaryTable metrics={metrics} label={`Frozen AFSE final metric set ${index + 1}`} />
                </article>
              ))}
              {demoState.registry.final_raw_baseline_metrics.map((metrics, index) => (
                <article className="nested-result-card" key={`${metrics.task_id}-${metrics.partition}-raw`}>
                  <p className="panel-kicker">RAW STATE8 MLP · FINAL {index + 1}</p>
                  <DemoMetricSummaryTable metrics={metrics} label={`Frozen raw baseline final metric set ${index + 1}`} />
                </article>
              ))}
            </div>
            <div className="worksheet-grid two-columns demo-validation-grid model-comparison-grid">
              {demoState.registry.final_comparisons.map((comparison) => (
                <article className="nested-result-card" key={`${comparison.task_id}-${comparison.partition}`}>
                  <DemoModelComparisonSummary comparison={comparison} label={`${comparison.task_id} frozen final paired comparison`} />
                </article>
              ))}
            </div>
          </>
        ) : <p className="empty-copy">No frozen final metric set is present in the registry.</p>}
      </article>

      <article className="workbench-panel knowledge-registry-panel" aria-labelledby="knowledge-registry-title">
        <div className="panel-heading-row">
          <div>
            <p className="panel-kicker">BOUNDED KNOWLEDGE REGISTRY</p>
            <h2 id="knowledge-registry-title">Observation → human review → TRAIN approval</h2>
          </div>
          <button type="button" className="button button-secondary" disabled={knowledgeBusy} onClick={() => void refreshKnowledge()}>
            Refresh registry
          </button>
        </div>
        <p className="boundary-note">
          Captures contain observable State8 data only. Model output is never accepted as a label. Human review and TRAIN approval are separate explicit writes; neither action starts retraining or applies a bundle.
        </p>
        {knowledgeError && <p className="inline-message error-message" role="alert">{knowledgeError}</p>}
        {knowledgeNotice && <p className="inline-message success-message" role="status">{knowledgeNotice}</p>}
        <dl className="quality-grid">
          <div><dt>Observation episodes</dt><dd>{knowledge?.observations.length ?? "—"}</dd></div>
          <div><dt>Human-reviewed labels</dt><dd>{knowledge?.reviewed_labels.length ?? "—"}</dd></div>
          <div><dt>Approved TRAIN collections</dt><dd>{knowledge?.approved_train_collections.length ?? "—"}</dd></div>
          <div><dt>Current eligible result</dt><dd>{currentResult?.feature_valid ? currentResult.sensor_id : "none"}</dd></div>
        </dl>

        <div className="worksheet-grid two-columns knowledge-actions-grid">
          <section className="nested-result-card" aria-labelledby="knowledge-capture-title">
            <p className="panel-kicker">STEP 1 · OBSERVATION ONLY</p>
            <h3 id="knowledge-capture-title">Capture current analyzed window</h3>
            <dl className="quality-grid">
              <div><dt>Session</dt><dd>{compactId(currentResult?.session_id)}</dd></div>
              <div><dt>Sensor</dt><dd>{currentResult?.sensor_id ?? "—"}</dd></div>
              <div><dt>Task</dt><dd>{currentResult?.task_id ?? "—"}</dd></div>
              <div><dt>Window</dt><dd>{compactId(currentResult?.window_id)}</dd></div>
            </dl>
            <button
              type="button"
              className="button button-primary full-button"
              disabled={knowledgeBusy || !currentResult?.feature_valid}
              onClick={() => void captureCurrentObservation()}
            >
              Capture observation without label
            </button>
            <p className="field-help">The MLP class, raw baseline class, observable-rule status and simulator truth are excluded.</p>
          </section>

          <section className="nested-result-card" aria-labelledby="knowledge-review-title">
            <p className="panel-kicker">STEP 2 · EXPLICIT HUMAN REVIEW</p>
            <h3 id="knowledge-review-title">Assign a separate supervisory label</h3>
            <label className="input-control full"><span className="input-label">Observation</span>
              <select
                aria-label="Observation for human review"
                value={selectedObservationId}
                onChange={(event) => {
                  setSelectedObservationId(event.target.value);
                  setReviewLabel("");
                }}
              >
                <option value="">Select an immutable observation</option>
                {knowledge?.observations.map((observation) => (
                  <option value={observation.observation_id} key={observation.observation_id}>
                    {observation.observation_id} · {observation.task} · {observation.feature.sensor_id}
                  </option>
                ))}
              </select>
            </label>
            <label className="input-control full"><span className="input-label">Human-reviewed class</span>
              <select aria-label="Human-reviewed class" value={reviewLabel} onChange={(event) => setReviewLabel(event.target.value)} disabled={!selectedObservation}>
                <option value="">Select after reviewing external evidence</option>
                {(selectedObservation ? KNOWLEDGE_CLASS_ORDER[selectedObservation.task] : []).map((label) => <option value={label} key={label}>{label}</option>)}
              </select>
            </label>
            <label className="input-control full"><span className="input-label">Reviewer identifier</span>
              <input value={reviewerId} maxLength={128} onChange={(event) => setReviewerId(event.target.value)} placeholder="Human reviewer ID" />
            </label>
            <button
              type="button"
              className="button button-primary full-button"
              disabled={knowledgeBusy || !selectedObservation || !reviewLabel || !reviewerId.trim() || reviewedObservationIds.has(selectedObservationId)}
              onClick={() => void saveHumanReview()}
            >
              Save human-reviewed label
            </button>
            <p className="field-help">The UI never copies the model prediction into this field. Existing immutable labels cannot be overwritten.</p>
          </section>
        </div>

        <section className="nested-result-card knowledge-train-approval" aria-labelledby="knowledge-approval-title">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">STEP 3 · EXPLICIT TRAIN-ONLY APPROVAL</p><h3 id="knowledge-approval-title">Freeze selected reviewed episodes</h3></div>
            <span className="result-badge stale">NO AUTO-TRAINING</span>
          </div>
          <div className="control-grid">
            <label className="input-control full"><span className="input-label">Task</span>
              <select
                aria-label="Knowledge TRAIN task"
                value={collectionTask}
                onChange={(event) => {
                  setCollectionTask(event.target.value as KnowledgeTask);
                  setSelectedTrainIds([]);
                }}
              >
                <option value="local">Local change</option>
                <option value="network">Network pattern</option>
              </select>
            </label>
            <label className="input-control full"><span className="input-label">Approver identifier</span>
              <input value={approverId} maxLength={128} onChange={(event) => setApproverId(event.target.value)} placeholder="Human approver ID" />
            </label>
          </div>
          {collectionCandidates.length ? (
            <fieldset className="knowledge-candidate-list">
              <legend>Reviewed observations eligible for explicit TRAIN approval</legend>
              {collectionCandidates.map((observation) => (
                <label className="toggle-control" key={observation.observation_id}>
                  <input
                    type="checkbox"
                    checked={selectedTrainIds.includes(observation.observation_id)}
                    onChange={() => setSelectedTrainIds((current) => toggleId(current, observation.observation_id))}
                  />
                  <span><strong>{observation.observation_id}</strong><small>{observation.feature.sensor_id} · {observation.node_count} node(s) · human label recorded separately</small></span>
                </label>
              ))}
            </fieldset>
          ) : <p className="empty-copy">No human-reviewed observations are available for this task.</p>}
          <button
            type="button"
            className="button button-primary full-button"
            disabled={knowledgeBusy || !approverId.trim() || selectedTrainIds.length === 0}
            onClick={() => void approveTrainCollection()}
          >
            Approve selected episodes for TRAIN only
          </button>
          <p className="field-help">Approval creates a content-addressed TRAIN collection. The previous active model remains active until a separately trained compatible pair is explicitly applied.</p>
        </section>

        {knowledge?.approved_train_collections.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Collection</th><th>Task</th><th>Episodes</th><th>Approved by</th><th>Partition</th></tr></thead>
              <tbody>
                {knowledge.approved_train_collections.map((collection) => (
                  <tr key={collection.collection_id}>
                    <td title={collection.collection_id}>{compactId(collection.collection_id)}</td>
                    <td>{collection.task}</td>
                    <td>{collection.entries.length}</td>
                    <td>{collection.approved_by}</td>
                    <td>{collection.partition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}

        <section className="nested-result-card knowledge-retraining" aria-labelledby="knowledge-retraining-title">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">STEP 4 · BOUNDED RETRAINING</p><h3 id="knowledge-retraining-title">Start one explicit compatible collection pair</h3></div>
            <span className="result-badge stale">MANUAL ONLY</span>
          </div>
          <div className="control-grid">
            <label className="input-control full"><span className="input-label">Local TRAIN collection</span>
              <select aria-label="Local TRAIN collection" value={localTrainCollectionId} onChange={(event) => setLocalTrainCollectionId(event.target.value)}>
                <option value="">Select approved local collection</option>
                {knowledge?.approved_train_collections.filter(({ task }) => task === "local").map((collection) => <option value={collection.collection_id} key={collection.collection_id}>{collection.collection_id}</option>)}
              </select>
            </label>
            <label className="input-control full"><span className="input-label">Network TRAIN collection</span>
              <select aria-label="Network TRAIN collection" value={networkTrainCollectionId} onChange={(event) => setNetworkTrainCollectionId(event.target.value)}>
                <option value="">Select approved network collection</option>
                {knowledge?.approved_train_collections.filter(({ task }) => task === "network").map((collection) => <option value={collection.collection_id} key={collection.collection_id}>{collection.collection_id}</option>)}
              </select>
            </label>
          </div>
          <button
            type="button"
            className="button button-primary full-button"
            disabled={!localTrainCollectionId || !networkTrainCollectionId || demoState.training_request === "loading" || demoState.training?.state === "running"}
            onClick={() => void startTraining({
              local_train_collection_id: localTrainCollectionId,
              network_train_collection_id: networkTrainCollectionId,
            })}
          >
            Start bounded retraining from approved collections
          </button>
          <p className="field-help">This submits an idempotent bounded job. It does not promote the result: the previous active pair remains in service until a compatible selection freeze is explicitly applied.</p>
        </section>
      </article>

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

export function DemoBundleCard({ bundle }: { bundle: DemoBundleSummary }) {
  return (
    <article className="demo-bundle-card">
      <header>
        <span className="panel-kicker">{bundle.task_id}</span>
        <strong>{bundle.bundle_id}</strong>
        <span className={bundle.active ? "result-badge" : "result-badge stale"}>{bundle.active ? "ACTIVE" : "SAVED"}</span>
      </header>
      <dl className="quality-grid">
        <div><dt>Profile</dt><dd>{bundle.profile_id}</dd></div>
        <div><dt>θ candidate</dt><dd>{bundle.theta_candidate_name}</dd></div>
        <div><dt>Accepted updates</dt><dd>{bundle.accepted_qng_updates}</dd></div>
        <div><dt>AFSE</dt><dd>{bundle.afse_method_id}</dd></div>
        <div><dt>M / λ</dt><dd>{bundle.afse_reference_size} / {bundle.afse_ridge_lambda}</dd></div>
        <div><dt>Classifier</dt><dd>{bundle.classifier_model_id}</dd></div>
      </dl>
      <DemoMetricSummaryTable metrics={bundle.validation} label={`${bundle.task_id} validation metrics`} />
      <details className="metric-node-details">
        <summary>Same-architecture raw State8 MLP baseline</summary>
        <DemoMetricSummaryTable metrics={bundle.raw_baseline_validation} label={`${bundle.task_id} raw baseline validation metrics`} />
      </details>
      <DemoModelComparisonSummary comparison={bundle.validation_comparison} label={`${bundle.task_id} validation comparison`} />
    </article>
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

function toggleId(current: string[], id: string): string[] {
  return current.includes(id)
    ? current.filter((candidate) => candidate !== id)
    : [...current, id];
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
