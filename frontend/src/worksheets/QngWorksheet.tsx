import { CapabilityBadge } from "../components/CapabilityBadge";
import {
  DemoMetricSummaryTable,
  compactId,
  formatDemoNumber,
} from "../components/DemoReadouts";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { TrainingLoopDiagram } from "../diagrams/TrainingLoopDiagram";
import { useDemo } from "../state/demo";
import { useWorkbench } from "../state/workbench";

export function QngWorksheet() {
  const { state: workbenchState } = useWorkbench();
  const {
    state,
    startTraining,
    cancelTraining,
    applyBundlePair,
    refreshRegistry,
  } = useDemo();
  const capability = workbenchState.capabilities?.qng_training;
  const training = state.training;
  const isTraining = training?.state === "created" || training?.state === "running";
  const canTrain = Boolean(state.registry?.prepared && state.registry.study_artifact_id);
  const canApply = Boolean(
    training?.state === "completed"
    && training.selection_freeze_id
    && training.selected_local_bundle_id
    && training.selected_network_bundle_id,
  );
  const localBundle = state.registry?.bundles.find(
    ({ bundle_id }) => bundle_id === training?.selected_local_bundle_id,
  );
  const networkBundle = state.registry?.bundles.find(
    ({ bundle_id }) => bundle_id === training?.selected_network_bundle_id,
  );

  const applySelection = () => {
    if (
      !training?.selection_freeze_id
      || !training.selected_local_bundle_id
      || !training.selected_network_bundle_id
    ) return;
    void applyBundlePair(
      training.selected_local_bundle_id,
      training.selected_network_bundle_id,
      training.selection_freeze_id,
    );
  };

  return (
    <section className="worksheet" aria-labelledby="qng-title">
      <WorksheetHeader
        titleId="qng-title"
        index="05"
        eyebrow="BOUNDED TRAINING WORKSHEET"
        title="QNG Training"
        description="Run the approved two-candidate bounded training job, inspect accepted updates and validation selection, then apply a compatible local/network bundle pair explicitly. Completion never promotes a model automatically."
        actions={<CapabilityBadge status={canTrain ? "implemented" : capability?.status ?? "available_not_connected"} />}
      />

      {!state.registry?.prepared && (
        <p className="inline-message warning-message" role="status">
          Training requires prepared immutable demo-study artifacts. {state.registry?.preparation_detail ?? "The registry has not been received."}
        </p>
      )}
      {state.training_error && <p className="inline-message error-message" role="alert">{state.training_error}</p>}
      {state.registry_error && <p className="inline-message error-message" role="alert">{state.registry_error}</p>}

      <div className="worksheet-grid two-columns">
        <article className="workbench-panel span-two">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">ACTUAL DEPENDENCY LOOP</p><h2>Protected quantum natural-gradient boundary</h2></div>
            <span className={isTraining ? "result-badge" : "result-badge stale"}>
              {training?.state.toUpperCase() ?? "IDLE"}
            </span>
          </div>
          <TrainingLoopDiagram />
          <p className="boundary-note">TRAIN labels enter only the kernel-alignment loss. They never enter live inference. The Fubini–Study metric is computed from VQC states and derivatives, independently of the loss value.</p>
        </article>

        <article className="workbench-panel">
          <div className="panel-heading-row"><div><p className="panel-kicker">TRAINING CONTROL</p><h2>Idempotent bounded job</h2></div></div>
          <dl className="quality-grid">
            <div><dt>Study</dt><dd title={state.registry?.study_artifact_id ?? undefined}>{compactId(state.registry?.study_artifact_id)}</dd></div>
            <div><dt>Intent</dt><dd title={training?.intent_id}>{compactId(training?.intent_id)}</dd></div>
            <div><dt>Job</dt><dd title={training?.job_id}>{compactId(training?.job_id)}</dd></div>
            <div><dt>State</dt><dd>{training?.state ?? "not created"}</dd></div>
            <div className="span-two"><dt>Stage</dt><dd>{training?.current_stage ?? "—"}</dd></div>
            <div><dt>Created</dt><dd>{training ? formatDate(training.created_at_utc) : "—"}</dd></div>
            <div><dt>Updated</dt><dd>{training ? formatDate(training.updated_at_utc) : "—"}</dd></div>
          </dl>
          <div className="training-actions">
            <button type="button" className="button button-primary" disabled={!canTrain || isTraining || state.training_request === "loading"} onClick={() => void startTraining()}>
              {isTraining ? "Training…" : "Train bounded candidates"}
            </button>
            <button type="button" className="button button-secondary" disabled={!isTraining || state.training_request === "loading"} onClick={() => void cancelTraining()}>
              Cancel
            </button>
            <button type="button" className="button button-secondary" disabled={state.registry_request === "loading"} onClick={() => void refreshRegistry()}>
              Refresh registry
            </button>
          </div>
          <p className="field-help">Retries reuse one execution intent until the job reaches a terminal state. Double-clicks do not request a second scientific trajectory.</p>
        </article>

        <article className="workbench-panel">
          <div className="panel-heading-row"><div><p className="panel-kicker">ACCEPTED-STEP REGISTER</p><h2>Measured optimizer progress</h2></div></div>
          {training?.progress.length ? (
            <div className="table-scroll">
              <table className="data-table">
                <thead><tr><th>Task</th><th>Candidate</th><th>Accepted steps</th><th>Current loss</th></tr></thead>
                <tbody>
                  {training.progress.map((entry) => (
                    <tr key={`${entry.task_id}-${entry.candidate_name}`}>
                      <td>{entry.task_id}</td>
                      <td>{entry.candidate_name}</td>
                      <td>{entry.accepted_updates}</td>
                      <td>{formatDemoNumber(entry.current_loss)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p className="empty-copy">No accepted-step record is available. The interface does not draw synthetic progress curves.</p>}
          {training?.error && <p className="inline-message error-message" role="alert">{training.error}</p>}
        </article>

        <article className="workbench-panel span-two">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">VALIDATION SELECTION</p><h2>Candidate bundle pair</h2></div>
            <span className={canApply ? "result-badge" : "result-badge stale"}>{canApply ? "READY TO APPLY" : "NOT AVAILABLE"}</span>
          </div>
          <dl className="quality-grid">
            <div><dt>Selection freeze</dt><dd title={training?.selection_freeze_id ?? undefined}>{compactId(training?.selection_freeze_id)}</dd></div>
            <div><dt>Current application</dt><dd title={state.registry?.active?.application_id ?? undefined}>{compactId(state.registry?.active?.application_id)}</dd></div>
            <div><dt>Selected local bundle</dt><dd title={training?.selected_local_bundle_id ?? undefined}>{compactId(training?.selected_local_bundle_id)}</dd></div>
            <div><dt>Selected network bundle</dt><dd title={training?.selected_network_bundle_id ?? undefined}>{compactId(training?.selected_network_bundle_id)}</dd></div>
          </dl>
          {(localBundle || networkBundle) && (
            <div className="worksheet-grid two-columns demo-validation-grid">
              {localBundle && <article className="nested-result-card"><p className="panel-kicker">LOCAL VALIDATION</p><h3>{localBundle.theta_candidate_name}</h3><DemoMetricSummaryTable metrics={localBundle.validation} label="Local bundle validation metrics" /></article>}
              {networkBundle && <article className="nested-result-card"><p className="panel-kicker">NETWORK VALIDATION</p><h3>{networkBundle.theta_candidate_name}</h3><DemoMetricSummaryTable metrics={networkBundle.validation} label="Network bundle validation metrics" /></article>}
            </div>
          )}
          <div className="training-actions">
            <button type="button" className="button button-primary" disabled={!canApply || state.registry_request === "loading"} onClick={applySelection}>
              Apply compatible bundle pair
            </button>
          </div>
          <p className="boundary-note">Apply is atomic and explicit. The selected θ, encoder, TQK reference, AFSE map, output scaler, and classifier move together; training completion alone changes no live model.</p>
        </article>
      </div>
    </section>
  );
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}
