import { CapabilityBadge } from "../components/CapabilityBadge";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { useWorkbench } from "../state/workbench";

export function NeuralWorksheet() {
  const { state } = useWorkbench();
  const capability = state.capabilities?.neural_model;
  return (
    <section className="worksheet" aria-labelledby="neural-title">
      <WorksheetHeader
        titleId="neural-title"
        index="07"
        eyebrow="DOWNSTREAM MODEL"
        title="Neural Model"
        description="Reserved worksheet for a future classical task model and calibrated outputs. No architecture, labels, or performance values are assumed."
        actions={<CapabilityBadge status={capability?.status ?? "not_implemented"} />}
      />
      <div className="worksheet-grid two-columns">
        <article className="workbench-panel">
          <p className="panel-kicker">MODEL CONTRACT</p><h2>Not implemented</h2>
          <p className="empty-copy">{capability?.detail ?? "No classical model contract has been published."}</p>
          <dl className="quality-grid"><div><dt>Model identifier</dt><dd>—</dd></div><div><dt>Task</dt><dd>—</dd></div><div><dt>Input schema</dt><dd>—</dd></div><div><dt>Calibration</dt><dd>—</dd></div></dl>
        </article>
        <article className="workbench-panel">
          <p className="panel-kicker">OUTPUT ENGINE</p><h2>No prediction stream</h2>
          <p className="boundary-note">The sensor charts show measured observations only. Simulator truth is a separate validation channel. Neither is presented as a prediction, classification, localization, or tracking result.</p>
        </article>
      </div>
    </section>
  );
}
