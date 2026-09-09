import { CapabilityBadge } from "../components/CapabilityBadge";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { TrainingLoopDiagram } from "../diagrams/TrainingLoopDiagram";
import { useWorkbench } from "../state/workbench";

export function QngWorksheet() {
  const { state } = useWorkbench();
  const capability = state.capabilities?.qng_training;
  return (
    <section className="worksheet" aria-labelledby="qng-title">
      <WorksheetHeader
        titleId="qng-title"
        index="05"
        eyebrow="TRAINING WORKSHEET"
        title="QNG Training"
        description="Reserved control surface for a future sensor-integrated training workflow. The current frontend cannot initiate, resume, or report QNG optimization."
        actions={<CapabilityBadge status={capability?.status ?? "available_not_connected"} />}
      />
      <div className="worksheet-grid two-columns">
        <article className="workbench-panel span-two">
          <div className="panel-heading-row"><div><p className="panel-kicker">DECLARED LOOP</p><h2>Quantum natural-gradient boundary</h2></div><span className="result-badge stale">NOT CONNECTED</span></div>
          <TrainingLoopDiagram />
        </article>
        <article className="workbench-panel">
          <p className="panel-kicker">CURRENT STATUS</p>
          <h2>Scientific engine available; sensor training not connected</h2>
          <p className="empty-copy">{capability?.detail ?? "QNG capability metadata has not been received."}</p>
          <dl className="quality-grid"><div><dt>Training dataset</dt><dd>not configured</dd></div><div><dt>Objective</dt><dd>not configured</dd></div><div><dt>Optimizer state</dt><dd>not created</dd></div><div><dt>θ updates</dt><dd>none</dd></div></dl>
        </article>
        <article className="workbench-panel">
          <p className="panel-kicker">GUARDRAIL</p>
          <h2>Preview isolation</h2>
          <p className="boundary-note">Runs from the Quantum Engine worksheet are fixed-θ infrastructure previews. They do not populate this worksheet, calculate gradients, construct a Fubini–Study metric, or modify the local θ draft.</p>
        </article>
      </div>
    </section>
  );
}
