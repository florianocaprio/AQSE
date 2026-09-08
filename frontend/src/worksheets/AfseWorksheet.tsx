import { CapabilityBadge } from "../components/CapabilityBadge";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { useWorkbench } from "../state/workbench";

export function AfseWorksheet() {
  const { state } = useWorkbench();
  const capability = state.capabilities?.local_embedding_afse;
  return (
    <section className="worksheet" aria-labelledby="afse-title">
      <WorksheetHeader
        titleId="afse-title"
        index="06"
        eyebrow="LOCAL EMBEDDING BOUNDARY"
        title="Local Embedding / AFSE"
        description="A modular hand-off surface for future AFSE mathematics supplied or explicitly approved by the project author."
        actions={<CapabilityBadge status={capability?.status ?? "architecture_defined"} />}
      />
      <article className="workbench-panel afse-boundary-panel">
        <div className="afse-mark" aria-hidden="true">AFSE</div>
        <div>
          <p className="panel-kicker">MATHEMATICS PENDING</p>
          <h2>No embedding is computed</h2>
          <p>{capability?.detail ?? "The input/output architecture is reserved; an algorithm has not been supplied."}</p>
          <div className="boundary-flow" aria-label="Reserved AFSE data boundary">
            <span>Kernel artifact</span><i aria-hidden="true">→</i><span className="pending">AFSE · pending</span><i aria-hidden="true">→</i><span>Embedding artifact</span>
          </div>
          <p className="boundary-note">This worksheet intentionally exposes no tunable AFSE parameters, charts, coordinates, quality scores, or inferred outputs. Adding those would imply scientific logic that is not part of Milestone 1C.</p>
        </div>
      </article>
    </section>
  );
}
