import type { CapabilityStatus } from "../types/workbench";

type Stage = {
  label: string;
  detail: string;
  status: CapabilityStatus;
};

type PipelineDiagramProps = {
  stages?: Stage[];
};

const DEFAULT_STAGES: Stage[] = [
  { label: "Sensor", detail: "observations", status: "implemented" },
  { label: "Signal", detail: "causal processing", status: "implemented" },
  { label: "Features", detail: "8D vectors", status: "implemented" },
  { label: "VQC", detail: "8 qubits", status: "implemented" },
  { label: "TQK", detail: "kernel geometry", status: "implemented" },
  { label: "AFSE", detail: "frozen embedding", status: "implemented" },
  { label: "Neural", detail: "frozen model", status: "implemented" },
  { label: "Output", detail: "conditional result", status: "implemented" },
];

export function PipelineDiagram({ stages = DEFAULT_STAGES }: PipelineDiagramProps) {
  return (
    <div className="pipeline-diagram" aria-label="Canonical AQSE inference architecture">
      {stages.map((stage, index) => (
        <div className="pipeline-stage-wrap" key={stage.label}>
          <div className={`pipeline-stage ${stage.status}`}>
            <strong>{stage.label}</strong>
            <span>{stage.detail}</span>
          </div>
          {index < stages.length - 1 && (
            <span className="pipeline-arrow" aria-hidden="true">→</span>
          )}
        </div>
      ))}
    </div>
  );
}
