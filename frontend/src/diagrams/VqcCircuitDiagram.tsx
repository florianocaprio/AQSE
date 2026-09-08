import type { CircuitDescription } from "../types/quantum";

type VqcCircuitDiagramProps = {
  circuit: CircuitDescription | null;
};

const STAGES = [
  ["feature_upload_1", "Ry data I"],
  ["alpha", "Rz α"],
  ["entanglement", "CZ graph"],
  ["beta", "Ry β"],
  ["feature_upload_2", "Rz data II"],
] as const;

export function VqcCircuitDiagram({ circuit }: VqcCircuitDiagramProps) {
  if (!circuit) {
    return <div className="empty-chart">Circuit metadata is unavailable. No topology is inferred.</div>;
  }

  const width = 860;
  const height = 310;
  const left = 72;
  const top = 48;
  const rowGap = 28;
  const stageGap = 145;

  return (
    <figure className="technical-diagram-card">
      <figcaption>
        <span>
          <strong>{circuit.name}</strong>
          <small>Rendered from the backend circuit description; no circuit synthesis in the UI.</small>
        </span>
        <span className="chart-unit">{circuit.qubits} qubits · {circuit.trainable_parameters} θ</span>
      </figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${circuit.name} circuit structure`}>
        {Array.from({ length: circuit.qubits }, (_, qubit) => {
          const y = top + qubit * rowGap;
          return (
            <g key={qubit}>
              <text x="18" y={y + 4} className="circuit-wire-label">q{qubit}</text>
              <line x1={left} x2={width - 24} y1={y} y2={y} className="circuit-wire" />
            </g>
          );
        })}
        {STAGES.map(([stage, label], stageIndex) => {
          const x = left + 54 + stageIndex * stageGap;
          const count = circuit.operations.filter((operation) => operation.stage === stage).length;
          return (
            <g key={stage}>
              <rect x={x - 38} y={top - 24} width="82" height={rowGap * 7 + 48} rx="8" className={`circuit-stage circuit-${stage}`} />
              <text x={x + 3} y={top - 8} textAnchor="middle" className="circuit-stage-label">{label}</text>
              <text x={x + 3} y={top + rowGap * 7 + 34} textAnchor="middle" className="circuit-stage-count">{count} ops</text>
            </g>
          );
        })}
        {circuit.cz_edges.map(([from, to], index) => {
          const x = left + 54 + 2 * stageGap + ((index % 2) - 0.5) * 22;
          const fromY = top + from * rowGap;
          const toY = top + to * rowGap;
          return (
            <g key={`${from}-${to}`}>
              <line x1={x} x2={x} y1={fromY} y2={toY} className="cz-edge" />
              <circle cx={x} cy={fromY} r="4" className="cz-node" />
              <circle cx={x} cy={toY} r="4" className="cz-node" />
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
