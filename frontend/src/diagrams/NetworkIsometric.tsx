import type { DipoleSourceConfiguration, SensorNodeConfiguration } from "../types/network";

type NetworkIsometricProps = {
  nodes: SensorNodeConfiguration[];
  dipoles: DipoleSourceConfiguration[];
  selectedNodeId: string | null;
};

export function NetworkIsometric({ nodes, dipoles, selectedNodeId }: NetworkIsometricProps) {
  const width = 720;
  const height = 400;
  const allPoints = [
    ...nodes.map(({ position_m }) => position_m),
    ...dipoles.filter(({ enabled }) => enabled).map(({ initial_position_m }) => initial_position_m),
  ];
  const extent = Math.max(1, ...allPoints.flatMap(([x, y, z]) => [Math.abs(x), Math.abs(y), Math.abs(z)]));
  const scale = 120 / extent;
  const project = ([x, y, z]: readonly [number, number, number]) => ({
    x: width / 2 + (x - y) * scale,
    y: height / 2 + (x + y) * scale * 0.45 + z * scale,
  });
  const origin = project([0, 0, 0]);
  const axes = [
    { label: "N / X", point: project([extent, 0, 0]), className: "axis-n" },
    { label: "E / Y", point: project([0, extent, 0]), className: "axis-e" },
    { label: "D / Z", point: project([0, 0, extent]), className: "axis-d" },
  ];

  return (
    <figure className="network-plane-card">
      <figcaption>
        <span><strong>Sensor geometry · isometric 3D</strong><small>Read-only projection of the current draft; no coverage or observability claim.</small></span>
        <span className="chart-unit">N / E / D frame</span>
      </figcaption>
      <svg className="network-isometric" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Isometric projection of sensor and dipole positions">
        <rect width={width} height={height} className="network-plane-bg" />
        {axes.map(({ label, point, className }) => <g key={label}><line x1={origin.x} y1={origin.y} x2={point.x} y2={point.y} className={`isometric-axis ${className}`} /><text x={point.x + 7} y={point.y - 5} className="network-axis-label">{label}</text></g>)}
        {nodes.map((node) => {
          const point = project(node.position_m);
          const ground = project([node.position_m[0], node.position_m[1], 0]);
          return <g key={node.sensor_id}><line x1={ground.x} y1={ground.y} x2={point.x} y2={point.y} className="height-guide" /><circle cx={point.x} cy={point.y} r={node.sensor_id === selectedNodeId ? 10 : 7} className={node.sensor_id === selectedNodeId ? "iso-node selected" : "iso-node"}><title>{node.sensor_id}: [{node.position_m.join(", ")}] m</title></circle><text x={point.x + 12} y={point.y + 4} className="network-node-label">{node.sensor_id}</text></g>;
        })}
        {dipoles.filter(({ enabled }) => enabled).map((dipole) => {
          const point = project(dipole.initial_position_m);
          return <g key={dipole.source_id} transform={`translate(${point.x} ${point.y})`}><path d="M -8 0 L 0 -11 L 8 0 L 0 11 Z" className="network-dipole"><title>{dipole.source_id}: [{dipole.initial_position_m.join(", ")}] m</title></path></g>;
        })}
      </svg>
    </figure>
  );
}
