import { useRef, type KeyboardEvent, type PointerEvent } from "react";

import type { DipoleSourceConfiguration, SensorNodeConfiguration } from "../types/network";

type NetworkPlaneProps = {
  nodes: SensorNodeConfiguration[];
  dipoles: DipoleSourceConfiguration[];
  selectedNodeId: string | null;
  disabled?: boolean;
  onSelect: (sensorId: string) => void;
  onMove: (sensorId: string, x: number, y: number) => void;
};

const WIDTH = 720;
const HEIGHT = 440;
const PADDING = 46;

export function NetworkPlane({
  nodes,
  dipoles,
  selectedNodeId,
  disabled = false,
  onSelect,
  onMove,
}: NetworkPlaneProps) {
  const activePointer = useRef<{ pointerId: number; sensorId: string } | null>(null);
  const extent = Math.max(
    1,
    ...nodes.flatMap(({ position_m }) => [Math.abs(position_m[0]), Math.abs(position_m[1])]),
    ...dipoles.flatMap(({ initial_position_m }) => [
      Math.abs(initial_position_m[0]),
      Math.abs(initial_position_m[1]),
    ]),
  ) * 1.25;

  const toSvgX = (x: number) => PADDING + ((x + extent) / (2 * extent)) * (WIDTH - 2 * PADDING);
  const toSvgY = (y: number) => HEIGHT - PADDING - ((y + extent) / (2 * extent)) * (HEIGHT - 2 * PADDING);
  const fromSvg = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    const rect = svg.getBoundingClientRect();
    const localX = ((clientX - rect.left) / rect.width) * WIDTH;
    const localY = ((clientY - rect.top) / rect.height) * HEIGHT;
    return {
      x: ((localX - PADDING) / (WIDTH - 2 * PADDING)) * 2 * extent - extent,
      y: ((HEIGHT - PADDING - localY) / (HEIGHT - 2 * PADDING)) * 2 * extent - extent,
    };
  };

  const startDrag = (event: PointerEvent<SVGGElement>, sensorId: string) => {
    if (disabled) return;
    event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId);
    activePointer.current = { pointerId: event.pointerId, sensorId };
    onSelect(sensorId);
  };

  const moveDrag = (event: PointerEvent<SVGSVGElement>) => {
    if (disabled || activePointer.current?.pointerId !== event.pointerId) return;
    const point = fromSvg(event.clientX, event.clientY, event.currentTarget);
    onMove(
      activePointer.current.sensorId,
      clamp(point.x, -extent, extent),
      clamp(point.y, -extent, extent),
    );
  };

  const moveWithKeyboard = (
    event: KeyboardEvent<SVGGElement>,
    node: SensorNodeConfiguration,
  ) => {
    if (disabled) return;
    const step = event.shiftKey ? 0.5 : 0.05;
    const [x, y] = node.position_m;
    const delta = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowDown: [0, -step],
      ArrowUp: [0, step],
    }[event.key];
    if (!delta) return;
    event.preventDefault();
    onSelect(node.sensor_id);
    onMove(node.sensor_id, x + delta[0], y + delta[1]);
  };

  return (
    <figure className="network-plane-card">
      <figcaption>
        <span>
          <strong>Sensor layout · XY plane</strong>
          <small>Drag nodes or focus and use arrow keys. Shift = 0.5 m.</small>
        </span>
        <span className="chart-unit">metres</span>
      </figcaption>
      <svg
        className="network-plane"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Interactive top view of configured sensor and dipole positions"
        onPointerMove={moveDrag}
        onPointerUp={() => {
          activePointer.current = null;
        }}
        onPointerCancel={() => {
          activePointer.current = null;
        }}
      >
        <defs>
          <pattern id="network-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" className="network-grid-line" />
          </pattern>
        </defs>
        <rect x="0" y="0" width={WIDTH} height={HEIGHT} className="network-plane-bg" />
        <rect
          x={PADDING}
          y={PADDING}
          width={WIDTH - 2 * PADDING}
          height={HEIGHT - 2 * PADDING}
          fill="url(#network-grid)"
        />
        <line x1={PADDING} x2={WIDTH - PADDING} y1={toSvgY(0)} y2={toSvgY(0)} className="network-axis" />
        <line x1={toSvgX(0)} x2={toSvgX(0)} y1={PADDING} y2={HEIGHT - PADDING} className="network-axis" />
        <text x={WIDTH - PADDING} y={toSvgY(0) - 8} className="network-axis-label">+X</text>
        <text x={toSvgX(0) + 8} y={PADDING + 12} className="network-axis-label">+Y</text>

        {nodes.slice(1).map((node, index) => {
          const previous = nodes[index];
          return (
            <line
              key={`baseline-${previous.sensor_id}-${node.sensor_id}`}
              x1={toSvgX(previous.position_m[0])}
              y1={toSvgY(previous.position_m[1])}
              x2={toSvgX(node.position_m[0])}
              y2={toSvgY(node.position_m[1])}
              className="network-baseline"
            />
          );
        })}

        {dipoles.filter(({ enabled }) => enabled).map((source) => (
          <g key={source.source_id} aria-label={`Dipole ${source.source_id}`}>
            <path
              d="M -9 0 L 0 -13 L 9 0 L 0 13 Z"
              transform={`translate(${toSvgX(source.initial_position_m[0])} ${toSvgY(source.initial_position_m[1])})`}
              className="network-dipole"
            />
            <text
              x={toSvgX(source.initial_position_m[0]) + 13}
              y={toSvgY(source.initial_position_m[1]) - 10}
              className="network-node-label"
            >
              {source.source_id}
            </text>
          </g>
        ))}

        {nodes.map((node) => {
          const selected = node.sensor_id === selectedNodeId;
          return (
            <g
              key={node.sensor_id}
              className={`network-node${selected ? " selected" : ""}${disabled ? " disabled" : ""}`}
              transform={`translate(${toSvgX(node.position_m[0])} ${toSvgY(node.position_m[1])})`}
              tabIndex={disabled ? -1 : 0}
              role="button"
              aria-label={`${node.sensor_id}, x ${node.position_m[0].toFixed(2)} metres, y ${node.position_m[1].toFixed(2)} metres`}
              aria-pressed={selected}
              onPointerDown={(event) => startDrag(event, node.sensor_id)}
              onKeyDown={(event) => moveWithKeyboard(event, node)}
              onClick={() => onSelect(node.sensor_id)}
            >
              <circle r={selected ? 14 : 11} />
              <line x1="-6" x2="6" y1="0" y2="0" />
              <line x1="0" x2="0" y1="-6" y2="6" />
              <text x="17" y="5" className="network-node-label">{node.sensor_id}</text>
            </g>
          );
        })}
      </svg>
      <div className="network-plane-legend" aria-label="Layout legend">
        <span><i className="legend-node" /> Sensor node</span>
        <span><i className="legend-dipole" /> Dipole source</span>
        <span><i className="legend-baseline" /> Declared baseline</span>
      </div>
    </figure>
  );
}
function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
