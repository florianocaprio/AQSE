type KernelHeatmapProps = {
  matrix: number[][];
  label: string;
};

export function KernelHeatmap({ matrix, label }: KernelHeatmapProps) {
  const size = matrix.length;
  const width = 520;
  const plot = 420;
  const cell = size > 0 ? plot / size : plot;

  return (
    <figure className="technical-diagram-card kernel-card">
      <figcaption>
        <span>
          <strong>{label}</strong>
          <small>{size} × {size} values returned by the preview endpoint.</small>
        </span>
        <span className="chart-unit">similarity</span>
      </figcaption>
      {size === 0 ? (
        <div className="empty-chart">No kernel has been executed.</div>
      ) : (
        <svg viewBox={`0 0 ${width} ${width}`} role="img" aria-label={`${label}, ${size} by ${size}`}>
          <g transform="translate(58 30)">
            {matrix.flatMap((row, rowIndex) =>
              row.map((value, columnIndex) => (
                <rect
                  key={`${rowIndex}-${columnIndex}`}
                  x={columnIndex * cell}
                  y={rowIndex * cell}
                  width={cell + 0.2}
                  height={cell + 0.2}
                  fill={kernelColor(value)}
                >
                  <title>K[{rowIndex}, {columnIndex}] = {value.toPrecision(5)}</title>
                </rect>
              )),
            )}
            <rect width={plot} height={plot} className="kernel-outline" />
            <text x={plot / 2} y={plot + 32} textAnchor="middle" className="kernel-axis-label">reference window</text>
            <text transform={`translate(-38 ${plot / 2}) rotate(-90)`} textAnchor="middle" className="kernel-axis-label">reference window</text>
          </g>
        </svg>
      )}
    </figure>
  );
}
function kernelColor(value: number): string {
  const normalized = Math.max(0, Math.min(1, value));
  const lightness = 14 + normalized * 58;
  return `hsl(${190 - normalized * 25} 72% ${lightness}%)`;
}
