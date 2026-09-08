import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { SensorAcquisition } from "../types";

const MAX_CHART_POINTS = 800;

type SignalChartProps = {
  acquisition: SensorAcquisition;
};

export function SignalChart({ acquisition }: SignalChartProps) {
  const wasDownsampled = acquisition.time.length > MAX_CHART_POINTS;
  const data = downsampleSignal(
    acquisition.time,
    acquisition.signal,
    MAX_CHART_POINTS,
  );
  const fieldValues = data.map(({ field }) => field);
  const fieldMinimum = Math.min(...fieldValues);
  const fieldMaximum = Math.max(...fieldValues);
  const fieldSpan = fieldMaximum - fieldMinimum;
  const fieldPadding =
    fieldSpan > 0
      ? fieldSpan * 0.08
      : Math.max(Math.abs(fieldMinimum) * 1.0e-6, 1.0);
  const fieldDomain: [number, number] = [
    fieldMinimum - fieldPadding,
    fieldMaximum + fieldPadding,
  ];

  return (
    <figure className="chart-card">
      <figcaption>
        <span>
          <strong>Magnetic field</strong>
          <small>Time-domain acquisition</small>
        </span>
        <span className="chart-unit">{acquisition.physical_unit}</span>
      </figcaption>
      <div
        className="chart-frame"
        role="img"
        aria-label={`Magnetic field in ${acquisition.physical_unit} plotted against time in ${acquisition.time_unit}`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 12, right: 16, bottom: 8, left: 4 }}
          >
            <CartesianGrid stroke="#20344b" strokeDasharray="3 5" />
            <XAxis
              dataKey="time"
              stroke="#8296ae"
              tick={{ fontSize: 11 }}
              tickFormatter={formatAxisNumber}
              label={{
                value: `Time (${acquisition.time_unit})`,
                position: "insideBottom",
                offset: -4,
                fill: "#8296ae",
                fontSize: 11,
              }}
            />
            <YAxis
              domain={fieldDomain}
              stroke="#8296ae"
              tick={{ fontSize: 11 }}
              tickFormatter={formatAxisNumber}
              width={62}
            />
            <Tooltip
              isAnimationActive={false}
              contentStyle={tooltipStyle}
              labelFormatter={(value) =>
                `Time: ${formatTooltipNumber(Number(value))} ${acquisition.time_unit}`
              }
              formatter={(value) => [
                `${formatTooltipNumber(Number(value))} ${acquisition.physical_unit}`,
                "Magnetic field",
              ]}
            />
            <Line
              type="linear"
              dataKey="field"
              stroke="#5eead4"
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {wasDownsampled && (
        <p className="chart-note">
          Display uses a min/max envelope of at most {MAX_CHART_POINTS} points;
          extracted features use the complete acquisition.
        </p>
      )}
    </figure>
  );
}

type SignalPoint = {
  time: number;
  field: number;
};

function downsampleSignal(
  time: number[],
  signal: number[],
  maximumPoints: number,
): SignalPoint[] {
  if (time.length <= maximumPoints) {
    return time.map((sampleTime, index) => ({
      time: sampleTime,
      field: signal[index],
    }));
  }

  const points: SignalPoint[] = [{ time: time[0], field: signal[0] }];
  const interiorCount = time.length - 2;
  const bucketCount = Math.max(1, Math.floor((maximumPoints - 2) / 2));

  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = 1 + Math.floor((bucket * interiorCount) / bucketCount);
    const end = 1 + Math.floor(((bucket + 1) * interiorCount) / bucketCount);
    let minimumIndex = start;
    let maximumIndex = start;

    for (let index = start + 1; index < end; index += 1) {
      if (signal[index] < signal[minimumIndex]) minimumIndex = index;
      if (signal[index] > signal[maximumIndex]) maximumIndex = index;
    }

    const orderedIndices =
      minimumIndex === maximumIndex
        ? [minimumIndex]
        : [minimumIndex, maximumIndex].sort((left, right) => left - right);
    for (const index of orderedIndices) {
      points.push({ time: time[index], field: signal[index] });
    }
  }

  const lastIndex = time.length - 1;
  points.push({ time: time[lastIndex], field: signal[lastIndex] });
  return points;
}

const tooltipStyle = {
  backgroundColor: "#0c1a2b",
  border: "1px solid #314963",
  borderRadius: "6px",
  color: "#e8f0fa",
  fontSize: "0.78rem",
};

function formatAxisNumber(value: number): string {
  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: 2,
  });
}

function formatTooltipNumber(value: number): string {
  return Number.isFinite(value) ? value.toPrecision(6) : "—";
}
