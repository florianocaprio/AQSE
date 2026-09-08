import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FrequencySpectrum } from "../types";

const MAX_CHART_POINTS = 800;

type SpectrumChartProps = {
  spectrum: FrequencySpectrum;
};

export function SpectrumChart({ spectrum }: SpectrumChartProps) {
  const wasDownsampled = spectrum.frequencies.length > MAX_CHART_POINTS;
  const data = downsampleSpectrum(
    spectrum.frequencies,
    spectrum.power_spectral_density,
    MAX_CHART_POINTS,
  );

  return (
    <figure className="chart-card">
      <figcaption>
        <span>
          <strong>Frequency spectrum</strong>
          <small>Power spectral density</small>
        </span>
        <span className="chart-unit">{spectrum.power_unit}</span>
      </figcaption>
      <div
        className="chart-frame"
        role="img"
        aria-label={`Power spectral density in ${spectrum.power_unit} plotted against frequency in ${spectrum.frequency_unit}`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 12, right: 16, bottom: 8, left: 4 }}
          >
            <CartesianGrid stroke="#20344b" strokeDasharray="3 5" />
            <XAxis
              dataKey="frequency"
              stroke="#8296ae"
              tick={{ fontSize: 11 }}
              tickFormatter={formatAxisNumber}
              label={{
                value: `Frequency (${spectrum.frequency_unit})`,
                position: "insideBottom",
                offset: -4,
                fill: "#8296ae",
                fontSize: 11,
              }}
            />
            <YAxis
              stroke="#8296ae"
              tick={{ fontSize: 11 }}
              tickFormatter={formatScientific}
              width={62}
            />
            <Tooltip
              isAnimationActive={false}
              contentStyle={tooltipStyle}
              labelFormatter={(value) =>
                `Frequency: ${formatTooltipNumber(Number(value))} ${spectrum.frequency_unit}`
              }
              formatter={(value) => [
                `${formatTooltipNumber(Number(value))} ${spectrum.power_unit}`,
                "PSD",
              ]}
            />
            <Line
              type="linear"
              dataKey="power"
              stroke="#60a5fa"
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {wasDownsampled && (
        <p className="chart-note">
          Display uses peak-preserving frequency buckets of at most{" "}
          {MAX_CHART_POINTS} points; extracted features use the complete spectrum.
        </p>
      )}
    </figure>
  );
}

type SpectrumPoint = {
  frequency: number;
  power: number;
};

function downsampleSpectrum(
  frequencies: number[],
  power: number[],
  maximumPoints: number,
): SpectrumPoint[] {
  if (frequencies.length <= maximumPoints) {
    return frequencies.map((frequency, index) => ({
      frequency,
      power: power[index],
    }));
  }

  const points: SpectrumPoint[] = [
    { frequency: frequencies[0], power: power[0] },
  ];
  const interiorCount = frequencies.length - 2;
  const bucketCount = Math.max(1, maximumPoints - 2);

  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = 1 + Math.floor((bucket * interiorCount) / bucketCount);
    const end = 1 + Math.floor(((bucket + 1) * interiorCount) / bucketCount);
    let peakIndex = start;

    for (let index = start + 1; index < end; index += 1) {
      if (power[index] > power[peakIndex]) peakIndex = index;
    }

    points.push({
      frequency: frequencies[peakIndex],
      power: power[peakIndex],
    });
  }

  const lastIndex = frequencies.length - 1;
  points.push({
    frequency: frequencies[lastIndex],
    power: power[lastIndex],
  });
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

function formatScientific(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return value.toExponential(1);
}

function formatTooltipNumber(value: number): string {
  return Number.isFinite(value) ? value.toPrecision(6) : "—";
}
