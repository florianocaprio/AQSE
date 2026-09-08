import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ObservationFrame } from "../types/network";

type NetworkComparisonChartProps = {
  frames: ObservationFrame[];
  sensorIds: string[];
};

const COLORS = ["#5eead4", "#60a5fa", "#fbbf24", "#c4b5fd", "#fb7185", "#a3e635", "#38bdf8", "#fdba74"];

export function NetworkComparisonChart({ frames, sensorIds }: NetworkComparisonChartProps) {
  const data = frames.slice(-400).map((frame) => {
    const row: Record<string, number | null> & { time_s: number } = { time_s: frame.sim_time_s };
    for (const sensorId of sensorIds) {
      const reading = frame.readings.find((candidate) => candidate.sensor_id === sensorId);
      if (!reading?.valid) row[sensorId] = null;
      else if (reading.components_T) row[sensorId] = Math.hypot(...reading.components_T) * 1e9;
      else row[sensorId] = reading.value_T === null || reading.value_T === undefined ? null : reading.value_T * 1e9;
    }
    return row;
  });

  return (
    <figure className="workbench-chart-card">
      <figcaption><span><strong>Network observation overlay</strong><small>Measured magnitude/scalar readout per node on common backend frame time.</small></span><span className="chart-unit">nT</span></figcaption>
      {data.length === 0 ? <div className="empty-chart">No network observations received.</div> : <div className="workbench-chart" role="img" aria-label="Observed network field overlay"><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 8 }}><CartesianGrid stroke="#20354a" strokeDasharray="3 4" /><XAxis dataKey="time_s" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(value: number) => value.toFixed(2)} stroke="#7890a8" /><YAxis stroke="#7890a8" width={66} /><Tooltip contentStyle={{ background: "#081523", border: "1px solid #35506d" }} /><Legend />{sensorIds.map((sensorId, index) => <Line key={sensorId} type="linear" dataKey={sensorId} stroke={COLORS[index % COLORS.length]} dot={false} isAnimationActive={false} connectNulls={false} />)}</LineChart></ResponsiveContainer></div>}
      <p className="chart-note">Overlay is descriptive only. Localization, coverage, correlation, and PSD are not inferred in the browser.</p>
    </figure>
  );
}
