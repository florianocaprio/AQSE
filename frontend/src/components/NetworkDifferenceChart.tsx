import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  differenceCompatibility,
  observedDifferenceTesla,
} from "../state/selectors";
import type { ObservationFrame, SensorNodeConfiguration } from "../types/network";

type NetworkDifferenceChartProps = {
  frames: ObservationFrame[];
  sensorId: string;
  referenceSensorId: string;
  baselineM: number | null;
  sensorNode: SensorNodeConfiguration | undefined;
  referenceNode: SensorNodeConfiguration | undefined;
};

export function NetworkDifferenceChart({ frames, sensorId, referenceSensorId, baselineM, sensorNode, referenceNode }: NetworkDifferenceChartProps) {
  const compatibility = differenceCompatibility(sensorNode, referenceNode);
  const data = frames.slice(-400).map((frame) => {
    const selected = frame.readings.find((reading) => reading.sensor_id === sensorId);
    const reference = frame.readings.find((reading) => reading.sensor_id === referenceSensorId);
    const difference = observedDifferenceTesla(selected, reference, compatibility);
    return {
      time_s: frame.sim_time_s,
      difference_nt: difference === null ? null : difference * 1e9,
    };
  });

  return (
    <figure className="workbench-chart-card">
      <figcaption><span><strong>Observed difference · {sensorId} − {referenceSensorId}</strong><small>{compatibility.compatible ? `${compatibility.quantity}; invalid, stuck, or clock-errored readings remain gaps.` : "A pointwise difference requires compatible declared readouts."}</small></span><span className="chart-unit">{compatibility.compatible ? "nT" : "UNAVAILABLE"}</span></figcaption>
      {!compatibility.compatible
        ? <div className="empty-chart" role="status">Difference unavailable: {compatibility.reason}</div>
        : data.length === 0
          ? <div className="empty-chart">No synchronized observations received.</div>
          : <div className="workbench-chart" role="img" aria-label={`Observed difference between ${sensorId} and ${referenceSensorId}`}><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 8 }}><CartesianGrid stroke="#20354a" strokeDasharray="3 4" /><XAxis dataKey="time_s" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(value: number) => value.toFixed(2)} stroke="#7890a8" /><YAxis stroke="#7890a8" width={66} /><Tooltip contentStyle={{ background: "#081523", border: "1px solid #35506d" }} /><Line type="linear" dataKey="difference_nt" name="ΔB" stroke="#fbbf24" dot={false} isAnimationActive={false} connectNulls={false} /></LineChart></ResponsiveContainer></div>}
      <p className="chart-note">Baseline: {baselineM === null ? "not available" : `${baselineM.toFixed(4)} m`}. ΔB is never converted into a gradient or localization estimate.</p>
    </figure>
  );
}
