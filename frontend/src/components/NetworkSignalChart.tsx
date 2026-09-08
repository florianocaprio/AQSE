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
import { chartPointsForSensor } from "../state/selectors";

type NetworkSignalChartProps = {
  frames: ObservationFrame[];
  sensorId: string;
};

export function NetworkSignalChart({ frames, sensorId }: NetworkSignalChartProps) {
  const data = chartPointsForSensor(frames, sensorId);
  const hasVector = data.some((point) => point.x_nt !== null);
  const hasScalar = data.some((point) => point.scalar_nt !== null);

  return (
    <figure className="workbench-chart-card">
      <figcaption>
        <span>
          <strong>Observed field · {sensorId}</strong>
          <small>Missing samples remain explicit gaps; quality flags are preserved separately.</small>
        </span>
        <span className="chart-unit">nT</span>
      </figcaption>
      {data.length === 0 ? (
        <div className="empty-chart">Start or step a network session to collect observations.</div>
      ) : (
        <div className="workbench-chart" role="img" aria-label={`Observed magnetic field for ${sensorId}`}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 8 }}>
              <CartesianGrid stroke="#20354a" strokeDasharray="3 4" />
              <XAxis
                dataKey="time_s"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(value: number) => value.toFixed(2)}
                stroke="#7890a8"
                label={{ value: "simulation time (s)", position: "insideBottom", offset: -4, fill: "#7890a8" }}
              />
              <YAxis stroke="#7890a8" width={66} />
              <Tooltip
                contentStyle={{ background: "#081523", border: "1px solid #35506d" }}
                labelFormatter={(value) => `t = ${Number(value).toFixed(4)} s`}
              />
              <Legend verticalAlign="top" height={30} />
              {hasVector && (
                <>
                  <Line type="linear" dataKey="x_nt" name="Bx" stroke="#5eead4" dot={false} isAnimationActive={false} connectNulls={false} />
                  <Line type="linear" dataKey="y_nt" name="By" stroke="#60a5fa" dot={false} isAnimationActive={false} connectNulls={false} />
                  <Line type="linear" dataKey="z_nt" name="Bz" stroke="#fbbf24" dot={false} isAnimationActive={false} connectNulls={false} />
                </>
              )}
              {hasScalar && (
                <Line type="linear" dataKey="scalar_nt" name="B" stroke="#c4b5fd" dot={false} isAnimationActive={false} connectNulls={false} />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </figure>
  );
}
