import {
  useCallback,
  useEffect,
  useId,
  useState,
  type FormEvent,
} from "react";

import type {
  FeatureVector,
  MagnetometerConfiguration,
  MagnetometerSimulationResponse,
} from "../types";
import { useWorkbench } from "../state/workbench";
import { SignalChart } from "./SignalChart";
import { SpectrumChart } from "./SpectrumChart";

const EXPECTED_FEATURE_NAMES = [
  "amplitude",
  "phase",
  "frequency",
  "variance",
  "drift",
  "snr",
  "spectral_peak",
  "temperature",
] as const;

const MIN_ACQUISITION_SAMPLES = 16;
const MAX_ACQUISITION_SAMPLES = 20_000;
const MAX_RANDOM_SEED = 2 ** 32 - 1;

type NumericConfigurationKey = Exclude<
  keyof MagnetometerConfiguration,
  "sensor_id" | "anomaly_enabled"
>;

type NumericControlDefinition = {
  key: NumericConfigurationKey;
  label: string;
  unit?: string;
  description: string;
  min?: number;
  max?: number;
  step: number;
};

const ACQUISITION_CONTROLS: NumericControlDefinition[] = [
  {
    key: "duration",
    label: "Duration",
    unit: "s",
    description: "Acquisition window",
    min: 0.01,
    step: 0.01,
  },
  {
    key: "sampling_rate",
    label: "Sampling rate",
    unit: "Hz",
    description: "Samples acquired per second",
    min: 1,
    step: 1,
  },
  {
    key: "random_seed",
    label: "Random seed",
    description: "Repeatable stochastic noise",
    min: 0,
    max: MAX_RANDOM_SEED,
    step: 1,
  },
  {
    key: "temperature",
    label: "Temperature",
    unit: "K",
    description: "Acquisition metadata",
    min: 0.01,
    step: 0.01,
  },
];

const SIGNAL_CONTROLS: NumericControlDefinition[] = [
  {
    key: "background_field",
    label: "Background field",
    unit: "nT",
    description: "Static magnetic-field offset",
    step: 0.1,
  },
  {
    key: "amplitude",
    label: "Signal amplitude",
    unit: "nT",
    description: "Sinusoidal component amplitude",
    min: 0,
    step: 0.1,
  },
  {
    key: "frequency",
    label: "Signal frequency",
    unit: "Hz",
    description: "Must remain below the Nyquist limit",
    min: 0.01,
    step: 0.01,
  },
  {
    key: "phase",
    label: "Signal phase",
    unit: "rad",
    description: "Initial sinusoidal phase",
    step: 0.01,
  },
  {
    key: "drift_rate",
    label: "Linear drift",
    unit: "nT/s",
    description: "Field change per second",
    step: 0.01,
  },
  {
    key: "noise_std",
    label: "Noise standard deviation",
    unit: "nT",
    description: "Gaussian stochastic noise level",
    min: 0,
    step: 0.01,
  },
];

const ANOMALY_CONTROLS: NumericControlDefinition[] = [
  {
    key: "anomaly_time",
    label: "Anomaly time",
    unit: "s",
    description: "Transient center within the acquisition",
    min: 0,
    step: 0.1,
  },
  {
    key: "anomaly_amplitude",
    label: "Anomaly amplitude",
    unit: "nT",
    description: "Signed transient field amplitude",
    step: 0.1,
  },
];

export function SensorSimulator() {
  const { dispatch } = useWorkbench();
  const [configuration, setConfiguration] =
    useState<MagnetometerConfiguration | null>(null);
  const [defaults, setDefaults] =
    useState<MagnetometerConfiguration | null>(null);
  const [simulation, setSimulation] =
    useState<MagnetometerSimulationResponse | null>(null);
  const [loadingDefaults, setLoadingDefaults] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [resultIsStale, setResultIsStale] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadDefaults = useCallback(async (signal?: AbortSignal) => {
    setLoadingDefaults(true);
    setError(null);

    try {
      const response = await fetch("/api/sensors/magnetometer/defaults", {
        signal,
      });
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }

      const payload = (await response.json()) as MagnetometerConfiguration;
      setDefaults(payload);
      setConfiguration(payload);
    } catch (requestError) {
      if (requestError instanceof DOMException && requestError.name === "AbortError") {
        return;
      }
      setError(toErrorMessage(requestError, "Unable to load sensor defaults."));
    } finally {
      if (!signal?.aborted) setLoadingDefaults(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadDefaults(controller.signal);
    return () => controller.abort();
  }, [loadDefaults]);

  const updateNumericValue = (
    key: NumericConfigurationKey,
    value: number | null,
  ) => {
    if (value === null && key !== "anomaly_time") return;
    if (value !== null && !Number.isFinite(value)) return;
    setConfiguration((current) =>
      current
        ? ({ ...current, [key]: value } as MagnetometerConfiguration)
        : current,
    );
    if (simulation) setResultIsStale(true);
  };

  const updateSensorId = (sensorId: string) => {
    setConfiguration((current) =>
      current ? { ...current, sensor_id: sensorId } : current,
    );
    if (simulation) setResultIsStale(true);
  };

  const updateAnomalyEnabled = (anomalyEnabled: boolean) => {
    setConfiguration((current) =>
      current ? { ...current, anomaly_enabled: anomalyEnabled } : current,
    );
    if (simulation) setResultIsStale(true);
  };

  const resetConfiguration = () => {
    if (!defaults) return;
    setConfiguration(defaults);
    setError(null);
    if (simulation) setResultIsStale(true);
  };

  const runSimulation = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!configuration) return;

    const validationError = validateConfiguration(configuration);
    if (validationError) {
      setError(validationError);
      return;
    }

    setSimulating(true);
    setError(null);
    const requestedConfiguration = { ...configuration };
    const startedAt = performance.now();

    try {
      const response = await fetch("/api/sensors/magnetometer/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(configuration),
      });
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }

      const payload = (await response.json()) as MagnetometerSimulationResponse;
      validateSimulationResponse(payload);
      setSimulation(payload);
      setResultIsStale(false);
      dispatch({
        type: "EXPERIMENT_ADD",
        experiment: {
          id: `experiment-${crypto.randomUUID()}`,
          created_at: new Date().toISOString(),
          kind: "scalar_simulation",
          status: "completed",
          title: `Scalar simulation · ${payload.acquisition.sensor_id}`,
          sensor_type: payload.acquisition.sensor_type,
          scenario: requestedConfiguration.anomaly_enabled
            ? "configured transient anomaly"
            : "baseline signal",
          seed: requestedConfiguration.random_seed,
          sample_count: payload.acquisition.sample_count,
          request_snapshot: {
            domain: "scalar_simulation",
            configuration: requestedConfiguration,
          },
          warnings: [],
          timings_ms: { request_total: performance.now() - startedAt },
        },
      });
    } catch (requestError) {
      setError(toErrorMessage(requestError, "Simulation request failed."));
    } finally {
      setSimulating(false);
    }
  };

  return (
    <section className="workspace-section" aria-labelledby="sensor-heading">
      <div className="section-heading">
        <div>
          <p className="section-index">01 / SENSOR LAYER</p>
          <h2 id="sensor-heading">Quantum Magnetometer</h2>
          <p>
            Configure and inspect a deterministic synthetic magnetic-field
            acquisition. Parameters are inputs; features are estimates from the
            generated signal.
          </p>
        </div>
        <div className="connection-status" aria-label="Sensor to quantum engine connection status">
          <span>Sensor → Quantum Engine</span>
          <strong>NOT CONNECTED</strong>
        </div>
      </div>

      {loadingDefaults && !configuration && (
        <div className="panel-state" role="status">
          Loading simulator defaults…
        </div>
      )}

      {!loadingDefaults && !configuration && (
        <div className="panel-state panel-state-error" role="alert">
          <p>{error ?? "Sensor configuration is unavailable."}</p>
          <button className="button button-secondary" onClick={() => void loadDefaults()}>
            Retry
          </button>
        </div>
      )}

      {configuration && (
        <form onSubmit={runSimulation} className="simulator-layout">
          <div className="control-panel">
            <div className="panel-title-row">
              <div>
                <p className="panel-kicker">INPUT</p>
                <h3>Simulation parameters</h3>
              </div>
              <button
                type="button"
                className="text-button"
                onClick={resetConfiguration}
                disabled={simulating}
              >
                Reset defaults
              </button>
            </div>

            <fieldset>
              <legend>Acquisition</legend>
              <div className="control-grid">
                <TextControl
                  label="Sensor ID"
                  description="Identifier stored with the acquisition"
                  value={configuration.sensor_id}
                  onChange={updateSensorId}
                  disabled={simulating}
                />
                {ACQUISITION_CONTROLS.map((control) => (
                  <NumericControl
                    key={control.key}
                    control={control}
                    value={configuration[control.key]}
                    onChange={(value) => updateNumericValue(control.key, value)}
                    disabled={simulating}
                  />
                ))}
              </div>
            </fieldset>

            <fieldset>
              <legend>Signal model</legend>
              <div className="control-grid">
                {SIGNAL_CONTROLS.map((control) => (
                  <NumericControl
                    key={control.key}
                    control={control}
                    value={configuration[control.key]}
                    onChange={(value) => updateNumericValue(control.key, value)}
                    disabled={simulating}
                  />
                ))}
              </div>
            </fieldset>

            <fieldset>
              <legend>Transient anomaly</legend>
              <label className="toggle-control">
                <input
                  type="checkbox"
                  checked={configuration.anomaly_enabled}
                  onChange={(event) => updateAnomalyEnabled(event.target.checked)}
                  disabled={simulating}
                />
                <span>
                  <strong>Enable anomaly</strong>
                  <small>Add an optional transient to the signal model</small>
                </span>
              </label>
              <div className="control-grid anomaly-controls">
                {ANOMALY_CONTROLS.map((control) => (
                  <NumericControl
                    key={control.key}
                    control={control}
                    value={configuration[control.key]}
                    onChange={(value) => updateNumericValue(control.key, value)}
                    disabled={simulating || !configuration.anomaly_enabled}
                    allowEmpty={control.key === "anomaly_time"}
                    max={
                      control.key === "anomaly_time"
                        ? configuration.duration
                        : control.max
                    }
                  />
                ))}
              </div>
            </fieldset>

            {error && (
              <div className="inline-message error-message" role="alert">
                {error}
              </div>
            )}

            <div className="form-actions">
              <p>
                No sensor output is forwarded to the quantum engine in
                Milestone 1B.
              </p>
              <button className="button button-primary" type="submit" disabled={simulating}>
                {simulating ? "SIMULATING…" : "SIMULATE"}
              </button>
            </div>
          </div>

          <SimulationOutput simulation={simulation} stale={resultIsStale} />
        </form>
      )}
    </section>
  );
}

type SimulationOutputProps = {
  simulation: MagnetometerSimulationResponse | null;
  stale: boolean;
};

function SimulationOutput({ simulation, stale }: SimulationOutputProps) {
  if (!simulation) {
    return (
      <div className="output-panel output-placeholder">
        <p className="panel-kicker">OUTPUT</p>
        <h3>Measured / extracted features</h3>
        <p>
          Run the simulator to generate the time series, frequency spectrum,
          and eight-dimensional feature vector.
        </p>
      </div>
    );
  }

  const { acquisition, spectrum, features } = simulation;

  return (
    <div className="output-panel">
      <div className="panel-title-row output-title-row">
        <div>
          <p className="panel-kicker">OUTPUT</p>
          <h3>Measured / extracted features</h3>
        </div>
        <span className={stale ? "result-badge stale" : "result-badge"}>
          {stale ? "PARAMETERS CHANGED" : "CURRENT RUN"}
        </span>
      </div>

      <dl className="acquisition-meta">
        <div>
          <dt>Sensor</dt>
          <dd>{acquisition.sensor_id}</dd>
        </div>
        <div>
          <dt>Samples</dt>
          <dd>{acquisition.sample_count.toLocaleString("en-US")}</dd>
        </div>
        <div>
          <dt>Sampling rate</dt>
          <dd>{formatValue(acquisition.sampling_rate)} Hz</dd>
        </div>
        <div>
          <dt>Acquired</dt>
          <dd>{formatTimestamp(acquisition.timestamp)}</dd>
        </div>
      </dl>

      {stale && (
        <p className="inline-message warning-message" role="status">
          These results represent the previous run. Select SIMULATE to apply
          the edited parameters.
        </p>
      )}

      <div className="chart-grid">
        <SignalChart acquisition={acquisition} />
        <SpectrumChart spectrum={spectrum} />
      </div>

      <div className="feature-heading">
        <h4>8D feature vector</h4>
        <span>Unscaled values</span>
      </div>
      <FeatureCards features={features} />
    </div>
  );
}

function FeatureCards({ features }: { features: FeatureVector }) {
  return (
    <ol className="feature-grid" aria-label="Extracted magnetometer feature vector">
      {EXPECTED_FEATURE_NAMES.map((expectedName, index) => {
        const name = features.names[index] ?? expectedName;
        const value = features.values[index];
        const unit = features.units[index] ?? "";

        return (
          <li className="feature-card" key={expectedName}>
            <div>
              <span className="feature-index">f{index}</span>
              <span className="feature-name">{humanizeFeatureName(name)}</span>
            </div>
            <strong>{formatValue(value)}</strong>
            <span className="feature-unit">{unit || "dimensionless"}</span>
          </li>
        );
      })}
    </ol>
  );
}

type NumericControlProps = {
  control: NumericControlDefinition;
  value: number | null;
  onChange: (value: number | null) => void;
  disabled: boolean;
  max?: number;
  allowEmpty?: boolean;
};

function NumericControl({
  control,
  value,
  onChange,
  disabled,
  max,
  allowEmpty = false,
}: NumericControlProps) {
  const inputId = useId();
  const descriptionId = `${inputId}-description`;

  return (
    <label className="input-control" htmlFor={inputId}>
      <span className="input-label">
        {control.label}
        {control.unit && <small>{control.unit}</small>}
      </span>
      <input
        id={inputId}
        type="number"
        value={value ?? ""}
        min={control.min}
        max={max ?? control.max}
        step={control.step}
        onChange={(event) => {
          if (allowEmpty && event.target.value === "") {
            onChange(null);
            return;
          }
          onChange(event.target.valueAsNumber);
        }}
        aria-describedby={descriptionId}
        disabled={disabled}
        required
      />
      <small id={descriptionId}>{control.description}</small>
    </label>
  );
}

type TextControlProps = {
  label: string;
  description: string;
  value: string;
  onChange: (value: string) => void;
  disabled: boolean;
};

function TextControl({
  label,
  description,
  value,
  onChange,
  disabled,
}: TextControlProps) {
  const inputId = useId();
  const descriptionId = `${inputId}-description`;

  return (
    <label className="input-control" htmlFor={inputId}>
      <span className="input-label">{label}</span>
      <input
        id={inputId}
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-describedby={descriptionId}
        disabled={disabled}
        required
      />
      <small id={descriptionId}>{description}</small>
    </label>
  );
}

function validateConfiguration(
  configuration: MagnetometerConfiguration,
): string | null {
  if (!configuration.sensor_id.trim()) return "Sensor ID is required.";
  if (configuration.duration <= 0) return "Duration must be greater than zero.";
  if (configuration.sampling_rate <= 0) {
    return "Sampling rate must be greater than zero.";
  }
  const sampleCount = Math.round(
    configuration.duration * configuration.sampling_rate,
  );
  if (sampleCount < MIN_ACQUISITION_SAMPLES) {
    return `Configuration must produce at least ${MIN_ACQUISITION_SAMPLES} samples.`;
  }
  if (sampleCount > MAX_ACQUISITION_SAMPLES) {
    return `Configuration may produce at most ${MAX_ACQUISITION_SAMPLES.toLocaleString("en-US")} samples.`;
  }
  if (configuration.frequency <= 0) {
    return "Signal frequency must be greater than zero.";
  }
  if (configuration.frequency >= configuration.sampling_rate / 2) {
    return "Signal frequency must be below the Nyquist limit (sampling rate / 2).";
  }
  if (configuration.amplitude < 0 || configuration.noise_std < 0) {
    return "Signal amplitude and noise must not be negative.";
  }
  if (configuration.temperature <= 0) {
    return "Temperature must be greater than zero kelvin.";
  }
  if (
    !Number.isInteger(configuration.random_seed) ||
    configuration.random_seed < 0 ||
    configuration.random_seed > MAX_RANDOM_SEED
  ) {
    return `Random seed must be an integer between 0 and ${MAX_RANDOM_SEED}.`;
  }
  if (
    configuration.anomaly_enabled &&
    (configuration.anomaly_time === null ||
      configuration.anomaly_time < 0 ||
      configuration.anomaly_time >= configuration.duration)
  ) {
    return "Anomaly time must fall within the acquisition duration.";
  }
  return null;
}

function validateSimulationResponse(
  response: MagnetometerSimulationResponse,
): void {
  if (
    !response.acquisition ||
    response.acquisition.time.length !== response.acquisition.signal.length
  ) {
    throw new Error("Backend returned an invalid acquisition.");
  }
  if (
    !response.spectrum ||
    response.spectrum.frequencies.length !==
      response.spectrum.power_spectral_density.length
  ) {
    throw new Error("Backend returned an invalid frequency spectrum.");
  }
  if (
    !response.features ||
    response.features.names.length !== 8 ||
    response.features.values.length !== 8 ||
    response.features.units.length !== 8 ||
    !EXPECTED_FEATURE_NAMES.every(
      (expectedName, index) => response.features.names[index] === expectedName,
    ) ||
    !response.features.values.every(Number.isFinite)
  ) {
    throw new Error("Backend returned an invalid 8D feature vector.");
  }
}

async function readApiError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as {
      detail?: string | Array<{ msg?: string }>;
    };
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
      const messages = payload.detail
        .map((item) => item.msg)
        .filter((message): message is string => Boolean(message));
      if (messages.length > 0) return messages.join(" ");
    }
  } catch {
    // Fall through to the HTTP status when the response is not JSON.
  }
  return `Request failed with HTTP ${response.status}.`;
}

function toErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function humanizeFeatureName(name: string): string {
  return name
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatValue(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  if (absolute !== 0 && (absolute >= 100_000 || absolute < 0.001)) {
    return value.toExponential(4);
  }
  return value.toLocaleString("en-US", { maximumSignificantDigits: 7 });
}

function formatTimestamp(timestamp: string): string {
  const parsed = new Date(timestamp);
  return Number.isNaN(parsed.getTime())
    ? timestamp
    : parsed.toLocaleString("en-US", {
        dateStyle: "medium",
        timeStyle: "medium",
      });
}
