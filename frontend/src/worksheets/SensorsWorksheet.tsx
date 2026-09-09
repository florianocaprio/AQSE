import { useEffect, useRef, useState, type ChangeEvent } from "react";

import { errorMessage } from "../api/client";
import { scheduleNetworkEvent } from "../api/network";
import { NetworkSignalChart } from "../components/NetworkSignalChart";
import { NetworkComparisonChart } from "../components/NetworkComparisonChart";
import { NetworkDifferenceChart } from "../components/NetworkDifferenceChart";
import { SensorSimulator } from "../components/SensorSimulator";
import { WorksheetHeader } from "../components/WorksheetHeader";
import { NetworkPlane } from "../diagrams/NetworkPlane";
import { NetworkIsometric } from "../diagrams/NetworkIsometric";
import { readingForSensor } from "../state/selectors";
import {
  networkResultIsStale,
  useWorkbench,
} from "../state/workbench";
import { useWorkbenchActions } from "../state/useWorkbenchController";
import type {
  EnvironmentConfiguration,
  EventKind,
  NetworkEventConfiguration,
  NetworkSessionConfiguration,
  NodeErrorConfiguration,
  ScheduledEventResponse,
  SensorNodeConfiguration,
  SessionStatus,
  TemperatureDriverConfiguration,
} from "../types/network";
import type { Matrix3 } from "../types/network";
import type { Vector3 } from "../types/workbench";

const EVENT_KINDS: EventKind[] = [
  "world_field_offset",
  "node_bias",
  "node_drift",
  "node_noise_burst",
  "shared_instrument_offset",
  "dropout",
  "stuck",
];

export type LiveEventIntent =
  | "environment_common"
  | "node_bias"
  | "node_drift"
  | "node_noise"
  | "shared_instrument"
  | "dropout"
  | "clipping"
  | "stuck";

const LIVE_EVENT_OPTIONS: ReadonlyArray<{
  intent: LiveEventIntent;
  label: string;
  description: string;
  defaultMagnitude: number;
}> = [
  {
    intent: "environment_common",
    label: "Common environmental field step",
    description: "World-frame field perturbation applied to the environment, not to an individual device.",
    defaultMagnitude: 100,
  },
  {
    intent: "node_bias",
    label: "Single-node bias deviation",
    description: "Device-local X-axis offset for the selected sensor.",
    defaultMagnitude: 100,
  },
  {
    intent: "node_drift",
    label: "Single-node drift",
    description: "Device-local X-axis drift for the selected sensor.",
    defaultMagnitude: 5,
  },
  {
    intent: "node_noise",
    label: "Single-node noise burst",
    description: "Temporary multiplication of the selected sensor's configured white noise.",
    defaultMagnitude: 8,
  },
  {
    intent: "shared_instrument",
    label: "Shared instrument offset",
    description: "Identical device-local X-axis offset applied to every node; requires at least two sensors.",
    defaultMagnitude: 75,
  },
  {
    intent: "dropout",
    label: "Sensor dropout",
    description: "Explicit temporary loss of signal for the selected sensor.",
    defaultMagnitude: 0,
  },
  {
    intent: "clipping",
    label: "Clipping stress",
    description: "Bias stress derived from the selected node's lowest saturation limit; verify clipping in observed quality flags.",
    defaultMagnitude: 0,
  },
  {
    intent: "stuck",
    label: "Stuck reading",
    description: "Explicit temporary held reading for the selected sensor.",
    defaultMagnitude: 0,
  },
];

const DEFAULT_TEMPERATURE_DRIVER: TemperatureDriverConfiguration = {
  kind: "constant",
  ramp_rate_K_per_s: 0,
  ramp_duration_s: 1,
  sinusoidal_amplitude_K: 0,
  sinusoidal_frequency_Hz: 0.1,
  sinusoidal_phase_rad: 0,
};

export function SensorsWorksheet() {
  const { state, dispatch } = useWorkbench();
  const actions = useWorkbenchActions();
  const [selectedPreset, setSelectedPreset] = useState("localization_2d");
  const [referenceNodeId, setReferenceNodeId] = useState<string | null>(null);
  const draft = state.network.draft?.value ?? null;
  const observationConfiguration = state.network.executed?.value ?? draft;
  const selectedNode = draft?.nodes.find(({ sensor_id }) => sensor_id === state.network.selected_node_id) ?? null;
  const latestFrame = state.network.observations.at(-1);
  const latestReading = latestFrame && state.network.selected_node_id
    ? readingForSensor(latestFrame, state.network.selected_node_id)
    : null;
  const effectiveReferenceId = referenceNodeId && observationConfiguration?.nodes.some(({ sensor_id }) => sensor_id === referenceNodeId)
    ? referenceNodeId
    : observationConfiguration?.nodes.find(({ sensor_id }) => sensor_id !== state.network.selected_node_id)?.sensor_id ?? null;
  const selectedPosition = observationConfiguration?.nodes.find(({ sensor_id }) => sensor_id === state.network.selected_node_id)?.position_m;
  const referencePosition = observationConfiguration?.nodes.find(({ sensor_id }) => sensor_id === effectiveReferenceId)?.position_m;
  const baselineM = selectedPosition && referencePosition
    ? Math.hypot(...selectedPosition.map((value, index) => value - referencePosition[index]))
    : null;
  const deviceState = !latestReading
    ? "NO OBSERVATION"
    : latestReading.valid
      ? latestReading.quality_flags.length > 0 ? "DEGRADED" : "VALID"
      : "SIMULATED FAULT";

  const patchDraft = (next: NetworkSessionConfiguration) => {
    dispatch({ type: "NETWORK_DRAFT", configuration: next });
  };
  const updateNode = (sensorId: string, updater: (node: SensorNodeConfiguration) => SensorNodeConfiguration) => {
    if (!draft) return;
    patchDraft({ ...draft, nodes: draft.nodes.map((node) => node.sensor_id === sensorId ? updater(node) : node) });
  };
  const updateSelectedNode = (updater: (node: SensorNodeConfiguration) => SensorNodeConfiguration) => {
    if (selectedNode) updateNode(selectedNode.sensor_id, updater);
  };
  const applyPreset = async () => {
    const preset = state.network.presets.find(({ preset_id }) => preset_id === selectedPreset);
    if (preset) await actions.loadNetworkPreset(preset.preset_id);
  };

  return (
    <section className="worksheet" aria-labelledby="sensors-title">
      <WorksheetHeader
        titleId="sensors-title"
        index="02"
        eyebrow="CONTINUOUS SENSOR LAB"
        title="Sensors"
        description="Configure one to eight synthetic magnetometers, execute backend-owned simulation time, and inspect observed data separately from simulator truth."
        actions={
          <label className="blind-mode-control">
            <input
              type="checkbox"
              checked={state.network.blind_mode}
              onChange={(event) => void actions.setBlindMode(event.target.checked)}
            />
            <span><strong>Blind mode</strong><small>Hide realized truth and latent-cause details</small></span>
          </label>
        }
      />

      <div className="health-strip" aria-live="polite">
        <HealthDatum label="Backend" value={state.backend_health.status} tone={state.backend_health.status} />
        <HealthDatum label="Network API" value={state.network.health.status} tone={state.network.health.status} />
        <HealthDatum label="Simulation" value={state.network.session?.state ?? "not created"} tone={state.network.session?.state ?? "idle"} />
        <HealthDatum label="SSE stream" value={state.network.stream_status} tone={state.network.stream_status} />
        <HealthDatum label="Selected device" value={deviceState} tone={latestReading?.valid ? "ready" : latestReading ? "fault" : "idle"} />
      </div>

      {state.network.error && <p className="inline-message error-message" role="alert">Application request error: {state.network.error}</p>}
      {state.network.gap_detected && <p className="inline-message warning-message">The bounded server buffer reported a frame gap. Downstream extraction will use only a later contiguous segment.</p>}
      {state.network.blind_mode && <p className="inline-message blind-boundary-message">Blind mode hides realized truth, device-error inputs, latent-source editors, source markers, and the scheduled-cause timeline. Remaining controls are observable or operational inputs. This is a local UI barrier, not an API security control.</p>}

      <div className="provider-strip" aria-label="Magnetic field providers">
        {state.network.field_providers.length > 0 ? state.network.field_providers.map((provider) => (
          <div key={provider.provider_id}>
            <span>{provider.display_name}</span>
            <strong className={provider.configured ? "health-tone ready" : "health-tone paused"}>{provider.configured ? "CONFIGURED" : "NOT CONFIGURED"}</strong>
            <small>{provider.description}</small>
          </div>
        )) : <p className="empty-copy">Field-provider registry unavailable; no fallback provider is assumed.</p>}
      </div>

      <div className="worksheet-grid sensor-layout-grid">
        <aside className="workbench-panel configuration-panel">
          <div className="panel-heading-row">
            <div><p className="panel-kicker">DRAFT CONFIGURATION</p><h2>Network controls</h2></div>
            <span className={networkResultIsStale(state) ? "result-badge stale" : "result-badge"}>
              {networkResultIsStale(state) ? "DRAFT CHANGED" : `REV ${state.network.draft?.revision ?? 0}`}
            </span>
          </div>

          <fieldset>
            <legend>Scenario geometry</legend>
            {!state.network.blind_mode && <><div className="inline-controls">
              <label className="input-control grow"><span className="input-label">Preset</span>
                <select value={selectedPreset} onChange={(event) => setSelectedPreset(event.target.value)}>
                  {state.network.presets.map((preset) => <option value={preset.preset_id} key={preset.preset_id}>{preset.display_name} · {preset.recommended_node_count} nodes</option>)}
                </select>
              </label>
              <button type="button" className="button button-secondary" onClick={() => void applyPreset()} disabled={state.network.presets.length === 0}>Load</button>
            </div>
            {state.network.presets.find(({ preset_id }) => preset_id === selectedPreset)?.description && (
              <p className="field-help">{state.network.presets.find(({ preset_id }) => preset_id === selectedPreset)?.description}</p>
            )}</>}
            {state.network.blind_mode && <p className="field-help">Preset names and descriptions are hidden because they may disclose a latent scenario. Node-count defaults remain available.</p>}
            <div className="node-count-buttons" aria-label="Load defaults by node count">
              {Array.from({ length: 8 }, (_, index) => index + 1).map((count) => (
                <button
                  type="button"
                  key={count}
                  className={draft?.nodes.length === count ? "node-count active" : "node-count"}
                  onClick={() => void actions.loadNetworkDefaults(count)}
                  aria-label={`Load defaults for ${count} sensor ${count === 1 ? "node" : "nodes"}`}
                >{count}</button>
              ))}
            </div>
          </fieldset>

          {draft ? (
            <>
              <GlobalConfiguration configuration={draft} onChange={patchDraft} />
              {state.network.blind_mode
                ? <p className="blind-configuration-boundary">Environment, device-error, scheduled-event, and injected-cause controls are hidden while blind mode is active. Disable blind mode to edit them.</p>
                : <><EnvironmentControls configuration={draft} onChange={patchDraft} /><EventControls configuration={draft} selectedNodeId={selectedNode?.sensor_id ?? null} onChange={patchDraft} /><LiveEventControls session={state.network.session} configuration={state.network.executed?.value ?? null} selectedNodeId={state.network.selected_node_id} /></>}
            </>
          ) : <p className="empty-copy">Waiting for backend configuration defaults.</p>}
        </aside>

        <div className="sensor-work-area">
          {draft && (
            <div className="geometry-grid">
              <NetworkPlane
                nodes={draft.nodes}
                dipoles={state.network.blind_mode ? [] : draft.environment.dipoles}
                selectedNodeId={state.network.selected_node_id}
                onSelect={(sensor_id) => dispatch({ type: "NETWORK_NODE_SELECTED", sensor_id })}
                onMove={(sensorId, x, y) => updateNode(sensorId, (node) => ({ ...node, position_m: [round(x), round(y), node.position_m[2]] }))}
              />
              <NetworkIsometric nodes={draft.nodes} dipoles={state.network.blind_mode ? [] : draft.environment.dipoles} selectedNodeId={state.network.selected_node_id} />
            </div>
          )}

          <div className="session-toolbar" aria-label="Network session controls">
            <button type="button" className="button button-primary" onClick={() => void actions.createSession()} disabled={!draft || Boolean(state.network.session) || state.network.request_status === "loading"}>Create session</button>
            <button type="button" className="button button-secondary" onClick={() => void actions.controlSession("start")} disabled={state.network.session?.state !== "created"}>Start</button>
            <button type="button" className="button button-secondary" onClick={() => void actions.controlSession("pause")} disabled={state.network.session?.state !== "running"}>Pause</button>
            <button type="button" className="button button-secondary" onClick={() => void actions.controlSession("resume")} disabled={state.network.session?.state !== "paused"}>Resume</button>
            <button type="button" className="button button-secondary" onClick={() => void actions.stepSession()} disabled={!state.network.session || state.network.session.state === "running" || state.network.session.state === "stopped"}>Step</button>
            <button type="button" className="button button-secondary" onClick={() => void actions.controlSession("stop")} disabled={!state.network.session || !["running", "paused"].includes(state.network.session.state)}>Stop</button>
            <button type="button" className="button button-secondary" onClick={() => void actions.controlSession("reset")} disabled={!state.network.session || state.network.session.state === "running"}>Reset</button>
            <button type="button" className="button button-secondary" onClick={() => void actions.controlSession("replay")} disabled={state.network.session?.state !== "stopped"}>Re-run same realization</button>
            <button type="button" className="button button-secondary" onClick={() => draft && patchDraft({ ...draft, random_seed: nextRandomSeed(draft.random_seed) })} disabled={!draft}>Generate new realization</button>
            <button type="button" className="button button-secondary" onClick={() => void actions.discardSession()} disabled={!state.network.session || ["running", "paused"].includes(state.network.session.state)}>Delete session</button>
          </div>
          <p className="session-seed-help">Re-run uses the executed configuration and seed. Generate new realization changes only the visible draft seed; it never starts or replaces a session. Draft seed: {draft?.random_seed ?? "—"} · executed seed: {state.network.executed?.value.random_seed ?? "—"}.</p>

          <dl className="session-metadata">
            <div><dt>Session</dt><dd>{state.network.session?.session_id ?? "—"}</dd></div>
            <div><dt>Backend time</dt><dd>{state.network.session ? `${state.network.session.sim_time_s.toFixed(4)} s` : "—"}</dd></div>
            <div><dt>Latest frame</dt><dd>{state.network.session?.latest_frame_id ?? "—"}</dd></div>
            <div><dt>Buffer</dt><dd>{state.network.session ? `${state.network.session.buffer.size} / ${state.network.session.buffer.capacity}` : "—"}</dd></div>
            <div><dt>Configuration lineage</dt><dd>{latestFrame ? `frame v${latestFrame.configuration_version}` : state.network.session ? `session v${state.network.session.configuration_version}` : "—"}</dd></div>
            <div><dt>Data revision</dt><dd>{state.network.data_revision}</dd></div>
          </dl>

          {selectedNode && <NodeControls node={selectedNode} samplingRateHz={draft?.sampling_rate_Hz ?? 1} blindMode={state.network.blind_mode} onChange={updateSelectedNode} />}
          {state.network.selected_node_id && <NetworkSignalChart frames={state.network.observations} sensorId={state.network.selected_node_id} />}
          {observationConfiguration && <NetworkComparisonChart frames={state.network.observations} sensorIds={observationConfiguration.nodes.map(({ sensor_id }) => sensor_id)} />}
          {observationConfiguration && observationConfiguration.nodes.length > 1 && state.network.selected_node_id && effectiveReferenceId && (
            <>
              <article className="workbench-panel compact-panel analysis-controls"><div><p className="panel-kicker">DESCRIPTIVE ANALYSIS</p><h2>Reference selection</h2></div><label className="input-control"><span className="input-label">Reference node</span><select value={effectiveReferenceId} onChange={(event) => setReferenceNodeId(event.target.value)}>{observationConfiguration.nodes.filter(({ sensor_id }) => sensor_id !== state.network.selected_node_id).map((node) => <option key={node.sensor_id} value={node.sensor_id}>{node.sensor_id}</option>)}</select></label><p className="field-help">Correlation, continuous-network PSD, coverage, localization and tracking remain unavailable until explicit analysis contracts are implemented.</p></article>
              <NetworkDifferenceChart frames={state.network.observations} sensorId={state.network.selected_node_id} referenceSensorId={effectiveReferenceId} baselineM={baselineM} sensorNode={observationConfiguration.nodes.find(({ sensor_id }) => sensor_id === state.network.selected_node_id)} referenceNode={observationConfiguration.nodes.find(({ sensor_id }) => sensor_id === effectiveReferenceId)} />
            </>
          )}
          <ObservationQuality reading={latestReading} />
          <TruthPanel />
        </div>
      </div>

      <details className="legacy-workspace">
        <summary>Milestone 1B scalar magnetometer simulator</summary>
        <p>The finite scalar simulator remains available as an independent compatibility tool. It is not connected to the network, features, or quantum preview.</p>
        <SensorSimulator />
      </details>
    </section>
  );
}

function HealthDatum({ label, value, tone }: { label: string; value: string; tone: string }) {
  return <div><span>{label}</span><strong className={`health-tone ${tone}`}>{value.toUpperCase()}</strong></div>;
}

function GlobalConfiguration({ configuration, onChange }: ConfigProps) {
  return (
    <fieldset>
      <legend>Session</legend>
      <label className="input-control full"><span className="input-label">Session name</span><input type="text" value={configuration.session_name} onChange={(event) => onChange({ ...configuration, session_name: event.target.value })} /></label>
      <div className="control-grid">
        <NumberInput label="Random seed" value={configuration.random_seed} min={0} step={1} onChange={(random_seed) => onChange({ ...configuration, random_seed })} />
        <NumberInput label="Sampling rate" unit="Hz" value={configuration.sampling_rate_Hz} min={1} max={2000} onChange={(sampling_rate_Hz) => onChange({ ...configuration, sampling_rate_Hz })} />
        <NumberInput label="UI refresh" unit="Hz" value={configuration.ui_refresh_rate_Hz} min={0.1} max={60} step={0.1} onChange={(ui_refresh_rate_Hz) => onChange({ ...configuration, ui_refresh_rate_Hz })} />
        <NumberInput label="Time scale" unit="×" value={configuration.time_scale} min={0.1} max={100} step={0.1} onChange={(time_scale) => onChange({ ...configuration, time_scale })} />
        <NumberInput label="Retained buffer" unit="s" value={configuration.buffer_duration_s} min={1} max={600} onChange={(buffer_duration_s) => onChange({ ...configuration, buffer_duration_s })} />
      </div>
      <p className="field-help">Backend validation limits the retained buffer to 20,000 frames.</p>
    </fieldset>
  );
}

function EnvironmentControls({ configuration, onChange }: ConfigProps) {
  const environment = configuration.environment;
  const updateEnvironment = (next: EnvironmentConfiguration) => onChange({ ...configuration, environment: next });
  const gradient = environment.gradient_T_per_m;
  const setGradient = (key: "xx" | "yy" | "xy" | "xz" | "yz", valueNtPerM: number) => {
    const xx = key === "xx" ? valueNtPerM * 1e-9 : gradient[0][0];
    const yy = key === "yy" ? valueNtPerM * 1e-9 : gradient[1][1];
    const xy = key === "xy" ? valueNtPerM * 1e-9 : gradient[0][1];
    const xz = key === "xz" ? valueNtPerM * 1e-9 : gradient[0][2];
    const yz = key === "yz" ? valueNtPerM * 1e-9 : gradient[1][2];
    const matrix: Matrix3 = [[xx, xy, xz], [xy, yy, yz], [xz, yz, -xx - yy]];
    updateEnvironment({ ...environment, gradient_T_per_m: matrix });
  };
  const dipole = environment.dipoles[0];

  return (
    <fieldset>
      <legend>Environment</legend>
      <VectorInputs label="Uniform field" unit="nT" vector={scaleVector(environment.uniform_field_T, 1e9)} onChange={(vector) => updateEnvironment({ ...environment, uniform_field_T: scaleVector(vector, 1e-9) })} />
      <VectorInputs label="Gradient reference position" unit="m" vector={environment.gradient_reference_position_m} onChange={(gradient_reference_position_m) => updateEnvironment({ ...environment, gradient_reference_position_m })} />
      <VectorInputs label="Common OU sigma" unit="nT" vector={scaleVector(environment.common_ou_sigma_T, 1e9)} min={0} onChange={(vector) => updateEnvironment({ ...environment, common_ou_sigma_T: scaleVector(vector, 1e-9) })} />
      <NumberInput label="Common OU correlation" unit="s" value={environment.common_ou_tau_s} min={0.001} step={0.1} onChange={(common_ou_tau_s) => updateEnvironment({ ...environment, common_ou_tau_s })} />
      <details className="advanced-controls">
        <summary>Source-free gradient tensor</summary>
        <p className="field-help">Enter five independent values. Symmetry and zero trace are preserved automatically; Gzz = −Gxx − Gyy.</p>
        <div className="control-grid">
          <NumberInput label="Gxx" unit="nT/m" value={gradient[0][0] * 1e9} onChange={(value) => setGradient("xx", value)} />
          <NumberInput label="Gyy" unit="nT/m" value={gradient[1][1] * 1e9} onChange={(value) => setGradient("yy", value)} />
          <NumberInput label="Gxy = Gyx" unit="nT/m" value={gradient[0][1] * 1e9} onChange={(value) => setGradient("xy", value)} />
          <NumberInput label="Gxz = Gzx" unit="nT/m" value={gradient[0][2] * 1e9} onChange={(value) => setGradient("xz", value)} />
          <NumberInput label="Gyz = Gzy" unit="nT/m" value={gradient[1][2] * 1e9} onChange={(value) => setGradient("yz", value)} />
          <NumberInput label="Gzz (derived)" unit="nT/m" value={gradient[2][2] * 1e9} disabled onChange={() => undefined} />
        </div>
      </details>
      {dipole && (
        <details className="advanced-controls">
          <summary>Dipole source · {dipole.source_id}</summary>
          <label className="toggle-control"><input type="checkbox" checked={dipole.enabled} onChange={(event) => updateEnvironment({ ...environment, dipoles: environment.dipoles.map((item, index) => index === 0 ? { ...item, enabled: event.target.checked } : item) })} /><span><strong>Enabled</strong><small>Point dipole with explicit minimum-distance domain.</small></span></label>
          <VectorInputs label="Initial position" unit="m" vector={dipole.initial_position_m} onChange={(initial_position_m) => updateFirstDipole(environment, updateEnvironment, { initial_position_m })} />
          <VectorInputs label="Velocity" unit="m/s" vector={dipole.velocity_m_per_s} onChange={(velocity_m_per_s) => updateFirstDipole(environment, updateEnvironment, { velocity_m_per_s })} />
          <VectorInputs label="Magnetic moment" unit="A·m²" vector={dipole.moment_A_m2} onChange={(moment_A_m2) => updateFirstDipole(environment, updateEnvironment, { moment_A_m2 })} />
          <NumberInput label="Minimum distance" unit="m" value={dipole.minimum_distance_m} min={0.001} step={0.01} onChange={(minimum_distance_m) => updateFirstDipole(environment, updateEnvironment, { minimum_distance_m })} />
        </details>
      )}
      <details className="advanced-controls">
        <summary>Gaussian field anomalies · {environment.gaussian_anomalies.length}</summary>
        <p className="field-help">Smooth world-frame environment providers. They are distinct from hidden device faults.</p>
        {environment.gaussian_anomalies.map((anomaly, index) => (
          <div className="event-row" key={anomaly.anomaly_id}>
            <label className="toggle-control"><input type="checkbox" checked={anomaly.enabled} onChange={(event) => updateEnvironment({ ...environment, gaussian_anomalies: environment.gaussian_anomalies.map((item, current) => current === index ? { ...item, enabled: event.target.checked } : item) })} /><span><strong>{anomaly.anomaly_id}</strong><small>World-frame Gaussian provider</small></span></label>
            <NumberInput label="Peak amplitude" unit="nT" value={anomaly.peak_amplitude_T * 1e9} onChange={(value) => updateGaussian(environment, updateEnvironment, index, { peak_amplitude_T: value * 1e-9 })} />
            <VectorInputs label="Direction" unit="unit vector" vector={anomaly.direction_world} onChange={(direction_world) => updateGaussian(environment, updateEnvironment, index, { direction_world })} />
            <VectorInputs label="Centre position" unit="m" vector={anomaly.center_position_m} onChange={(center_position_m) => updateGaussian(environment, updateEnvironment, index, { center_position_m })} />
            <NumberInput label="Spatial scale" unit="m" value={anomaly.spatial_scale_m} min={0.001} onChange={(spatial_scale_m) => updateGaussian(environment, updateEnvironment, index, { spatial_scale_m })} />
            <button type="button" className="text-button danger" onClick={() => updateEnvironment({ ...environment, gaussian_anomalies: environment.gaussian_anomalies.filter((_, current) => current !== index) })}>Remove anomaly</button>
          </div>
        ))}
        <button type="button" className="button button-secondary" onClick={() => updateEnvironment({ ...environment, gaussian_anomalies: [...environment.gaussian_anomalies, { anomaly_id: `anomaly-${crypto.randomUUID()}`, peak_amplitude_T: 100e-9, direction_world: [0, 0, 1], center_position_m: [0, 0, 0], spatial_scale_m: 0.25, enabled: true }] })}>Add Gaussian anomaly</button>
      </details>
      <details className="advanced-controls">
        <summary>Periodic world fields · {environment.periodic_fields.length}</summary>
        <p className="field-help">Explicit sinusoidal sources in the world frame. Frequency must remain below the session Nyquist frequency.</p>
        {environment.periodic_fields.map((source, index) => (
          <div className="event-row" key={source.source_id}>
            <label className="toggle-control"><input type="checkbox" checked={source.enabled} onChange={(event) => updatePeriodic(environment, updateEnvironment, index, { enabled: event.target.checked, frequency_Hz: event.target.checked ? frequencyBelowNyquist(source.frequency_Hz, configuration.sampling_rate_Hz) : source.frequency_Hz })} /><span><strong>{source.source_id}</strong><small>Periodic field provider</small></span></label>
            <VectorInputs label="Amplitude" unit="nT" vector={scaleVector(source.amplitude_world_T, 1e9)} onChange={(value) => updatePeriodic(environment, updateEnvironment, index, { amplitude_world_T: scaleVector(value, 1e-9) })} />
            <div className="control-grid"><NumberInput label="Frequency" unit="Hz" value={source.frequency_Hz} min={0.001} max={source.enabled ? frequencyBelowNyquist(1_000, configuration.sampling_rate_Hz) : 1_000} onChange={(frequency_Hz) => updatePeriodic(environment, updateEnvironment, index, { frequency_Hz })} /><NumberInput label="Phase" unit="rad" value={source.phase_rad} step={0.01} onChange={(phase_rad) => updatePeriodic(environment, updateEnvironment, index, { phase_rad })} /></div>
            <button type="button" className="text-button danger" onClick={() => updateEnvironment({ ...environment, periodic_fields: environment.periodic_fields.filter((_, current) => current !== index) })}>Remove periodic field</button>
          </div>
        ))}
        <button type="button" className="button button-secondary" onClick={() => updateEnvironment({ ...environment, periodic_fields: [...environment.periodic_fields, { source_id: `periodic-${crypto.randomUUID()}`, amplitude_world_T: [10e-9, 0, 0], frequency_Hz: Math.min(5, configuration.sampling_rate_Hz / 4), phase_rad: 0, enabled: true }] })}>Add periodic field</button>
      </details>
    </fieldset>
  );
}

function NodeControls({ node, samplingRateHz, blindMode, onChange }: {
  node: SensorNodeConfiguration;
  samplingRateHz: number;
  blindMode: boolean;
  onChange: (updater: (node: SensorNodeConfiguration) => SensorNodeConfiguration) => void;
}) {
  const updateErrors = (errors: NodeErrorConfiguration) => onChange((current) => ({ ...current, errors }));
  const temperatureDriver = node.errors.temperature_driver;
  const updateTemperatureDriver = (patch: Partial<TemperatureDriverConfiguration>) => {
    updateErrors({
      ...node.errors,
      temperature_driver: { ...(temperatureDriver ?? DEFAULT_TEMPERATURE_DRIVER), ...patch },
    });
  };
  return (
    <article className="workbench-panel node-editor">
      <div className="panel-heading-row"><div><p className="panel-kicker">SELECTED NODE</p><h2>{node.sensor_id}</h2></div><span className="result-badge">{node.measurement_mode.toUpperCase()}</span></div>
      <div className="control-grid three-columns">
        <label className="input-control"><span className="input-label">Role</span><select value={node.role} onChange={(event) => onChange((current) => ({ ...current, role: event.target.value as SensorNodeConfiguration["role"] }))}><option value="sensor">Sensor</option><option value="remote_reference">Remote reference</option></select></label>
        <label className="input-control"><span className="input-label">Measurement mode</span><select value={node.measurement_mode} onChange={(event) => onChange((current) => ({ ...current, measurement_mode: event.target.value as SensorNodeConfiguration["measurement_mode"] }))}><option value="vector">Vector</option><option value="monoaxial">Monoaxial</option><option value="total_field">Total field</option></select></label>
        <label className="input-control"><span className="input-label">Calibration version</span><input value={node.calibration_version} onChange={(event) => onChange((current) => ({ ...current, calibration_version: event.target.value }))} /></label>
      </div>
      <VectorInputs label="Position" unit="m" vector={node.position_m} onChange={(position_m) => onChange((current) => ({ ...current, position_m }))} />
      <VectorInputs label="Monoaxial direction" unit="unit vector" vector={node.monoaxial_axis_sensor} onChange={(monoaxial_axis_sensor) => onChange((current) => ({ ...current, monoaxial_axis_sensor }))} />
      <QuaternionInputs value={node.orientation_world_to_sensor_wxyz} onChange={(orientation_world_to_sensor_wxyz) => onChange((current) => ({ ...current, orientation_world_to_sensor_wxyz }))} />
      <details className="advanced-controls">
        <summary>Motion model</summary>
        <label className="input-control full"><span className="input-label">Motion kind</span><select value={node.motion.kind} onChange={(event) => onChange((current) => ({ ...current, motion: { ...current.motion, kind: event.target.value as SensorNodeConfiguration["motion"]["kind"] } }))}><option value="static">Static</option><option value="calibration_tumble">Calibration tumble</option><option value="high_dynamic">High dynamic</option><option value="linear_translation">Linear translation</option><option value="local_anomaly_crossing">Local anomaly crossing</option><option value="combined_stress">Combined stress</option></select></label>
        <VectorInputs label="Velocity" unit="m/s" vector={node.motion.velocity_m_per_s} onChange={(velocity_m_per_s) => onChange((current) => ({ ...current, motion: { ...current.motion, velocity_m_per_s } }))} />
        <VectorInputs label="Angular rate" unit="rad/s" vector={node.motion.angular_rate_rad_per_s} onChange={(angular_rate_rad_per_s) => onChange((current) => ({ ...current, motion: { ...current.motion, angular_rate_rad_per_s } }))} />
        <VectorInputs label="Modulation frequency" unit="Hz" vector={node.motion.modulation_frequency_Hz} min={0} onChange={(modulation_frequency_Hz) => onChange((current) => ({ ...current, motion: { ...current.motion, modulation_frequency_Hz } }))} />
      </details>
      {blindMode ? <p className="blind-configuration-boundary">Device-error, thermal-driver, clock, and saturation parameters are hidden. Blind mode is a local UI barrier, not an API security control.</p> : <details className="advanced-controls">
        <summary>Error, thermal, clock and bandwidth model</summary>
        <VectorInputs label="Bias" unit="nT" vector={scaleVector(node.errors.bias_T, 1e9)} onChange={(value) => updateErrors({ ...node.errors, bias_T: scaleVector(value, 1e-9) })} />
        <VectorInputs label="Deterministic drift" unit="nT/s" vector={scaleVector(node.errors.deterministic_drift_T_per_s, 1e9)} onChange={(value) => updateErrors({ ...node.errors, deterministic_drift_T_per_s: scaleVector(value, 1e-9) })} />
        <VectorInputs label="White-noise standard deviation" unit="nT/sample" vector={scaleVector(node.errors.white_noise_std_T_per_sample, 1e9)} min={0} onChange={(value) => updateErrors({ ...node.errors, white_noise_std_T_per_sample: scaleVector(value, 1e-9) })} />
        <VectorInputs label="Random-walk q" unit="T²/s" vector={node.errors.random_walk_q_T2_per_s} min={0} step={1e-23} onChange={(random_walk_q_T2_per_s) => updateErrors({ ...node.errors, random_walk_q_T2_per_s })} />
        <VectorInputs label="Node OU sigma" unit="nT" vector={scaleVector(node.errors.ou_sigma_T, 1e9)} min={0} onChange={(value) => updateErrors({ ...node.errors, ou_sigma_T: scaleVector(value, 1e-9) })} />
        <div className="control-grid three-columns">
          <NumberInput label="Node OU correlation" unit="s" value={node.errors.ou_tau_s} min={0.001} onChange={(ou_tau_s) => updateErrors({ ...node.errors, ou_tau_s })} />
          <NumberInput label="Initial temperature" unit="K" value={node.errors.initial_temperature_K} min={0.001} onChange={(initial_temperature_K) => updateErrors({ ...node.errors, initial_temperature_K })} />
          <NumberInput label="Ambient temperature" unit="K" value={node.errors.ambient_temperature_K} min={0.001} onChange={(ambient_temperature_K) => updateErrors({ ...node.errors, ambient_temperature_K })} />
          <NumberInput label="Thermal time constant" unit="s" value={node.errors.thermal_time_constant_s} min={0.001} onChange={(thermal_time_constant_s) => updateErrors({ ...node.errors, thermal_time_constant_s })} />
          <NumberInput label="Reference temperature" unit="K" value={node.errors.reference_temperature_K} min={0.001} onChange={(reference_temperature_K) => updateErrors({ ...node.errors, reference_temperature_K })} />
          <NumberInput label="Bandwidth" unit="Hz" value={node.errors.bandwidth_Hz ?? 0} min={0} onChange={(bandwidth_Hz) => updateErrors({ ...node.errors, bandwidth_Hz: bandwidth_Hz > 0 ? bandwidth_Hz : null })} />
          <NumberInput label="Saturation" unit="µT" value={node.errors.saturation_limit_T * 1e6} min={0.001} onChange={(value) => updateErrors({ ...node.errors, saturation_limit_T: value * 1e-6 })} />
          <NumberInput label="Clock offset" unit="s" value={node.errors.clock_offset_s} onChange={(clock_offset_s) => updateErrors({ ...node.errors, clock_offset_s })} />
          <NumberInput label="Clock drift" unit="ppm" value={node.errors.clock_drift_ppm} onChange={(clock_drift_ppm) => updateErrors({ ...node.errors, clock_drift_ppm })} />
          <NumberInput label="Dropout rate" unit="s⁻¹" value={node.errors.dropout_rate_per_s} min={0} onChange={(dropout_rate_per_s) => updateErrors({ ...node.errors, dropout_rate_per_s })} />
          <NumberInput label="Dropout duration" unit="s" value={node.errors.dropout_duration_s} min={0.001} onChange={(dropout_duration_s) => updateErrors({ ...node.errors, dropout_duration_s })} />
          <NumberInput label="Stuck rate" unit="s⁻¹" value={node.errors.stuck_rate_per_s} min={0} onChange={(stuck_rate_per_s) => updateErrors({ ...node.errors, stuck_rate_per_s })} />
          <NumberInput label="Stuck duration" unit="s" value={node.errors.stuck_duration_s} min={0.001} onChange={(stuck_duration_s) => updateErrors({ ...node.errors, stuck_duration_s })} />
        </div>
        <label className="input-control full"><span className="input-label">Ambient temperature driver</span><select value={temperatureDriver?.kind ?? "none"} onChange={(event) => {
          const kind = event.target.value;
          updateErrors({
            ...node.errors,
            temperature_driver: kind === "none"
              ? null
              : { ...(temperatureDriver ?? DEFAULT_TEMPERATURE_DRIVER), kind: kind as TemperatureDriverConfiguration["kind"] },
          });
        }}><option value="none">Legacy constant ambient target</option><option value="constant">Constant driver</option><option value="ramp">Bounded ramp</option><option value="sinusoidal">Sinusoidal driver</option></select></label>
        {temperatureDriver?.kind === "ramp" && <div className="control-grid"><NumberInput label="Ramp rate" unit="K/s" value={temperatureDriver.ramp_rate_K_per_s} onChange={(ramp_rate_K_per_s) => updateTemperatureDriver({ ramp_rate_K_per_s })} /><NumberInput label="Ramp duration" unit="s" value={temperatureDriver.ramp_duration_s} min={0.001} onChange={(ramp_duration_s) => updateTemperatureDriver({ ramp_duration_s })} /></div>}
        {temperatureDriver?.kind === "sinusoidal" && <div className="control-grid three-columns"><NumberInput label="Temperature amplitude" unit="K" value={temperatureDriver.sinusoidal_amplitude_K} min={0} onChange={(sinusoidal_amplitude_K) => updateTemperatureDriver({ sinusoidal_amplitude_K })} /><NumberInput label="Temperature frequency" unit="Hz" value={temperatureDriver.sinusoidal_frequency_Hz} min={0.001} max={frequencyBelowNyquist(1_000, samplingRateHz)} onChange={(sinusoidal_frequency_Hz) => updateTemperatureDriver({ sinusoidal_frequency_Hz })} /><NumberInput label="Temperature phase" unit="rad" value={temperatureDriver.sinusoidal_phase_rad} onChange={(sinusoidal_phase_rad) => updateTemperatureDriver({ sinusoidal_phase_rad })} /></div>}
        <label className="toggle-control"><input type="checkbox" checked={node.errors.saturation_limits_T !== null} onChange={(event) => updateErrors({ ...node.errors, saturation_limits_T: event.target.checked ? [node.errors.saturation_limit_T, node.errors.saturation_limit_T, node.errors.saturation_limit_T] : null })} /><span><strong>Axis-specific saturation override</strong><small>When disabled, every axis uses the legacy scalar saturation limit.</small></span></label>
        {node.errors.saturation_limits_T && <VectorInputs label="Axis saturation X / Y / Z" unit="µT" vector={scaleVector(node.errors.saturation_limits_T, 1e6)} min={0.001} onChange={(value) => updateErrors({ ...node.errors, saturation_limits_T: scaleVector(value, 1e-6) })} />}
        <VectorInputs label="Thermal bias" unit="nT/K" vector={scaleVector(node.errors.thermal_bias_T_per_K, 1e9)} onChange={(value) => updateErrors({ ...node.errors, thermal_bias_T_per_K: scaleVector(value, 1e-9) })} />
        <MatrixInputs label="Gain matrix" matrix={node.errors.gain_matrix} onChange={(gain_matrix) => updateErrors({ ...node.errors, gain_matrix })} />
        <MatrixInputs label="Soft-iron matrix" matrix={node.errors.soft_iron_matrix} onChange={(soft_iron_matrix) => updateErrors({ ...node.errors, soft_iron_matrix })} />
        <MatrixInputs label="Cross-axis matrix" matrix={node.errors.cross_axis_matrix} onChange={(cross_axis_matrix) => updateErrors({ ...node.errors, cross_axis_matrix })} />
      </details>}
    </article>
  );
}

function EventControls({ configuration, selectedNodeId, onChange }: ConfigProps & { selectedNodeId: string | null }) {
  const addEvent = () => {
    const event: NetworkEventConfiguration = {
      event_id: `event-${crypto.randomUUID()}`,
      kind: "dropout",
      start_time_s: 1,
      duration_s: 0.5,
      target_sensor_ids: selectedNodeId ? [selectedNodeId] : [configuration.nodes[0].sensor_id],
      field_offset_T: [0, 0, 0],
      drift_rate_T_per_s: [0, 0, 0],
      noise_multiplier: 2,
    };
    onChange({ ...configuration, events: [...configuration.events, event] });
  };
  const updateEvent = (index: number, patch: Partial<NetworkEventConfiguration>) => {
    const events = configuration.events.map((event, current) => current === index ? { ...event, ...patch } : event);
    onChange({ ...configuration, events });
  };
  return (
    <fieldset>
      <legend>Scheduled events</legend>
      <p className="field-help">Draft events are included when a new session is created. Live manual injection is not connected in this UI.</p>
      {configuration.events.map((event, index) => {
        const minimumTargets = event.kind === "shared_instrument_offset" ? 2 : 1;
        return (
        <div className="event-row" key={event.event_id}>
          <div className="control-grid">
            <label className="input-control"><span className="input-label">Kind</span><select value={event.kind} onChange={(change) => {
              const kind = change.target.value as EventKind;
              updateEvent(index, { kind, target_sensor_ids: targetsForEventKind(kind, event.target_sensor_ids, configuration.nodes) });
            }}>{EVENT_KINDS.map((kind) => <option key={kind} value={kind} disabled={kind === "shared_instrument_offset" && configuration.nodes.length < 2}>{kind.replaceAll("_", " ")}</option>)}</select></label>
            <div className="input-control"><span className="input-label">Targets</span><div className="event-targets">{configuration.nodes.map((node) => {
              const checked = event.target_sensor_ids.includes(node.sensor_id);
              return <label key={node.sensor_id}><input type="checkbox" disabled={event.kind === "world_field_offset" || (checked && event.target_sensor_ids.length <= minimumTargets)} checked={checked} onChange={(change) => updateEvent(index, { target_sensor_ids: change.target.checked ? [...event.target_sensor_ids, node.sensor_id] : event.target_sensor_ids.filter((sensorId) => sensorId !== node.sensor_id) })} /> {node.sensor_id}</label>;
            })}</div>{event.kind === "shared_instrument_offset" && <small>Shared offsets require at least two target sensors.</small>}</div>
            <NumberInput label="Start" unit="s" value={event.start_time_s} min={0} onChange={(start_time_s) => updateEvent(index, { start_time_s })} />
            <NumberInput label="Duration" unit="s" value={event.duration_s} min={0.001} onChange={(duration_s) => updateEvent(index, { duration_s })} />
            {event.kind === "node_noise_burst" && <NumberInput label="Noise multiplier" unit="×" value={event.noise_multiplier} min={1.0001} onChange={(noise_multiplier) => updateEvent(index, { noise_multiplier })} />}
          </div>
          {["world_field_offset", "node_bias", "shared_instrument_offset"].includes(event.kind) && <VectorInputs label="Field offset" unit="nT" vector={scaleVector(event.field_offset_T, 1e9)} onChange={(value) => updateEvent(index, { field_offset_T: scaleVector(value, 1e-9) })} />}
          {event.kind === "node_drift" && <VectorInputs label="Drift rate" unit="nT/s" vector={scaleVector(event.drift_rate_T_per_s, 1e9)} onChange={(value) => updateEvent(index, { drift_rate_T_per_s: scaleVector(value, 1e-9) })} />}
          <button type="button" className="text-button danger" onClick={() => onChange({ ...configuration, events: configuration.events.filter((_, current) => current !== index) })}>Remove event</button>
        </div>
      );})}
      <button type="button" className="button button-secondary" onClick={addEvent} disabled={configuration.events.length >= 128}>Add event</button>
    </fieldset>
  );
}

type LiveEventStatus = "requesting" | "scheduled" | "rejected" | "stale";

type LiveEventLogEntry = {
  requestId: string;
  eventId: string;
  sessionId: string;
  requestedAt: string;
  status: LiveEventStatus;
  summary: string;
  detail: string;
};

type LiveEventScheduler = (
  sessionId: string,
  event: NetworkEventConfiguration,
  signal?: AbortSignal,
) => Promise<ScheduledEventResponse>;

export type LiveEventRequestOutcome =
  | { status: "scheduled"; response: ScheduledEventResponse; detail: string }
  | { status: "rejected"; detail: string }
  | { status: "stale"; response?: ScheduledEventResponse; detail: string };

type LiveEventControlsProps = {
  session: SessionStatus | null;
  configuration: NetworkSessionConfiguration | null;
  selectedNodeId: string | null;
};

export function LiveEventControls({
  session,
  configuration,
  selectedNodeId,
}: LiveEventControlsProps) {
  const sessionId = session?.session_id ?? null;
  const sessionIdRef = useRef(sessionId);
  const activeRequestRef = useRef<{
    requestId: string;
    sessionId: string;
    controller: AbortController;
  } | null>(null);
  const [intent, setIntent] = useState<LiveEventIntent>("environment_common");
  const [targetNodeId, setTargetNodeId] = useState(selectedNodeId ?? "");
  const [leadTimeS, setLeadTimeS] = useState(5);
  const [durationS, setDurationS] = useState(2);
  const [magnitude, setMagnitude] = useState(100);
  const [pendingRequestId, setPendingRequestId] = useState<string | null>(null);
  const [log, setLog] = useState<LiveEventLogEntry[]>([]);
  sessionIdRef.current = sessionId;

  useEffect(() => {
    const active = activeRequestRef.current;
    if (!active || active.sessionId === sessionId) return;
    active.controller.abort();
    activeRequestRef.current = null;
    setPendingRequestId(null);
    setLog((current) => updateLiveEventLog(current, active.requestId, {
      status: "stale",
      detail: "The UI session changed while the request was pending. Backend acceptance was not confirmed.",
    }));
  }, [sessionId]);

  useEffect(() => () => {
    activeRequestRef.current?.controller.abort();
    activeRequestRef.current = null;
  }, []);

  const nodes = configuration?.nodes ?? [];
  const effectiveTargetId = nodes.some(({ sensor_id }) => sensor_id === targetNodeId)
    ? targetNodeId
    : nodes.some(({ sensor_id }) => sensor_id === selectedNodeId)
      ? selectedNodeId ?? ""
      : nodes[0]?.sensor_id ?? "";
  const option = LIVE_EVENT_OPTIONS.find((candidate) => candidate.intent === intent)!;
  const minimumLeadS = configuration ? minimumLiveEventLeadS(configuration) : 0.1;
  const effectiveLeadS = Math.max(leadTimeS, minimumLeadS);
  const isSharedUnavailable = intent === "shared_instrument" && nodes.length < 2;
  const sessionUnavailable = !session || !configuration;
  const stopped = session?.state === "stopped";
  const canSchedule = !sessionUnavailable
    && !stopped
    && !isSharedUnavailable
    && !pendingRequestId;

  const schedule = async () => {
    if (!session || !configuration || !canSchedule || activeRequestRef.current) return;
    const requestId = `live-request-${crypto.randomUUID()}`;
    const eventId = `live-${intent}-${crypto.randomUUID()}`;
    let event: NetworkEventConfiguration;
    try {
      event = buildLiveNetworkEvent({
        eventId,
        intent,
        simTimeS: session.sim_time_s,
        leadTimeS,
        durationS,
        magnitude,
        targetSensorId: effectiveTargetId,
        configuration,
      });
    } catch (error: unknown) {
      setLog((current) => prependLiveEventLog(current, {
        requestId,
        eventId,
        sessionId: session.session_id,
        requestedAt: new Date().toISOString(),
        status: "rejected",
        summary: option.label,
        detail: errorMessage(error, "The live event request is invalid."),
      }));
      return;
    }

    const controller = new AbortController();
    activeRequestRef.current = { requestId, sessionId: session.session_id, controller };
    setPendingRequestId(requestId);
    setLog((current) => prependLiveEventLog(current, {
      requestId,
      eventId,
      sessionId: session.session_id,
      requestedAt: new Date().toISOString(),
      status: "requesting",
      summary: option.label,
      detail: `Requesting activation at simulated t=${event.start_time_s.toFixed(6)} s.`,
    }));

    const outcome = await submitLiveNetworkEvent({
      sessionId: session.session_id,
      event,
      signal: controller.signal,
      isCurrent: () => (
        sessionIdRef.current === session.session_id
        && activeRequestRef.current?.requestId === requestId
      ),
    });
    if (activeRequestRef.current?.requestId === requestId) {
      activeRequestRef.current = null;
      setPendingRequestId(null);
    }
    setLog((current) => updateLiveEventLog(current, requestId, {
      status: outcome.status,
      detail: outcome.detail,
    }));
  };

  return (
    <fieldset>
      <legend>Live event injection</legend>
      <p className="field-help">
        Submit an explicit event to the current backend session. A SCHEDULED entry confirms backend acceptance only; observe the data and quality flags to confirm the physical effect.
      </p>
      {sessionUnavailable && <p className="inline-message warning-message">Create a session before scheduling a live event. Draft events above are used only when the next session is created.</p>}
      {stopped && <p className="inline-message warning-message">The current session is stopped. Reset, replay, or create a new session before scheduling another event.</p>}
      <div className="control-grid">
        <label className="input-control">
          <span className="input-label">Event</span>
          <select
            value={intent}
            disabled={sessionUnavailable || Boolean(pendingRequestId)}
            onChange={(change) => {
              const next = change.target.value as LiveEventIntent;
              setIntent(next);
              setMagnitude(LIVE_EVENT_OPTIONS.find(({ intent: value }) => value === next)!.defaultMagnitude);
            }}
          >
            {LIVE_EVENT_OPTIONS.map((eventOption) => (
              <option
                key={eventOption.intent}
                value={eventOption.intent}
                disabled={eventOption.intent === "shared_instrument" && nodes.length < 2}
              >
                {eventOption.label}
              </option>
            ))}
          </select>
          <small>{option.description}</small>
        </label>
        <label className="input-control">
          <span className="input-label">Target node</span>
          <select
            value={effectiveTargetId}
            disabled={sessionUnavailable || intent === "environment_common" || intent === "shared_instrument" || Boolean(pendingRequestId)}
            onChange={(change) => setTargetNodeId(change.target.value)}
          >
            {nodes.map((node) => <option key={node.sensor_id} value={node.sensor_id}>{node.sensor_id}</option>)}
          </select>
          <small>{intent === "environment_common" ? "Environment-wide; no device target." : intent === "shared_instrument" ? "All nodes in the executed session." : "One node from the executed session."}</small>
        </label>
        <NumberInput label="Lead from backend time" unit="s" value={leadTimeS} min={0.001} step={0.1} disabled={sessionUnavailable || Boolean(pendingRequestId)} onChange={setLeadTimeS} />
        <NumberInput label="Duration" unit="s" value={durationS} min={0.001} step={0.1} disabled={sessionUnavailable || Boolean(pendingRequestId)} onChange={setDurationS} />
        {liveEventMagnitudeUnit(intent) && (
          <NumberInput
            label={intent === "node_noise" ? "Noise multiplier" : "X-axis magnitude"}
            unit={liveEventMagnitudeUnit(intent) ?? undefined}
            value={magnitude}
            min={intent === "node_noise" ? 1.0001 : undefined}
            step={intent === "node_noise" ? 0.1 : 1}
            disabled={sessionUnavailable || Boolean(pendingRequestId)}
            onChange={setMagnitude}
          />
        )}
      </div>
      {configuration && (
        <p className="field-help">
          Backend time: {session?.sim_time_s.toFixed(6) ?? "—"} s · requested lead: {leadTimeS.toFixed(3)} s · enforced lead: {effectiveLeadS.toFixed(3)} s · planned start: {futureLiveEventStart(session?.sim_time_s ?? 0, leadTimeS, configuration).toFixed(6)} s.
        </p>
      )}
      {intent === "clipping" && configuration && effectiveTargetId && (
        <p className="field-help">The generated node-bias stress is derived from the lowest configured axis limit. Scheduling does not claim that clipping occurred; verify the observed saturation mask and clipped quality flag.</p>
      )}
      {isSharedUnavailable && <p className="inline-message warning-message">A shared instrument event requires at least two nodes in the executed session.</p>}
      <button type="button" className="button button-primary" disabled={!canSchedule} onClick={() => void schedule()}>
        {pendingRequestId ? "Scheduling…" : "Schedule live event"}
      </button>

      <div aria-live="polite">
        <div className="panel-heading-row">
          <div><p className="panel-kicker">LIVE REQUEST REGISTER</p><h3>Accepted and rejected events</h3></div>
          {log.length > 0 && <button type="button" className="text-button" onClick={() => setLog([])} disabled={Boolean(pendingRequestId)}>Clear register</button>}
        </div>
        {log.length === 0 ? <p className="empty-copy">No live event request has been sent from this worksheet.</p> : log.map((entry) => (
          <div className="event-row" key={entry.requestId}>
            <div className="panel-heading-row">
              <div><strong>{entry.summary}</strong><p className="field-help">{entry.eventId}</p></div>
              <span className={entry.status === "scheduled" || entry.status === "requesting" ? "result-badge" : "result-badge stale"}>{entry.status.toUpperCase()}</span>
            </div>
            <dl className="quality-grid">
              <div><dt>Session</dt><dd>{entry.sessionId}</dd></div>
              <div><dt>Requested</dt><dd>{entry.requestedAt}</dd></div>
              <div className="span-two"><dt>Backend outcome</dt><dd>{entry.detail}</dd></div>
            </dl>
          </div>
        ))}
      </div>
    </fieldset>
  );
}

export function buildLiveNetworkEvent({
  eventId,
  intent,
  simTimeS,
  leadTimeS,
  durationS,
  magnitude,
  targetSensorId,
  configuration,
}: {
  eventId: string;
  intent: LiveEventIntent;
  simTimeS: number;
  leadTimeS: number;
  durationS: number;
  magnitude: number;
  targetSensorId: string;
  configuration: NetworkSessionConfiguration;
}): NetworkEventConfiguration {
  const target = configuration.nodes.find(({ sensor_id }) => sensor_id === targetSensorId);
  if (intent !== "environment_common" && intent !== "shared_instrument" && !target) {
    throw new Error("Select a node from the executed session.");
  }
  if (intent === "shared_instrument" && configuration.nodes.length < 2) {
    throw new Error("Shared instrument events require at least two nodes.");
  }
  if (![simTimeS, leadTimeS, durationS, magnitude].every(Number.isFinite)) {
    throw new Error("Live event parameters must be finite numbers.");
  }

  const kind = liveEventKind(intent);
  const targetSensorIds = intent === "environment_common"
    ? []
    : intent === "shared_instrument"
      ? configuration.nodes.map(({ sensor_id }) => sensor_id)
      : [target!.sensor_id];
  const fieldOffsetT: Vector3 = intent === "clipping"
    ? clippingStressVector(target!)
    : [
        ["environment_common", "node_bias", "shared_instrument"].includes(intent)
          ? magnitude * 1e-9
          : 0,
        0,
        0,
      ];

  return {
    event_id: eventId,
    kind,
    start_time_s: futureLiveEventStart(simTimeS, leadTimeS, configuration),
    duration_s: Math.max(0.001, durationS),
    target_sensor_ids: targetSensorIds,
    field_offset_T: fieldOffsetT,
    drift_rate_T_per_s: intent === "node_drift" ? [magnitude * 1e-9, 0, 0] : [0, 0, 0],
    noise_multiplier: intent === "node_noise" ? Math.max(1.0001, magnitude) : 1,
  };
}

export function minimumLiveEventLeadS(configuration: Pick<NetworkSessionConfiguration, "sampling_rate_Hz" | "time_scale">): number {
  return Math.max(0.1, 2 / configuration.sampling_rate_Hz, configuration.time_scale * 0.25);
}

export function futureLiveEventStart(
  simTimeS: number,
  requestedLeadS: number,
  configuration: Pick<NetworkSessionConfiguration, "sampling_rate_Hz" | "time_scale">,
): number {
  const lead = Math.max(requestedLeadS, minimumLiveEventLeadS(configuration));
  return Math.ceil((simTimeS + lead) * 1e6) / 1e6;
}

export async function submitLiveNetworkEvent({
  sessionId,
  event,
  signal,
  isCurrent,
  scheduler = scheduleNetworkEvent,
}: {
  sessionId: string;
  event: NetworkEventConfiguration;
  signal?: AbortSignal;
  isCurrent: () => boolean;
  scheduler?: LiveEventScheduler;
}): Promise<LiveEventRequestOutcome> {
  if (!isCurrent()) {
    return { status: "stale", detail: "The request was invalidated before transmission." };
  }
  try {
    const response = await scheduler(sessionId, event, signal);
    if (!isCurrent()) {
      return {
        status: "stale",
        response,
        detail: `The backend accepted this event for the previous session (${sessionId}), but the response is not attributed to the current session.`,
      };
    }
    return {
      status: "scheduled",
      response,
      detail: `Backend accepted · configuration v${response.configuration_version} · effective from frame ${response.effective_frame_id} · activation requested at t=${response.event.start_time_s.toFixed(6)} s.`,
    };
  } catch (error: unknown) {
    if (!isCurrent()) {
      return {
        status: "stale",
        detail: "The request was invalidated while in flight. Backend acceptance was not confirmed.",
      };
    }
    return {
      status: "rejected",
      detail: errorMessage(error, "The backend rejected the live event request."),
    };
  }
}

function liveEventKind(intent: LiveEventIntent): EventKind {
  switch (intent) {
    case "environment_common": return "world_field_offset";
    case "node_bias":
    case "clipping": return "node_bias";
    case "node_drift": return "node_drift";
    case "node_noise": return "node_noise_burst";
    case "shared_instrument": return "shared_instrument_offset";
    case "dropout": return "dropout";
    case "stuck": return "stuck";
  }
}

function liveEventMagnitudeUnit(intent: LiveEventIntent): string | null {
  if (["environment_common", "node_bias", "shared_instrument"].includes(intent)) return "nT";
  if (intent === "node_drift") return "nT/s";
  if (intent === "node_noise") return "×";
  return null;
}

function clippingStressVector(node: SensorNodeConfiguration): Vector3 {
  const limits = node.errors.saturation_limits_T ?? [
    node.errors.saturation_limit_T,
    node.errors.saturation_limit_T,
    node.errors.saturation_limit_T,
  ];
  const axis = limits.reduce((lowest, value, index) => value < limits[lowest] ? index : lowest, 0);
  const stress = Math.min(10, Math.max(limits[axis] * 4, limits[axis] + 1e-9));
  return [axis === 0 ? stress : 0, axis === 1 ? stress : 0, axis === 2 ? stress : 0];
}

function prependLiveEventLog(current: LiveEventLogEntry[], entry: LiveEventLogEntry): LiveEventLogEntry[] {
  return [entry, ...current].slice(0, 20);
}

function updateLiveEventLog(
  current: LiveEventLogEntry[],
  requestId: string,
  patch: Pick<LiveEventLogEntry, "status" | "detail">,
): LiveEventLogEntry[] {
  return current.map((entry) => entry.requestId === requestId ? { ...entry, ...patch } : entry);
}

function ObservationQuality({ reading }: { reading: ReturnType<typeof readingForSensor> | undefined }) {
  return (
    <article className="workbench-panel compact-panel">
      <div className="panel-heading-row"><div><p className="panel-kicker">OBSERVATION QUALITY</p><h2>Latest selected reading</h2></div></div>
      {!reading ? <p className="empty-copy">No reading received for the selected node.</p> : (
        <dl className="quality-grid">
          <div><dt>Sequence</dt><dd>{reading.sequence_id}</dd></div>
          <div><dt>Valid</dt><dd>{reading.valid ? "yes" : "no"}</dd></div>
          <div><dt>Mode</dt><dd>{reading.measurement_mode}</dd></div>
          <div><dt>Temperature</dt><dd>{reading.observed_temperature_K.toFixed(3)} K</dd></div>
          <div className="span-two"><dt>Quality flags</dt><dd>{reading.quality_flags.length ? reading.quality_flags.join(", ") : "none"}</dd></div>
        </dl>
      )}
    </article>
  );
}

function TruthPanel() {
  const { state } = useWorkbench();
  const latest = state.network.truth.at(-1);
  const selected = latest?.fields.find(({ sensor_id }) => sensor_id === state.network.selected_node_id);
  const matchingObservation = latest
    ? [...state.network.observations].reverse().find(({ frame_id }) => frame_id === latest.frame_id)
    : undefined;
  const measured = matchingObservation && state.network.selected_node_id
    ? readingForSensor(matchingObservation, state.network.selected_node_id)
    : null;
  const residual = measured?.components_T && selected?.ideal_field_sensor_T
    ? measured.components_T.map((value, index) => value - selected.ideal_field_sensor_T![index]) as unknown as Vector3
    : null;
  return (
    <article className="workbench-panel truth-panel">
      <div className="panel-heading-row"><div><p className="panel-kicker">SEPARATE VALIDATION CHANNEL</p><h2>Simulator truth</h2></div><span className={state.network.blind_mode ? "result-badge stale" : "result-badge"}>{state.network.blind_mode ? "HIDDEN" : "VISIBLE"}</span></div>
      {state.network.blind_mode ? (
        <p className="empty-copy">Blind mode discards truth frames in browser state. Predictions and downstream inputs are unaffected.</p>
      ) : selected ? (
        <>
          <dl className="quality-grid">
            <div><dt>Frame</dt><dd>{latest?.frame_id}</dd></div>
            <div><dt>Model valid</dt><dd>{selected.model_valid ? "yes" : "no"}</dd></div>
            <div className="span-two"><dt>True world field</dt><dd>{formatVectorNt(selected.field_true_world_T, "outside declared model domain")}</dd></div>
            <div className="span-two"><dt>Ideal field in sensor frame</dt><dd>{formatVectorNt(selected.ideal_field_sensor_T, "unavailable: model domain invalid")}</dd></div>
            <div className="span-two"><dt>Measured observed vector</dt><dd>{formatVectorNt(measured?.components_T, measured ? `unavailable for ${measured.measurement_mode} readout` : "matching observation frame unavailable")}</dd></div>
            <div className="span-two"><dt>Measured − ideal residual</dt><dd>{formatVectorNt(residual, "unavailable unless matching ideal and observed vectors exist")}</dd></div>
          </dl>
          <details className="advanced-controls truth-breakdown">
            <summary>World-frame component breakdown</summary>
            <dl className="quality-grid">
              <div><dt>Uniform</dt><dd>{formatVectorNt(selected.uniform_field_world_T)}</dd></div>
              <div><dt>Common</dt><dd>{formatVectorNt(selected.common_field_world_T)}</dd></div>
              <div><dt>Gradient</dt><dd>{formatVectorNt(selected.gradient_field_world_T)}</dd></div>
              <div><dt>Dipole</dt><dd>{formatVectorNt(selected.dipole_field_world_T, "not configured / unavailable")}</dd></div>
              <div><dt>Anomaly</dt><dd>{formatVectorNt(selected.anomaly_field_world_T)}</dd></div>
              <div><dt>Periodic</dt><dd>{formatVectorNt(selected.periodic_field_world_T)}</dd></div>
              <div><dt>Event</dt><dd>{formatVectorNt(selected.event_field_world_T)}</dd></div>
              <div><dt>Active causes</dt><dd>{latest?.active_causes.length ? latest.active_causes.map(({ event_id }) => event_id).join(", ") : "none"}</dd></div>
            </dl>
          </details>
          <p className="chart-note">This panel performs only a deterministic frame-aligned comparison. It is not a prediction or inferred decomposition.</p>
        </>
      ) : <p className="empty-copy">Truth is enabled; a validation frame has not yet been received.</p>}
    </article>
  );
}

type ConfigProps = { configuration: NetworkSessionConfiguration; onChange: (configuration: NetworkSessionConfiguration) => void };

function NumberInput({ label, unit, value, onChange, min, max, step = 0.01, disabled = false }: { label: string; unit?: string; value: number; onChange: (value: number) => void; min?: number; max?: number; step?: number; disabled?: boolean }) {
  return <label className="input-control"><span className="input-label">{label}{unit && <small>{unit}</small>}</span><input type="number" value={Number.isFinite(value) ? value : ""} min={min} max={max} step={step} disabled={disabled} onChange={(event: ChangeEvent<HTMLInputElement>) => { if (Number.isFinite(event.target.valueAsNumber)) onChange(event.target.valueAsNumber); }} /></label>;
}

function VectorInputs({ label, unit, vector, onChange, min, step = 0.01 }: { label: string; unit: string; vector: Vector3; onChange: (value: Vector3) => void; min?: number; step?: number }) {
  return <div className="vector-input-group"><span>{label}<small>{unit}</small></span><div className="vector-inputs">{(["X", "Y", "Z"] as const).map((axis, index) => <NumberInput key={axis} label={axis} value={vector[index]} min={min} step={step} onChange={(value) => onChange(vector.map((entry, current) => current === index ? value : entry) as unknown as Vector3)} />)}</div></div>;
}

function QuaternionInputs({ value, onChange }: { value: readonly [number, number, number, number]; onChange: (value: readonly [number, number, number, number]) => void }) {
  return <div className="vector-input-group"><span>Orientation<small>normalized quaternion wxyz</small></span><div className="quaternion-inputs">{(["W", "X", "Y", "Z"] as const).map((axis, index) => <NumberInput key={axis} label={axis} value={value[index]} step={0.001} onChange={(entry) => onChange(value.map((current, currentIndex) => currentIndex === index ? entry : current) as unknown as readonly [number, number, number, number])} />)}</div></div>;
}

function MatrixInputs({ label, matrix, onChange }: { label: string; matrix: Matrix3; onChange: (matrix: Matrix3) => void }) {
  return <div className="matrix-input-group"><span>{label}</span><div className="matrix-inputs">{matrix.flatMap((row, rowIndex) => row.map((value, columnIndex) => <NumberInput key={`${rowIndex}-${columnIndex}`} label={`${rowIndex + 1},${columnIndex + 1}`} value={value} step={0.001} onChange={(entry) => onChange(matrix.map((currentRow, currentRowIndex) => currentRow.map((current, currentColumnIndex) => currentRowIndex === rowIndex && currentColumnIndex === columnIndex ? entry : current) as unknown as Vector3) as unknown as Matrix3)} />))}</div></div>;
}

function updateFirstDipole(environment: EnvironmentConfiguration, apply: (environment: EnvironmentConfiguration) => void, patch: Partial<EnvironmentConfiguration["dipoles"][number]>) {
  apply({ ...environment, dipoles: environment.dipoles.map((dipole, index) => index === 0 ? { ...dipole, ...patch } : dipole) });
}

function updateGaussian(environment: EnvironmentConfiguration, apply: (environment: EnvironmentConfiguration) => void, index: number, patch: Partial<EnvironmentConfiguration["gaussian_anomalies"][number]>) {
  apply({ ...environment, gaussian_anomalies: environment.gaussian_anomalies.map((anomaly, current) => current === index ? { ...anomaly, ...patch } : anomaly) });
}

function updatePeriodic(environment: EnvironmentConfiguration, apply: (environment: EnvironmentConfiguration) => void, index: number, patch: Partial<EnvironmentConfiguration["periodic_fields"][number]>) {
  apply({ ...environment, periodic_fields: environment.periodic_fields.map((source, current) => current === index ? { ...source, ...patch } : source) });
}

function scaleVector(vector: Vector3, factor: number): Vector3 {
  return vector.map((value) => value * factor) as unknown as Vector3;
}

function formatVectorNt(vector: Vector3 | null | undefined, unavailable = "not available"): string {
  return vector
    ? `[${vector.map((value) => `${(value * 1e9).toFixed(3)} nT`).join(", ")}]`
    : unavailable;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

export function nextRandomSeed(currentSeed: number): number {
  const randomValue = crypto.getRandomValues(new Uint32Array(1))[0];
  return randomValue === currentSeed ? (currentSeed + 1) >>> 0 : randomValue;
}

function frequencyBelowNyquist(frequencyHz: number, samplingRateHz: number): number {
  const strictUpperBound = samplingRateHz / 2 * (1 - 1e-9);
  return Math.min(frequencyHz, strictUpperBound);
}

export function targetsForEventKind(
  kind: EventKind,
  currentTargets: string[],
  nodes: SensorNodeConfiguration[],
): string[] {
  if (kind === "world_field_offset") return [];
  const knownIds = new Set(nodes.map(({ sensor_id }) => sensor_id));
  const targets = currentTargets.filter((sensorId) => knownIds.has(sensorId));
  const minimum = kind === "shared_instrument_offset" ? 2 : 1;
  for (const node of nodes) {
    if (targets.length >= minimum) break;
    if (!targets.includes(node.sensor_id)) targets.push(node.sensor_id);
  }
  return targets;
}
