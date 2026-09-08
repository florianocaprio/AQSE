import { useState, type ChangeEvent } from "react";

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
  SensorNodeConfiguration,
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
                : <><EnvironmentControls configuration={draft} onChange={patchDraft} /><EventControls configuration={draft} selectedNodeId={selectedNode?.sensor_id ?? null} onChange={patchDraft} /></>}
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
