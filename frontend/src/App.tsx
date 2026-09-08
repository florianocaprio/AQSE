import { useEffect, useState } from "react";

import { QuantumCircuitControls } from "./components/QuantumCircuitControls";
import { SensorSimulator } from "./components/SensorSimulator";
import type { HealthResponse, QuantumHealthResponse } from "./types";

function App() {
  const [backend, setBackend] = useState<HealthResponse | null>(null);
  const [quantum, setQuantum] = useState<QuantumHealthResponse | null>(null);
  const [backendError, setBackendError] = useState(false);
  const [quantumError, setQuantumError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    void fetch("/api/health", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Backend unavailable");
        return response.json() as Promise<HealthResponse>;
      })
      .then((payload) => {
        setBackend(payload);
        setBackendError(false);
      })
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setBackendError(true);
        }
      });

    void fetch("/api/quantum/health", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Quantum engine unavailable");
        return response.json() as Promise<QuantumHealthResponse>;
      })
      .then((payload) => {
        setQuantum(payload);
        setQuantumError(false);
      })
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setQuantumError(true);
        }
      });

    return () => controller.abort();
  }, []);

  const statusLabel = (ready: boolean, failed: boolean) =>
    failed ? "UNAVAILABLE" : ready ? "READY" : "CHECKING";

  return (
    <main className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">AQSE · LOCAL RESEARCH DEMONSTRATOR</p>
          <h1>AQSE — Adaptive Quantum Sensor Engine</h1>
          <p className="summary">
            Milestone 1B workspace for synthetic sensor acquisition and
            infrastructure inspection. No sensor data is sent to the quantum
            engine.
          </p>
        </div>
        <div className="milestone-tag" aria-label="Current milestone">
          <span>MILESTONE</span>
          <strong>1B</strong>
          <small>Sensor simulation</small>
        </div>
      </header>

      <section className="system-section" aria-labelledby="system-status-heading">
        <div className="system-heading">
          <div>
            <p className="section-index">SYSTEM</p>
            <h2 id="system-status-heading">Environment status</h2>
          </div>
          <p>Local Docker services and validated quantum infrastructure</p>
        </div>
        <div aria-live="polite" className="status-panel">
          <StatusRow
            label="Backend"
            detail={backend?.service}
            value={statusLabel(backend?.status === "ok", backendError)}
            unavailable={backendError}
          />
          <StatusRow
            label="Quantum Engine"
            detail={quantum?.engine}
            value={statusLabel(quantum?.status === "ok", quantumError)}
            unavailable={quantumError}
          />
          <StatusRow
            label="Qiskit"
            detail={quantum?.simulation}
            value={statusLabel(quantum?.qiskit === "ready", quantumError)}
            unavailable={quantumError}
          />
          <StatusRow label="Qubits" value={quantum?.qubits ?? "—"} />
          <StatusRow label="Features" value={quantum?.features ?? "—"} />
          <StatusRow
            label="Trainable parameters"
            value={quantum?.trainable_parameters ?? "—"}
          />
          <StatusRow
            label="Sensor Simulator"
            detail="Quantum magnetometer"
            value={statusLabel(backend?.status === "ok", backendError)}
            unavailable={backendError}
          />
          <StatusRow label="AFSE Embedding" value="NOT IMPLEMENTED" muted />
        </div>
      </section>

      <SensorSimulator />
      <QuantumCircuitControls />

      <footer>
        <span>AQSE Milestone 1B</span>
        <span>Local simulation only · No QPU execution</span>
      </footer>
    </main>
  );
}

type StatusRowProps = {
  label: string;
  value: string | number;
  detail?: string;
  unavailable?: boolean;
  muted?: boolean;
};

function StatusRow({
  label,
  value,
  detail,
  unavailable = false,
  muted = false,
}: StatusRowProps) {
  const className = unavailable
    ? "status-value error"
    : muted
      ? "status-value muted"
      : "status-value";

  return (
    <div className="status-row">
      <span className="status-label">
        {label}
        {detail && <small>{detail}</small>}
      </span>
      <strong className={className}>{value}</strong>
    </div>
  );
}

export default App;
