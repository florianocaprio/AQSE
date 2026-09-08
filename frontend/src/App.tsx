import { useEffect, useState } from "react";

type HealthResponse = {
  status: string;
  service: string;
};

type QuantumHealthResponse = {
  status: string;
  engine: string;
  qiskit: string;
  qubits: number;
  features: number;
  trainable_parameters: number;
};

function App() {
  const [backend, setBackend] = useState<HealthResponse | null>(null);
  const [quantum, setQuantum] = useState<QuantumHealthResponse | null>(null);
  const [backendError, setBackendError] = useState(false);
  const [quantumError, setQuantumError] = useState(false);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => {
        if (!response.ok) throw new Error("Backend unavailable");
        return response.json() as Promise<HealthResponse>;
      })
      .then(setBackend)
      .catch(() => setBackendError(true));

    fetch("/api/quantum/health")
      .then((response) => {
        if (!response.ok) throw new Error("Quantum engine unavailable");
        return response.json() as Promise<QuantumHealthResponse>;
      })
      .then(setQuantum)
      .catch(() => setQuantumError(true));
  }, []);

  const statusLabel = (ready: boolean, failed: boolean) =>
    failed ? "UNAVAILABLE" : ready ? "READY" : "CHECKING";

  return (
    <main>
      <p className="eyebrow">Local demonstrator</p>
      <h1>AQSE — Adaptive Quantum Sensor Engine</h1>
      <p className="summary">
        Stato dell’ambiente locale. Nessuna pipeline scientifica viene eseguita
        da questa pagina.
      </p>
      <section aria-live="polite" className="status-panel">
        <StatusRow
          label="Backend"
          value={statusLabel(backend?.status === "ok", backendError)}
          unavailable={backendError}
        />
        <StatusRow
          label="Quantum Engine"
          value={statusLabel(quantum?.status === "ok", quantumError)}
          unavailable={quantumError}
        />
        <StatusRow
          label="Qiskit"
          value={statusLabel(quantum?.qiskit === "ready", quantumError)}
          unavailable={quantumError}
        />
        <StatusRow label="Qubits" value={quantum?.qubits ?? "—"} />
        <StatusRow label="Features" value={quantum?.features ?? "—"} />
        <StatusRow
          label="Parameters"
          value={quantum?.trainable_parameters ?? "—"}
        />
        <StatusRow label="Sensor Simulator" value="NOT IMPLEMENTED" muted />
        <StatusRow label="AFSE Embedding" value="NOT IMPLEMENTED" muted />
      </section>
    </main>
  );
}

type StatusRowProps = {
  label: string;
  value: string | number;
  unavailable?: boolean;
  muted?: boolean;
};

function StatusRow({ label, value, unavailable = false, muted = false }: StatusRowProps) {
  const className = unavailable ? "value error" : muted ? "value muted" : "value";

  return (
    <div className="status-row">
      <span>{label}</span>
      <strong className={className}>{value}</strong>
    </div>
  );
}

export default App;
