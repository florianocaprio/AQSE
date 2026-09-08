import { useEffect, useState } from "react";

type HealthResponse = {
  status: string;
  service: string;
};

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => {
        if (!response.ok) throw new Error("Backend unavailable");
        return response.json() as Promise<HealthResponse>;
      })
      .then(setHealth)
      .catch(() => setError(true));
  }, []);

  const status = error
    ? "Backend non raggiungibile"
    : health?.status === "ok"
      ? `${health.service}: connesso`
      : "Connessione al backend…";

  return (
    <main>
      <p className="eyebrow">Local demonstrator</p>
      <h1>AQSE — Adaptive Quantum Sensor Engine</h1>
      <p className="summary">
        Fondazione locale per una pipeline ibrida classica/quantistica dedicata
        ai sensori quantistici.
      </p>
      <section aria-live="polite" className={error ? "status error" : "status"}>
        <span aria-hidden="true" />
        {status}
      </section>
    </main>
  );
}

export default App;

