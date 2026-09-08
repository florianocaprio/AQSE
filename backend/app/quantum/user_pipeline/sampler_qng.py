"""Shot-based building blocks for the SAME TQK8 circuit, using eight qubits.

No QPU is contacted unless a user supplies and runs a hardware Sampler.
Gradients shift the two parameter occurrences SEPARATELY.
Metric from state overlaps: exact for ideal pure states; under device noise it
is only a regularized geometry proxy, NOT the mixed-state quantum Fisher matrix.
Full-metric measurement is expensive; diagonal mode is the default.
"""
from __future__ import annotations

import numpy as np

if __package__:
    from .tqk8 import (N_PARAMS, StateEngine, alignment_loss, features,
                       overlap_circuit, vector)
else:
    from tqk8 import (N_PARAMS, StateEngine, alignment_loss, features,
                      overlap_circuit, vector)


def circuit_budget(n_samples: int, n_metric_samples: int, metric="diagonal"):
    if n_samples < 2 or n_metric_samples < 1 or metric not in {"diagonal", "full"}:
        raise ValueError("Richiesti batch >=2, metric batch >=1 e metrica valida.")
    pairs = n_samples * (n_samples - 1) // 2
    metric_per_sample = (N_PARAMS if metric == "diagonal" else
                         N_PARAMS + 2 * N_PARAMS * (N_PARAMS - 1))
    return {"kernel_and_gradient": pairs * (1 + 4 * N_PARAMS),
            "metric": n_metric_samples * metric_per_sample,
            "total": pairs * (1 + 4 * N_PARAMS) + n_metric_samples * metric_per_sample}


class ExactOverlap:
    """Numerical oracle for testing measurement formulas; not a QPU."""
    def __init__(self, engine="numpy", max_circuits=100_000):
        self.engine = StateEngine(engine)
        self.max_circuits = max_circuits
        self.used_circuits = 0
        self._cache = {}

    def check_budget(self, count):
        if self.used_circuits + count > self.max_circuits:
            raise RuntimeError("Budget superato prima dell'esecuzione.")

    def _state(self, x, t):
        key = (np.asarray(x, float).tobytes(), np.asarray(t, float).tobytes())
        if key not in self._cache:
            self._cache[key] = self.engine.state(x, t)
        return self._cache[key]

    def probabilities(self, specs):
        self.check_budget(len(specs))
        values = [abs(np.vdot(self._state(z, b), self._state(x, a)))**2
                  for x, z, a, b in specs]
        self.used_circuits += len(specs)
        return np.asarray(values, float)


class SamplerOverlap:
    """Adapter for StatevectorSampler or a provider's compatible SamplerV2.

    QPU: provide an ISA pass manager with YOUR verified eight-qubit layout.
    No guessed physical qubits or backend names are supplied here.
    """
    def __init__(self, sampler, pass_manager=None, shots=1024,
                 circuits_per_job=64, max_circuits=2000, allowed_physical_qubits=None):
        if shots < 1 or circuits_per_job < 1 or max_circuits < 1:
            raise ValueError("shots e budget devono essere positivi.")
        self.sampler = sampler
        self.pass_manager = pass_manager
        self.shots = int(shots)
        self.circuits_per_job = int(circuits_per_job)
        self.max_circuits = int(max_circuits)
        self.used_circuits = 0
        self.allowed_physical_qubits = (None if allowed_physical_qubits is None
                                        else set(allowed_physical_qubits))

    def check_budget(self, count):
        if self.used_circuits + count > self.max_circuits:
            raise RuntimeError(
                f"Budget insufficiente: richiesti {count}, residui "
                f"{self.max_circuits-self.used_circuits}. Nessun nuovo job inviato.")

    def probabilities(self, specs):
        self.check_budget(len(specs))
        result_values = []
        for start in range(0, len(specs), self.circuits_per_job):
            chunk = specs[start:start + self.circuits_per_job]
            circuits = [overlap_circuit(x, z, a, b) for x, z, a, b in chunk]
            if self.pass_manager is not None:
                circuits = self.pass_manager.run(circuits)
            if self.allowed_physical_qubits is not None:
                for circuit in circuits:
                    for instruction in circuit.data:
                        if instruction.operation.name == "barrier":
                            continue
                        ids = {circuit.find_bit(q).index for q in instruction.qubits}
                        if not ids.issubset(self.allowed_physical_qubits):
                            raise RuntimeError("Il transpiler usa qubit fuori dal sottoinsieme scelto.")
            # Reserve first: a failed external request may still consume resources.
            self.used_circuits += len(circuits)
            job_result = self.sampler.run(circuits, shots=self.shots).result()
            for pub_result in job_result:
                counts = pub_result.data.meas.get_counts()
                total = sum(counts.values())
                if total <= 0:
                    raise RuntimeError("Sampler senza conteggi.")
                result_values.append(counts.get("00000000", 0) / total)
        return np.asarray(result_values, float)


def sampled_quantities(executor, X, y, theta, X_metric=None, metric="diagonal"):
    """Loss, gradient and FS-geometry estimate from overlap probabilities."""
    X, theta = features(X), vector(theta, N_PARAMS, "theta")
    # Validate labels before any potentially billable execution.
    alignment_loss(np.eye(len(X)), y)
    if X_metric is None:
        ids = np.linspace(0, len(X) - 1, min(4, len(X)), dtype=int)
        X_metric = X[ids]
    X_metric = features(X_metric)
    budget = circuit_budget(len(X), len(X_metric), metric)
    executor.check_budget(budget["total"])

    pairs = [(i, j) for i in range(len(X)) for j in range(i + 1, len(X))]
    K = np.eye(len(X))
    # Analytic ideal diagonal K(x,x)=1; measured diagonals are a separate diagnostic.
    probs = executor.probabilities([(X[i], X[j], theta, theta) for i, j in pairs])
    for (i, j), val in zip(pairs, probs):
        K[i, j] = K[j, i] = val
    loss, G = alignment_loss(K, y)
    grad = np.zeros(N_PARAMS)
    for r in range(N_PARAMS):
        d = np.zeros(N_PARAMS)
        d[r] = np.pi / 2
        specs = []
        for i, j in pairs:
            specs += [(X[i], X[j], theta + d, theta),
                      (X[i], X[j], theta - d, theta),
                      (X[i], X[j], theta, theta + d),
                      (X[i], X[j], theta, theta - d)]
        p = executor.probabilities(specs).reshape(-1, 4)
        dK = (p[:, 0] - p[:, 1] + p[:, 2] - p[:, 3]) / 2
        grad[r] = sum(2 * G[i, j] * val for (i, j), val in zip(pairs, dK))

    g = np.zeros((N_PARAMS, N_PARAMS))
    for x in X_metric:
        specs = []
        for a in range(N_PARAMS):
            d = np.zeros(N_PARAMS)
            d[a] = np.pi
            specs.append((x, x, theta, theta + d))
        f_pi = executor.probabilities(specs)
        g[np.diag_indices(N_PARAMS)] += (1 - f_pi) / 4
        if metric == "full":
            indices, specs = [], []
            for a in range(N_PARAMS):
                for b in range(a + 1, N_PARAMS):
                    da, db = np.zeros(N_PARAMS), np.zeros(N_PARAMS)
                    da[a], db[b] = np.pi / 2, np.pi / 2
                    indices.append((a, b))
                    specs += [(x, x, theta, theta + da + db),
                              (x, x, theta, theta + da - db),
                              (x, x, theta, theta - da + db),
                              (x, x, theta, theta - da - db)]
            p = executor.probabilities(specs).reshape(-1, 4)
            values = -(p[:, 0] - p[:, 1] - p[:, 2] + p[:, 3]) / 8
            for (a, b), val in zip(indices, values):
                g[a, b] += val
                g[b, a] += val
    g /= len(X_metric)
    return loss, grad, g, K, budget


def sampled_qng_step(executor, X, y, theta, X_metric=None, metric="diagonal",
                     learning_rate=0.03, damping=0.02, max_step=0.1):
    """One stochastic step; no false guarantee of monotonic loss with noisy shots."""
    if min(learning_rate, damping, max_step) <= 0:
        raise ValueError("learning_rate, damping, max_step devono essere positivi.")
    loss, grad, g, K, budget = sampled_quantities(executor, X, y, theta, X_metric, metric)
    w, V = np.linalg.eigh((g + g.T) / 2)
    # Mitigate finite-shot indefiniteness; this is a regularized geometry proxy.
    g_psd = (V * np.maximum(w, 0)) @ V.T
    step = learning_rate * np.linalg.solve(g_psd + damping * np.eye(N_PARAMS), grad)
    step *= min(1.0, max_step / max(np.linalg.norm(step), 1e-15))
    return np.asarray(theta) - step, {
        "loss_before": loss, "gradient_norm": float(np.linalg.norm(grad)),
        "step_norm": float(np.linalg.norm(step)), "budget": budget,
        "raw_metric_min_eigenvalue": float(w[0]), "metric_mode": metric,
        "kernel_min_eigenvalue": float(np.linalg.eigvalsh(K)[0])}
