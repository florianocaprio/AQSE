"""AQSE: VQC -> fidelity kernel -> centered-alignment loss -> QNG, 8 qubit.

Default: Qiskit Statevector. Optional independent NumPy state simulator for tests.
No QPU job is submitted by this file. Synthetic data are only a software demo.
Run: python tqk8.py --engine qiskit --steps 15
Data: python tqk8.py --csv measurements.csv --engine qiskit
CSV columns: f0,f1,...,f7,label (label must be -1 or +1 for the demo SVC).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_QUBITS = 8
N_PARAMS = 16
EDGES = ((0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6))


def vector(a, size: int, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name}: atteso vettore finito di dimensione {size}.")
    return a


def features(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] != N_QUBITS or len(a) == 0:
        raise ValueError("X deve avere forma (numero_campioni, 8).")
    if not np.isfinite(a).all():
        raise ValueError("X contiene NaN o inf: gestire i dati mancanti prima.")
    return a


def build_vqc():
    """Gate-by-gate. Each theta occurs ONCE; each input feature occurs twice."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    x = ParameterVector("x", N_QUBITS)
    theta = ParameterVector("theta", N_PARAMS)
    qc = QuantumCircuit(N_QUBITS, name="AQSE_VQC8")

    # 1. First data upload: |0> -> RY(x).
    for q in range(8):
        qc.ry(x[q], q)
    # 2. Trainable phases alpha_q = theta[q].
    for q in range(8):
        qc.rz(theta[q], q)
    # 3. Connected path: even edges, then odd edges. No ring/SWAP assumed.
    qc.cz(0, 1)
    qc.cz(2, 3)
    qc.cz(4, 5)
    qc.cz(6, 7)
    qc.cz(1, 2)
    qc.cz(3, 4)
    qc.cz(5, 6)
    # 4. Trainable rotations beta_q = theta[8+q].
    for q in range(8):
        qc.ry(theta[8 + q], q)
    # 5. Second data upload. Crucial: not a common trainable terminal unitary.
    for q in range(8):
        qc.rz(x[q], q)
    return qc, x, theta


def bound_vqc(x, theta):
    x = vector(x, 8, "x")
    theta = vector(theta, 16, "theta")
    qc, xp, tp = build_vqc()
    bind = dict(zip(xp, x)) | dict(zip(tp, theta))
    return qc.assign_parameters(bind, inplace=False)


def overlap_circuit(x, z, theta_left, theta_right=None, measure=True):
    """8 qubits, no ancilla: prepare U(x), then uncompute U(z)."""
    from qiskit import QuantumCircuit

    if theta_right is None:
        theta_right = theta_left
    qc = QuantumCircuit(8, name="AQSE_kernel8")
    qc.compose(bound_vqc(x, theta_left), inplace=True)
    qc.compose(bound_vqc(z, theta_right).inverse(), inplace=True)
    if measure:
        qc.measure_all()  # creates classical register 'meas'
    return qc


def _numpy_state(x, theta):
    """Independent 8-qubit simulator; Qiskit's little-endian convention."""
    s = np.zeros(1 << N_QUBITS, dtype=complex)
    s[0] = 1.0
    indices = np.arange(len(s))

    def ry(angle, q):
        view = s.reshape(-1, 2, 1 << q)
        a, b = view[:, 0, :].copy(), view[:, 1, :].copy()
        c, t = np.cos(angle / 2), np.sin(angle / 2)
        view[:, 0, :], view[:, 1, :] = c * a - t * b, t * a + c * b

    def rz(angle, q):
        phases = np.where((indices >> q) & 1,
                          np.exp(0.5j * angle), np.exp(-0.5j * angle))
        s[:] *= phases

    for q in range(8):
        ry(x[q], q)
    for q in range(8):
        rz(theta[q], q)
    for q, r in EDGES:
        mask = (((indices >> q) & 1) & ((indices >> r) & 1)).astype(bool)
        s[mask] *= -1
    for q in range(8):
        ry(theta[8 + q], q)
    for q in range(8):
        rz(x[q], q)
    return s


class StateEngine:
    """Exact pure-state backend; not a physical QPU state reader."""
    def __init__(self, engine="qiskit"):
        if engine not in {"qiskit", "numpy"}:
            raise ValueError("engine deve essere 'qiskit' oppure 'numpy'.")
        self.engine = engine
        if engine == "qiskit":
            try:
                from qiskit.quantum_info import Statevector
                self._statevector = Statevector
                self.qc, self.xp, self.tp = build_vqc()
            except ImportError as exc:
                raise ImportError("Installare Qiskit: pip install 'qiskit>=2,<3'") from exc

    def state(self, x, theta):
        x, theta = vector(x, 8, "x"), vector(theta, 16, "theta")
        if self.engine == "numpy":
            return _numpy_state(x, theta)
        values = dict(zip(self.xp, x)) | dict(zip(self.tp, theta))
        circuit = self.qc.assign_parameters(values, inplace=False)
        return np.asarray(self._statevector.from_instruction(circuit).data)

    def states(self, X, theta):
        return np.stack([self.state(x, theta) for x in features(X)])

    def gram(self, X, theta, Z=None):
        S = self.states(X, theta)
        T = S if Z is None else self.states(Z, theta)
        return np.abs(S.conj() @ T.T) ** 2

    def differential(self, X, theta):
        """Exact wavefunction derivatives, NOT the probability shift prefactor."""
        X, theta = features(X), vector(theta, 16, "theta")
        S = self.states(X, theta)
        D = np.empty((len(X), N_PARAMS, 1 << N_QUBITS), complex)
        for r in range(N_PARAMS):
            shift = np.zeros(N_PARAMS)
            shift[r] = np.pi / 2
            # Valid because each parameter appears exactly once in RY or RZ.
            D[:, r, :] = (self.states(X, theta + shift)
                          - self.states(X, theta - shift)) / (2 * np.sqrt(2))
        return S, D


@dataclass
class AngleScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X):
        X = features(X)
        std = X.std(axis=0)
        return cls(X.mean(axis=0), np.where(std > 1e-12, std, 1.0))

    def transform(self, X):
        # Smooth bounded map; fit mean/std ONLY on training data.
        z = (features(X) - self.mean) / self.scale
        return (np.pi / 2) * np.tanh(z / 2)


def center(K):
    return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()


def alignment_loss(K, y):
    """Returns L=1-centered alignment and dL/dK (classical chain rule)."""
    K, y = np.asarray(K, float), np.asarray(y, float)
    if y.ndim != 1 or K.shape != (len(y), len(y)) or len(y) < 2:
        raise ValueError("K quadrata e y coerente, con almeno 2 campioni.")
    if not np.isfinite(K).all() or not np.isfinite(y).all():
        raise ValueError("K/y non finiti.")
    yc = y - y.mean()
    T, C = np.outer(yc, yc), center(K)
    nT, nC = np.linalg.norm(T), np.linalg.norm(C)
    if nT < 1e-12 or nC < 1e-12:
        raise ValueError("Alignment non definito: label costanti o kernel collassato.")
    A = np.sum(C * T) / (nC * nT)
    G = A * C / nC**2 - T / (nC * nT)
    return float(1 - A), center(G)


def loss_grad_metric(engine, X, y, theta):
    """One forward/differential pass: K, loss, total gradient, mean FS metric."""
    S, D = engine.differential(X, theta)
    O = S.conj() @ S.T
    K = np.abs(O) ** 2
    loss, dL_dK = alignment_loss(K, y)
    grad = np.empty(N_PARAMS)
    for r in range(N_PARAMS):
        dS = D[:, r, :]
        # BOTH occurrences of theta: ket and bra.
        dO = dS.conj() @ S.T + S.conj() @ dS.T
        dK = 2 * np.real(O.conj() * dO)
        grad[r] = np.sum(dL_dK * dK)

    g = np.zeros((N_PARAMS, N_PARAMS))
    for s, d in zip(S, D):
        v = d.conj() @ s
        qgt = d.conj() @ d.T - np.outer(v, v.conj())
        g += qgt.real
    g /= len(S)
    g = (g + g.T) / 2
    return loss, grad, g, K


def fit_qng(engine, X, y, theta0, steps=15, learning_rate=0.2,
            damping=1e-3, max_step=0.4, verbose=True):
    """Damped QNG with exact-state Armijo backtracking.

    Geometry = empirical mean Fubini-Study metric over training inputs.
    This is a choice of state-space preconditioner, not a claim that it is
    the unique natural metric of the kernel matrix or SVM output.
    """
    if damping <= 0 or learning_rate <= 0 or max_step <= 0 or steps < 0:
        raise ValueError("steps >=0; damping, learning_rate, max_step >0.")
    X = features(X)
    theta = vector(theta0, N_PARAMS, "theta0").copy()
    history = []
    for epoch in range(steps):
        loss, grad, g, _ = loss_grad_metric(engine, X, y, theta)
        if np.linalg.norm(grad) < 1e-10:
            break
        direction = np.linalg.solve(g + damping * np.eye(N_PARAMS), grad)
        delta = learning_rate * direction
        delta *= min(1.0, max_step / max(np.linalg.norm(delta), 1e-15))
        accepted = False
        for _ in range(12):
            candidate = theta - delta
            new_loss, _ = alignment_loss(engine.gram(X, candidate), y)
            if new_loss <= loss - 1e-4 * float(grad @ delta):
                accepted = True
                break
            delta *= 0.5
        if not accepted:
            if verbose:
                print("Arresto: nessun passo accettato dalla line search.")
            break
        theta = candidate
        row = {"epoch": epoch, "loss_before": loss, "loss_after": new_loss,
               "gradient_norm": float(np.linalg.norm(grad)),
               "step_norm": float(np.linalg.norm(delta)),
               "metric_min_eigenvalue": float(np.linalg.eigvalsh(g)[0])}
        history.append(row)
        if verbose:
            print(f"{epoch:02d}  loss {loss:.6f} -> {new_loss:.6f}  "
                  f"|grad|={row['gradient_norm']:.3e}")
    return theta, history


def demo_data(seed=19, n=64):
    """Synthetic nonlinear binary labels; not data from an actual sensor."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    score = X[:, 0] + 0.8 * X[:, 1] * X[:, 2] - 0.5 * X[:, 3]
    y = np.where(score >= 0, 1, -1)
    return X, y


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import balanced_accuracy_score

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["qiskit", "numpy"], default="qiskit")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output_tqk8"))
    args = parser.parse_args()
    if args.csv:
        tab = np.genfromtxt(args.csv, delimiter=",", names=True)
        names = [f"f{i}" for i in range(8)]
        if not set(names + ["label"]).issubset(tab.dtype.names or ()):
            raise ValueError("CSV richiesto: f0,...,f7,label.")
        X = np.column_stack([np.atleast_1d(tab[name]) for name in names])
        y = np.atleast_1d(tab["label"]).astype(float)
        if set(np.unique(y)) != {-1.0, 1.0}:
            raise ValueError("La demo SVC richiede label -1/+1.")
        print("ATTENZIONE: sostituire lo split casuale con split temporale/per sensore.")
    else:
        X, y = demo_data()
        print("DEMO SINTETICA: nessuna conclusione sulla qualità di un sensore reale.")
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, train_size=0.625, random_state=19, stratify=y)
    scaler = AngleScaler.fit(Xtr_raw)
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)
    engine = StateEngine(args.engine)
    theta0 = np.random.default_rng(7).uniform(-0.8, 0.8, N_PARAMS)
    before, _ = alignment_loss(engine.gram(Xtr, theta0), ytr)
    theta, history = fit_qng(engine, Xtr, ytr, theta0, steps=args.steps)
    Ktr, Kte = engine.gram(Xtr, theta), engine.gram(Xte, theta, Xtr)
    clf = SVC(kernel="precomputed", C=1.0).fit(Ktr, ytr)
    pred = clf.predict(Kte)
    baseline = SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xtr, ytr)
    metrics = {"engine": args.engine, "loss_initial": before,
               "loss_final": alignment_loss(Ktr, ytr)[0],
               "test_balanced_accuracy_tqk": float(balanced_accuracy_score(yte, pred)),
               "test_balanced_accuracy_rbf": float(balanced_accuracy_score(
                   yte, baseline.predict(Xte))), "n_train": len(Xtr), "n_test": len(Xte)}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "trained_tqk8.npz", theta=theta, theta0=theta0,
             mean=scaler.mean, scale=scaler.scale, X_train=Xtr, y_train=ytr,
             X_test=Xte, y_test=yte, K_train=Ktr, K_test=Kte)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if args.engine == "qiskit":
        (args.output / "vqc_circuit.txt").write_text(str(engine.qc.draw("text", fold=140)),
                                                     encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
