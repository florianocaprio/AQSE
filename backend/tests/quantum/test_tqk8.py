"""Numerical tests. Qiskit cross-checks run only if Qiskit is installed."""
from __future__ import annotations
import importlib.util
import json
import unittest
import numpy as np
from tqk8 import StateEngine, alignment_loss, fit_qng, loss_grad_metric
from sampler_qng import ExactOverlap, circuit_budget, sampled_quantities

class TestTQK8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = StateEngine("numpy")
        rng = np.random.default_rng(23)
        cls.X = rng.uniform(-1.2, 1.2, (6, 8))
        cls.y = np.array([-1, 1, -1, 1, -1, 1])
        cls.theta = rng.uniform(-0.8, 0.8, 16)

    def test_kernel(self):
        K = self.engine.gram(self.X, self.theta)
        np.testing.assert_allclose(K, K.T, atol=1e-13)
        np.testing.assert_allclose(K.diagonal(), 1, atol=1e-13)
        self.assertGreater(np.linalg.eigvalsh(K)[0], -1e-12)

    def test_each_parameter_can_change_kernel(self):
        eps = 1e-5
        norms = []
        for r in range(16):
            d = np.eye(16)[r] * eps
            difference = (self.engine.gram(self.X, self.theta + d)
                          - self.engine.gram(self.X, self.theta - d)) / (2 * eps)
            norms.append(np.linalg.norm(difference))
        self.assertGreater(min(norms), 1e-5)

    def test_total_loss_gradient_finite_difference(self):
        _, grad, g, _ = loss_grad_metric(self.engine, self.X, self.y, self.theta)
        eps = 1e-6
        fd = []
        for r in range(16):
            d = np.eye(16)[r] * eps
            lp = alignment_loss(self.engine.gram(self.X, self.theta + d), self.y)[0]
            lm = alignment_loss(self.engine.gram(self.X, self.theta - d), self.y)[0]
            fd.append((lp-lm)/(2*eps))
        np.testing.assert_allclose(grad, fd, atol=2e-8, rtol=2e-6)
        self.assertGreater(np.linalg.eigvalsh(g)[0], -1e-12)

    def test_measurement_formulas_against_state_derivatives(self):
        loss, grad, g, K = loss_grad_metric(self.engine, self.X, self.y, self.theta)
        ex = ExactOverlap(max_circuits=20_000)
        l2, d2, g2, K2, budget = sampled_quantities(
            ex, self.X, self.y, self.theta, X_metric=self.X, metric="full")
        np.testing.assert_allclose(l2, loss, atol=1e-12)
        np.testing.assert_allclose(K2, K, atol=1e-12)
        np.testing.assert_allclose(d2, grad, atol=1e-12)
        np.testing.assert_allclose(g2, g, atol=1e-12)
        self.assertEqual(ex.used_circuits, budget["total"])

    def test_qng_decreases_demo_loss(self):
        theta, rows = fit_qng(self.engine, self.X, self.y, self.theta, steps=5, verbose=False)
        self.assertEqual(len(rows), 5)
        self.assertLess(rows[-1]["loss_after"], rows[0]["loss_before"])
        for row in rows:
            self.assertLessEqual(row["loss_after"], row["loss_before"])
        self.assertEqual(theta.shape, (16,))

    def test_budget_before_execution(self):
        ex = ExactOverlap(max_circuits=1)
        with self.assertRaises(RuntimeError):
            sampled_quantities(ex, self.X, self.y, self.theta)
        self.assertEqual(ex.used_circuits, 0)
        self.assertEqual(circuit_budget(8, 4, "diagonal")["total"], 1884)
        self.assertEqual(circuit_budget(8, 4, "full")["total"], 3804)

    @unittest.skipUnless(importlib.util.find_spec("qiskit"), "Qiskit non installato")
    def test_qiskit_state_and_gate_counts(self):
        from tqk8 import build_vqc, overlap_circuit
        from qiskit.quantum_info import Statevector
        eng = StateEngine("qiskit")
        np.testing.assert_allclose(eng.states(self.X, self.theta),
                                   self.engine.states(self.X, self.theta), atol=1e-12)
        qc, _, _ = build_vqc()
        self.assertEqual(qc.num_qubits, 8)
        self.assertEqual(qc.count_ops()["cz"], 7)
        self.assertEqual(qc.count_ops()["ry"], 16)
        self.assertEqual(qc.count_ops()["rz"], 16)
        pair = overlap_circuit(self.X[0], self.X[1], self.theta, measure=False)
        prob = abs(Statevector.from_instruction(pair).data[0])**2
        self.assertAlmostEqual(prob, self.engine.gram(self.X[:2], self.theta)[0,1], places=12)

    @unittest.skipUnless(importlib.util.find_spec("qiskit"), "Qiskit non installato")
    def test_sampler(self):
        from qiskit.primitives import StatevectorSampler
        from sampler_qng import SamplerOverlap
        ex = SamplerOverlap(StatevectorSampler(seed=42), shots=8192)
        est = ex.probabilities([(self.X[0], self.X[1], self.theta, self.theta)])[0]
        exact = self.engine.gram(self.X[:2], self.theta)[0,1]
        self.assertLess(abs(est-exact), 0.035)

if __name__ == "__main__":
    unittest.main(verbosity=2)
