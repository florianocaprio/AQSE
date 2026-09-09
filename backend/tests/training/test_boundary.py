from __future__ import annotations

from pathlib import Path


def test_1d1_has_no_training_quantum_or_neural_execution_path() -> None:
    training_root = Path("app/training")
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(training_root.glob("*.py"))
    ).lower()
    forbidden_imports = (
        "import qiskit",
        "from qiskit",
        "import torch",
        "from torch",
        "import tensorflow",
        "from tensorflow",
    )
    assert not any(item in source for item in forbidden_imports)
    assert "fit_transform(" not in source


def test_1d1_is_offline_and_registers_no_application_router() -> None:
    application = Path("app/main.py").read_text(encoding="utf-8")
    assert "app.training" not in application
    assert "/training" not in application
    assert "/datasets" not in application
