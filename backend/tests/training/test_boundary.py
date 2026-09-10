from __future__ import annotations

from pathlib import Path


def test_training_layer_has_no_direct_qiskit_or_neural_dependency() -> None:
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


def test_training_is_explicit_and_never_runs_from_application_startup() -> None:
    application = Path("app/main.py").read_text(encoding="utf-8")
    training_api = Path("app/api/training.py").read_text(encoding="utf-8")
    assert "training_router" in application
    assert "@router.post" in training_api
    assert "@router.get" in training_api
    assert "startup" not in training_api.lower()
    assert "/datasets" not in application
