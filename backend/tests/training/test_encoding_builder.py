from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.training.storage import LEDGER_NAME, write_dataset
from scripts import build_training_encoding


def test_opaque_seal_check_never_loads_numpy_or_changes_ledger(
    tmp_path: Path,
    small_development_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, manifest, _ = write_dataset(small_development_build, root=tmp_path)
    monkeypatch.setattr(
        build_training_encoding,
        "CANONICAL_DEVELOPMENT_DATASET_ID",
        manifest.dataset_id,
    )
    monkeypatch.setattr(
        build_training_encoding,
        "CANONICAL_DEVELOPMENT_DIGEST",
        manifest.scientific_digest,
    )
    monkeypatch.setattr(
        build_training_encoding,
        "CANONICAL_FEATURE_PROFILE_FINGERPRINT",
        manifest.feature_profile_fingerprint,
    )
    ledger_before = (path / LEDGER_NAME).read_bytes()

    def forbidden_numpy_load(*args, **kwargs):
        raise AssertionError("opaque seal verification must not load NumPy payloads")

    monkeypatch.setattr(np, "load", forbidden_numpy_load)
    observed_manifest, ledger_sha256 = build_training_encoding._sealed_snapshot(path)

    assert observed_manifest == manifest
    assert len(ledger_sha256) == 64
    assert (path / LEDGER_NAME).read_bytes() == ledger_before


def test_1d2_builder_rejects_test_open_environment(
    tmp_path: Path,
    small_development_build,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _, _ = write_dataset(small_development_build, root=tmp_path)
    monkeypatch.setenv("AQSE_ALLOW_TEST_OPEN", "1")

    with pytest.raises(PermissionError, match="must remain disabled"):
        build_training_encoding._sealed_snapshot(path)
