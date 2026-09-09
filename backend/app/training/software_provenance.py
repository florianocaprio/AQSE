from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

from app.training.models import SoftwareProvenance

SOURCE_FILES = (
    "app/features/models.py",
    "app/features/provenance.py",
    "app/features/windowed.py",
    "app/network/models.py",
    "app/network/simulation.py",
    "app/preprocessing/features.py",
    "app/training/canonical.py",
    "app/training/generation.py",
    "app/training/models.py",
    "app/training/software_provenance.py",
    "app/training/splits.py",
    "app/training/storage.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_software_provenance() -> SoftwareProvenance:
    backend_root = Path(__file__).resolve().parents[2]
    repository_root = backend_root.parent
    source_hashes = {
        relative: _sha256(backend_root / relative)
        for relative in SOURCE_FILES
        if (backend_root / relative).is_file()
    }
    base_sha = "unrecorded"
    dirty: bool | None = None
    try:
        base_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "backend/app"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    runtime_versions = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "pydantic": importlib.metadata.version("pydantic"),
    }
    return SoftwareProvenance(
        algorithm_ids=(
            "aqse.network-simulator.v1",
            "aqse.magnetometer.harmonic.v1@1.0.0",
            "aqse.dataset-archive.v2",
        ),
        unit_conversions={"magnetic_field": "nT_to_T:1e-9"},
        repository_base_sha=base_sha,
        repository_dirty=dirty,
        source_hashes=source_hashes,
        runtime_versions=runtime_versions,
    )


def validate_software_provenance(
    value: SoftwareProvenance, *, verify_effective_sources: bool = False
) -> None:
    if not value.source_hashes:
        raise ValueError("software provenance must contain effective source hashes")
    for relative, digest in value.source_hashes.items():
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("software provenance contains an unsafe source path")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("software provenance contains an invalid source digest")
    if value.repository_base_sha != "unrecorded" and (
        len(value.repository_base_sha) != 40
        or any(character not in "0123456789abcdef" for character in value.repository_base_sha)
    ):
        raise ValueError("software provenance repository SHA is invalid")
    required_runtime = {"python", "numpy", "pydantic"}
    if not required_runtime.issubset(value.runtime_versions):
        raise ValueError("software provenance runtime versions are incomplete")
    if verify_effective_sources:
        backend_root = Path(__file__).resolve().parents[2]
        expected_paths = {
            relative for relative in SOURCE_FILES if (backend_root / relative).is_file()
        }
        if set(value.source_hashes) != expected_paths:
            raise ValueError("software provenance source manifest is incomplete")
        for relative in expected_paths:
            if value.source_hashes[relative] != _sha256(backend_root / relative):
                raise ValueError(f"software provenance source hash mismatch: {relative}")
