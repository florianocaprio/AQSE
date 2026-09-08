from __future__ import annotations

from pathlib import Path


def test_backend_docker_healthcheck_uses_only_lightweight_service_health() -> None:
    compose_path = next(
        path
        for path in (
            Path(__file__).resolve().parents[2] / "docker-compose.yml",
            Path("/workspace/docker-compose.yml"),
        )
        if path.is_file()
    )
    compose = compose_path.read_text(encoding="utf-8")
    backend_service = compose.split("  frontend:", maxsplit=1)[0]
    healthcheck = backend_service.split("    healthcheck:", maxsplit=1)[1]

    assert "http://localhost:8000/api/health" in healthcheck
    assert "/api/quantum/health" not in healthcheck
    assert "/api/quantum/diagnostics" not in healthcheck
    assert "timeout: 3s" in healthcheck
