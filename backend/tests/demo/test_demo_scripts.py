from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import demo_acceptance, demo_soak


def _analysis_result(sensor_id: str, feature_zero: float) -> dict[str, Any]:
    return {
        "sensor_id": sensor_id,
        "window_start_s": 8.0,
        "window_end_exclusive_s": 12.0,
        "context_mode": "network",
        "peer_sensor_ids": ["S2", "S3", "S4"],
        "feature_values": [feature_zero, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
        "feature_valid": True,
        "quality_flags": [],
        "encoded_angles": [0.1] * 8,
        "afse_vector": [0.2, 0.3],
        "afse_reference_size": 2,
        "class_order": ["NORMAL", "DEVICE_COMPATIBLE"],
        "class_scores": [0.55, 0.45],
        "displayed_class": "NORMAL",
        "observable_rule_status": "NO_OBSERVED_CHANGE",
        "uncertain": False,
        "score_semantics": "model-score;not-probability-calibrated",
    }


class _BundleClient:
    def __init__(self, pointer: dict[str, Any]) -> None:
        self.pointer = pointer
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        expected: tuple[int, ...] = (200,),
        **_kwargs: Any,
    ) -> demo_acceptance.HttpResult:
        self.calls.append((method, path, payload))
        assert path == "/api/demo/bundles/apply"
        assert payload is not None
        if payload["local_bundle_id"] != self.pointer["local_bundle_id"]:
            assert expected == (409,)
            return demo_acceptance.HttpResult(
                status=409,
                body={
                    "detail": "requested bundle pair differs from its immutable selection freeze"
                },
                elapsed_ms=0.1,
            )
        assert expected == (200,)
        return demo_acceptance.HttpResult(
            status=200,
            body=self.pointer,
            elapsed_ms=0.1,
        )


def test_acceptance_report_is_atomic_and_outside_repository(tmp_path: Path) -> None:
    report_path = demo_acceptance._safe_report_path(tmp_path, "acceptance-fixture")
    demo_acceptance._write_report(report_path, {"result": "passed"})

    assert json.loads(report_path.read_text(encoding="utf-8")) == {"result": "passed"}
    assert not tuple(tmp_path.glob(".*.tmp"))
    with pytest.raises(demo_acceptance.AcceptanceError, match="outside the Git tree"):
        demo_acceptance._safe_report_path(
            demo_acceptance.REPOSITORY_ROOT / "acceptance-output",
            "invalid",
        )


def test_soak_rss_summary_reports_growth_and_real_time_slope() -> None:
    summary = demo_soak.summarize_rss([(0.0, 100), (1_800.0, 150), (3_600.0, 200)])

    assert summary == {
        "initial_bytes": 100,
        "final_bytes": 200,
        "minimum_bytes": 100,
        "maximum_bytes": 200,
        "growth_bytes": 100,
        "ols_slope_bytes_per_hour": 100.0,
    }


def test_operational_make_targets_use_explicit_preparation_and_live_services() -> None:
    candidates = (
        Path(__file__).resolve().parents[3] / "Makefile",
        Path("/workspace/Makefile"),
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        pytest.skip("Makefile is not mounted in this backend-only test environment")
    makefile = path.read_text(encoding="utf-8")

    assert "prepare-demo:" in makefile
    assert "python -m scripts.prepare_demo" in makefile
    assert "demo:" in makefile
    assert "docker compose up -d --wait" in makefile
    assert "acceptance:" in makefile
    assert "python -m scripts.demo_acceptance" in makefile
    assert "python -m scripts.demo_soak" in makefile
    assert "SOAK_DURATION_SECONDS:-600" in makefile


def test_operational_scripts_do_not_request_simulator_truth() -> None:
    acceptance_source = Path(demo_acceptance.__file__).read_text(encoding="utf-8")
    soak_source = Path(demo_soak.__file__).read_text(encoding="utf-8")

    assert "/truth/" not in acceptance_source
    assert "/truth/" not in soak_source


def test_acceptance_covers_required_observed_perturbations_without_causal_claims() -> None:
    cases = demo_acceptance._perturbation_cases()
    by_id = {case["case_id"]: case for case in cases}
    kinds = {event["kind"] for case in cases for event in case["events"]}

    assert set(by_id) == {
        "common_environment",
        "s3_drift_and_noise",
        "shared_instrument_offset",
        "mixed_shared_and_focal",
    }
    assert kinds == {
        "world_field_offset",
        "node_drift",
        "node_noise_burst",
        "shared_instrument_offset",
        "node_bias",
    }
    s3_events = by_id["s3_drift_and_noise"]["events"]
    assert {tuple(event["target_sensor_ids"]) for event in s3_events} == {("S3",)}
    shared = by_id["shared_instrument_offset"]["events"][0]
    assert shared["target_sensor_ids"] == ["S1", "S2", "S3", "S4"]
    assert all(
        any(
            word in case["interpretation"].lower()
            for word in ("not proof", "not establish", "compatible", "ambiguous")
        )
        for case in cases
    )


def test_invalid_acceptance_covers_dropout_stuck_and_observed_clipping() -> None:
    cases = demo_acceptance._invalid_input_cases()

    assert [(kind, sensor_id, flag) for kind, sensor_id, flag, _ in cases] == [
        ("dropout", "S1", "signal_absent"),
        ("stuck", "S2", "stuck"),
        ("node_bias", "S3", "clipped"),
    ]
    assert cases[-1][3]["field_offset_T"] == [0.002, 0.0, 0.0]


def test_observed_change_records_real_outputs_but_not_event_or_truth_fields() -> None:
    before = {
        "latest_results": [
            _analysis_result("S1", 1.0),
            _analysis_result("S2", 2.0),
        ]
    }
    after = {
        "latest_results": [
            _analysis_result("S1", 3.0),
            _analysis_result("S2", 2.5),
        ]
    }

    evidence = demo_acceptance._observed_change(before, after)

    assert evidence["maximum_absolute_feature_delta"] == 2.0
    assert evidence["per_sensor_maximum_absolute_feature_delta"] == {
        "S1": 2.0,
        "S2": 0.5,
    }
    assert {item["sensor_id"] for item in evidence["results"]} == {"S1", "S2"}
    assert all("event_id" not in item and "truth_label" not in item for item in evidence["results"])


def test_observation_evidence_rejects_control_or_truth_leakage() -> None:
    result = _analysis_result("S1", 1.0)
    result["event_kind"] = "node_drift"

    with pytest.raises(demo_acceptance.AcceptanceError, match="leaked simulator control"):
        demo_acceptance._result_evidence(result)


def test_bundle_acceptance_retries_idempotently_and_rejects_mismatch() -> None:
    pointer = {
        "application_id": "aqse-demo-application-1111111111111111",
        "local_bundle_id": "aqse-demo-bundle-2222222222222222",
        "network_bundle_id": "aqse-demo-bundle-3333333333333333",
        "selection_freeze_id": "aqse-demo-freeze-4444444444444444",
    }
    client = _BundleClient(pointer)
    registry = {
        "active": pointer,
        "selection_freeze_id": pointer["selection_freeze_id"],
    }

    evidence = demo_acceptance.run_bundle_acceptance(client, registry)  # type: ignore[arg-type]

    assert evidence["compatible_retry_idempotent"] is True
    assert evidence["compatible_retry_application_id"] == pointer["application_id"]
    assert evidence["mismatched_pair_status"] == 409
    assert len(client.calls) == 3


def test_acceptance_marks_process_restart_as_an_external_orchestrator_check() -> None:
    source = Path(demo_acceptance.__file__).read_text(encoding="utf-8")

    assert '"process_restart_check": "external-orchestrator-required"' in source
    assert "must not terminate the backend or Docker" in source
