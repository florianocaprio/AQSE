#!/usr/bin/env python3
"""Exercise the prepared AQSE demonstrator through its public REST API.

The script intentionally consumes observation and analysis endpoints only.  It never
requests simulator truth and never prepares, opens, or evaluates a TEST partition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BACKEND_ROOT.parent if BACKEND_ROOT.name == "backend" else BACKEND_ROOT
DEFAULT_REPORT_ROOT = (
    Path(
        os.environ.get(
            "AQSE_ARTIFACT_ROOT",
            str(REPOSITORY_ROOT.parent / "AQSE-artifacts"),
        )
    )
    / "validation"
)
HISTORICAL_TEST_LEDGER_SHA256 = "e0d4898141d8be07f4a4f1af7582791eae61994ced52bb7b3dba30a7ddda4b70"


class AcceptanceError(RuntimeError):
    """Raised when a required end-to-end observation cannot be demonstrated."""


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: Any
    elapsed_ms: float


@dataclass
class JsonHttpClient:
    base_url: str
    timeout_s: float = 10.0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, object] | None = None,
        expected: tuple[int, ...] = (200,),
    ) -> HttpResult:
        url = f"{self.base_url.rstrip('/')}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, method=method, data=data, headers=headers)
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
                status = response.status
        except HTTPError as exc:
            raw = exc.read()
            status = exc.code
        except (TimeoutError, URLError) as exc:
            raise AcceptanceError(f"{method} {path} failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        try:
            body = json.loads(raw) if raw else None
        except json.JSONDecodeError as exc:
            raise AcceptanceError(f"{method} {path} returned invalid JSON") from exc
        self.calls.append(
            {
                "method": method,
                "path": path,
                "status": status,
                "elapsed_ms": elapsed_ms,
            }
        )
        if status not in expected:
            raise AcceptanceError(
                f"{method} {path} returned HTTP {status}; expected {expected}: {body}"
            )
        return HttpResult(status=status, body=body, elapsed_ms=elapsed_ms)


def _get_text(url: str, *, timeout_s: float) -> HttpResult:
    started = time.perf_counter()
    try:
        with urlopen(Request(url, headers={"Accept": "text/html"}), timeout=timeout_s) as response:
            raw = response.read()
            status = response.status
    except (HTTPError, TimeoutError, URLError) as exc:
        raise AcceptanceError(f"GET {url} failed: {exc}") from exc
    return HttpResult(
        status=status,
        body=raw.decode("utf-8", errors="replace"),
        elapsed_ms=(time.perf_counter() - started) * 1_000.0,
    )


def _wait_for(
    description: str,
    fetch: Callable[[], dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout_s: float,
    interval_s: float = 0.1,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    latest: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        latest = fetch()
        if latest.get("state") == "failed":
            raise AcceptanceError(
                f"{description} failed in the analysis worker: {latest.get('error')}"
            )
        if predicate(latest):
            return latest
        time.sleep(interval_s)
    raise AcceptanceError(f"timed out waiting for {description}; latest={latest}")


def _assert_result_contract(result: dict[str, Any]) -> None:
    if len(result.get("feature_values", ())) != 8:
        raise AcceptanceError("analysis did not expose eight state8 features")
    if not result.get("feature_valid"):
        raise AcceptanceError(
            f"baseline inference unexpectedly abstained: {result.get('quality_flags')}"
        )
    if len(result.get("encoded_angles") or ()) != 8:
        raise AcceptanceError("analysis did not expose eight encoded angles")
    if len(result.get("afse_vector") or ()) != result.get("afse_reference_size"):
        raise AcceptanceError("AFSE output dimension differs from its frozen reference")
    if len(result.get("class_scores") or ()) != len(result.get("class_order") or ()):
        raise AcceptanceError("model score vector differs from its class order")
    if result.get("score_semantics") != "model-score;not-probability-calibrated":
        raise AcceptanceError("model scores are missing their non-calibration disclosure")


def _result_evidence(result: dict[str, Any]) -> dict[str, Any]:
    """Keep only observed/model output; scheduled-event commands are never features."""

    forbidden = {"event_id", "event_kind", "injected_cause", "truth_label"}
    leaked = forbidden.intersection(result)
    if leaked:
        raise AcceptanceError(f"analysis result leaked simulator control fields: {sorted(leaked)}")
    _assert_result_contract(result)
    return {
        "sensor_id": result["sensor_id"],
        "window_start_s": result["window_start_s"],
        "window_end_exclusive_s": result["window_end_exclusive_s"],
        "context_mode": result["context_mode"],
        "peer_sensor_ids": result["peer_sensor_ids"],
        "feature_values": result["feature_values"],
        "afse_vector": result["afse_vector"],
        "displayed_class": result["displayed_class"],
        "class_scores": result["class_scores"],
        "observable_rule_status": result.get("observable_rule_status"),
        "uncertain": result["uncertain"],
        "score_semantics": result["score_semantics"],
    }


def _observed_change(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    before_by_sensor = {item["sensor_id"]: item for item in before.get("latest_results", ())}
    after_by_sensor = {item["sensor_id"]: item for item in after.get("latest_results", ())}
    if before_by_sensor.keys() != after_by_sensor.keys() or not after_by_sensor:
        raise AcceptanceError("observed comparison has incompatible sensor sets")
    evidence = [
        _result_evidence(after_by_sensor[sensor_id]) for sensor_id in sorted(after_by_sensor)
    ]
    maximum_delta = 0.0
    per_sensor_delta: dict[str, float] = {}
    for sensor_id in sorted(after_by_sensor):
        left = before_by_sensor[sensor_id]["feature_values"]
        right = after_by_sensor[sensor_id]["feature_values"]
        if len(left) != 8 or len(right) != 8:
            raise AcceptanceError("observed comparison requires complete state8 features")
        delta = max(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))
        per_sensor_delta[sensor_id] = delta
        maximum_delta = max(maximum_delta, delta)
    if maximum_delta <= 1.0e-12:
        raise AcceptanceError("configured perturbation produced no observable feature change")
    return {
        "maximum_absolute_feature_delta": maximum_delta,
        "per_sensor_maximum_absolute_feature_delta": per_sensor_delta,
        "results": evidence,
    }


def _create_configuration(
    client: JsonHttpClient,
    *,
    node_count: int,
    seed: int,
    time_scale: float,
) -> dict[str, Any]:
    configuration = client.request_json(
        "GET",
        "/api/network/defaults",
        query={"node_count": node_count},
    ).body
    configuration.update(
        {
            "session_name": f"AQSE acceptance {node_count}-node observation-only run",
            "random_seed": seed,
            "time_scale": time_scale,
            "buffer_duration_s": 30.0,
        }
    )
    return configuration


def _cleanup_session(client: JsonHttpClient, session_id: str) -> None:
    client.request_json(
        "POST",
        f"/api/demo/analysis/{session_id}/stop",
        expected=(200, 404),
    )
    client.request_json(
        "POST",
        f"/api/network/sessions/{session_id}/stop",
        expected=(200, 404),
    )
    client.request_json(
        "DELETE",
        f"/api/network/sessions/{session_id}",
        expected=(204, 404),
    )


def run_node_acceptance(
    client: JsonHttpClient,
    *,
    node_count: int,
    timeout_s: float,
) -> dict[str, Any]:
    configuration = _create_configuration(
        client,
        node_count=node_count,
        seed=8_100 + node_count,
        time_scale=8.0,
    )
    created = client.request_json(
        "POST",
        "/api/network/sessions",
        payload=configuration,
        expected=(201,),
    ).body
    session_id = created["status"]["session_id"]
    try:
        client.request_json(
            "POST",
            f"/api/demo/analysis/{session_id}/start",
            payload={"acquire_reference": True},
        )
        client.request_json("POST", f"/api/network/sessions/{session_id}/start")

        def fetch() -> dict[str, Any]:
            return client.request_json("GET", f"/api/demo/analysis/{session_id}").body

        analysis = _wait_for(
            f"{node_count}-node inference",
            fetch,
            lambda value: (
                value.get("completed_window_count", 0) >= 1
                and len(value.get("latest_results", ())) == node_count
            ),
            timeout_s=timeout_s,
        )
        for result in analysis["latest_results"]:
            _assert_result_contract(result)
        expected_mode = {
            1: "local",
            2: "local_two_node_ambiguous",
        }.get(node_count, "network")
        actual_modes = sorted({result["context_mode"] for result in analysis["latest_results"]})
        if actual_modes != [expected_mode]:
            raise AcceptanceError(
                f"{node_count}-node routing was {actual_modes}, expected {expected_mode}"
            )
        expected_task = "aqse.local-change.v1" if node_count <= 2 else "aqse.network-pattern.v1"
        if {item["task_id"] for item in analysis["latest_results"]} != {expected_task}:
            raise AcceptanceError(f"{node_count}-node task routing is incompatible")
        return {
            "node_count": node_count,
            "session_id": session_id,
            "context_mode": expected_mode,
            "task_id": expected_task,
            "reference_id": analysis["reference_id"],
            "application_id": analysis["application_id"],
            "completed_window_count": analysis["completed_window_count"],
            "skipped_window_count": analysis["skipped_window_count"],
            "latency_p50_ms": analysis["latency_p50_ms"],
            "latency_p95_ms": analysis["latency_p95_ms"],
            "result_count": len(analysis["latest_results"]),
            "displayed_classes": [item["displayed_class"] for item in analysis["latest_results"]],
        }
    finally:
        _cleanup_session(client, session_id)


def _step(client: JsonHttpClient, session_id: str, frames: int) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    remaining = frames
    while remaining:
        batch = min(remaining, 500)
        latest = client.request_json(
            "POST",
            f"/api/network/sessions/{session_id}/step",
            payload={"frames": batch},
        ).body
        remaining -= batch
    return latest


def _invalid_input_cases() -> tuple[tuple[str, str, str, dict[str, Any]], ...]:
    return (
        ("dropout", "S1", "signal_absent", {}),
        ("stuck", "S2", "stuck", {}),
        (
            "node_bias",
            "S3",
            "clipped",
            {"field_offset_T": [0.002, 0.0, 0.0]},
        ),
    )


def run_invalid_input_acceptance(
    client: JsonHttpClient,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    configuration = _create_configuration(
        client,
        node_count=4,
        seed=8_404,
        time_scale=1.0,
    )
    created = client.request_json(
        "POST",
        "/api/network/sessions",
        payload=configuration,
        expected=(201,),
    ).body
    session_id = created["status"]["session_id"]
    event_evidence: list[dict[str, Any]] = []
    try:
        client.request_json(
            "POST",
            f"/api/demo/analysis/{session_id}/start",
            payload={"acquire_reference": True},
        )
        _step(client, session_id, 1_200)

        def fetch() -> dict[str, Any]:
            return client.request_json("GET", f"/api/demo/analysis/{session_id}").body

        analysis = _wait_for(
            "stepped baseline inference",
            fetch,
            lambda value: value.get("completed_window_count", 0) >= 1,
            timeout_s=timeout_s,
        )
        if any(not item["feature_valid"] for item in analysis["latest_results"]):
            raise AcceptanceError("invalid-input fixture baseline unexpectedly abstained")

        for index, (kind, sensor_id, expected_flag, extra) in enumerate(
            _invalid_input_cases(), start=1
        ):
            session = client.request_json("GET", f"/api/network/sessions/{session_id}").body
            start_time_s = round(session["status"]["sim_time_s"] + 0.01, 8)
            before = analysis["completed_window_count"]
            response = client.request_json(
                "POST",
                f"/api/network/sessions/{session_id}/events",
                payload={
                    "event_id": f"acceptance-{kind}-{index}",
                    "kind": kind,
                    "start_time_s": start_time_s,
                    "duration_s": 4.0,
                    "target_sensor_ids": [sensor_id],
                    **extra,
                },
                expected=(201,),
            ).body
            _step(client, session_id, 400)
            analysis = _wait_for(
                f"{kind} quality abstention",
                fetch,
                lambda value: (
                    value.get("completed_window_count", 0) > before
                    and any(
                        item["sensor_id"] == sensor_id
                        and not item["feature_valid"]
                        and expected_flag in item["quality_flags"]
                        and item["displayed_class"] == "ABSTAIN"
                        for item in value.get("latest_results", ())
                    )
                ),
                timeout_s=timeout_s,
            )
            evidence = next(
                item for item in analysis["latest_results"] if item["sensor_id"] == sensor_id
            )
            valid_peer_results = [
                item
                for item in analysis["latest_results"]
                if item["sensor_id"] != sensor_id
                and item["feature_valid"]
                and item["context_mode"] == "network"
            ]
            if len(valid_peer_results) < 3:
                raise AcceptanceError(
                    f"{kind} prevented the remaining three valid peers from using "
                    "their observed network context"
                )
            event_evidence.append(
                {
                    "kind": kind,
                    "sensor_id": sensor_id,
                    "effective_frame_id": response["effective_frame_id"],
                    "observed_quality_flags": evidence["quality_flags"],
                    "displayed_class": evidence["displayed_class"],
                    "valid_peer_network_count": len(valid_peer_results),
                }
            )
        return {
            "session_id": session_id,
            "observation_only": True,
            "truth_endpoint_requested": False,
            "cases": event_evidence,
            "quality_abstention_count": analysis["quality_abstention_count"],
        }
    finally:
        _cleanup_session(client, session_id)


def _wait_for_observed_window(
    client: JsonHttpClient,
    session_id: str,
    *,
    after_completed_count: int,
    timeout_s: float,
) -> dict[str, Any]:
    def fetch() -> dict[str, Any]:
        return client.request_json("GET", f"/api/demo/analysis/{session_id}").body

    return _wait_for(
        "new observation-only analysis window",
        fetch,
        lambda value: (
            value.get("completed_window_count", 0) > after_completed_count
            and len(value.get("latest_results", ())) == 4
        ),
        timeout_s=timeout_s,
    )


def _schedule_observed_events(
    client: JsonHttpClient,
    session_id: str,
    *,
    event_specs: tuple[dict[str, Any], ...],
    analysis: dict[str, Any],
    timeout_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session = client.request_json("GET", f"/api/network/sessions/{session_id}").body
    start_time_s = round(session["status"]["sim_time_s"] + 0.01, 8)
    scheduled: list[dict[str, Any]] = []
    for spec in event_specs:
        response = client.request_json(
            "POST",
            f"/api/network/sessions/{session_id}/events",
            payload={
                **spec,
                "start_time_s": start_time_s,
                "duration_s": 4.0,
            },
            expected=(201,),
        ).body
        scheduled.append(
            {
                "event_id": response["event"]["event_id"],
                "effective_frame_id": response["effective_frame_id"],
                "configuration_version": response["configuration_version"],
            }
        )
    before_count = analysis["completed_window_count"]
    _step(client, session_id, 400)
    return (
        _wait_for_observed_window(
            client,
            session_id,
            after_completed_count=before_count,
            timeout_s=timeout_s,
        ),
        scheduled,
    )


def _perturbation_cases() -> tuple[dict[str, Any], ...]:
    """Bounded acceptance controls; names are timing evidence, never model inputs."""

    return (
        {
            "case_id": "common_environment",
            "events": (
                {
                    "event_id": "acceptance-common-environment",
                    "kind": "world_field_offset",
                    "target_sensor_ids": [],
                    "field_offset_T": [24.0e-9, -6.0e-9, 4.0e-9],
                },
            ),
            "interpretation": (
                "A multi-node response is compatible with a shared environmental "
                "change. This is pattern recognition, not proof of physical cause."
            ),
        },
        {
            "case_id": "s3_drift_and_noise",
            "events": (
                {
                    "event_id": "acceptance-s3-drift",
                    "kind": "node_drift",
                    "target_sensor_ids": ["S3"],
                    "drift_rate_T_per_s": [3.0e-9, 0.0, -1.0e-9],
                },
                {
                    "event_id": "acceptance-s3-noise",
                    "kind": "node_noise_burst",
                    "target_sensor_ids": ["S3"],
                    "noise_multiplier": 8.0,
                },
            ),
            "interpretation": (
                "A focal S3 response with peer context is compatible with node-specific "
                "degradation. The drift/noise command name is not a prediction input."
            ),
        },
        {
            "case_id": "shared_instrument_offset",
            "events": (
                {
                    "event_id": "acceptance-shared-instrument-offset",
                    "kind": "shared_instrument_offset",
                    "target_sensor_ids": ["S1", "S2", "S3", "S4"],
                    "field_offset_T": [18.0e-9, 0.0, 0.0],
                },
            ),
            "interpretation": (
                "A shared instrument offset can resemble a common-field response; "
                "several changed nodes do not establish a physical cause."
            ),
        },
        {
            "case_id": "mixed_shared_and_focal",
            "events": (
                {
                    "event_id": "acceptance-mixed-common",
                    "kind": "world_field_offset",
                    "target_sensor_ids": [],
                    "field_offset_T": [-16.0e-9, 5.0e-9, 0.0],
                },
                {
                    "event_id": "acceptance-mixed-s3-bias",
                    "kind": "node_bias",
                    "target_sensor_ids": ["S3"],
                    "field_offset_T": [31.0e-9, 0.0, 0.0],
                },
            ),
            "interpretation": (
                "Simultaneous shared and focal observations remain mixed or ambiguous; "
                "reported model scores are not calibrated probabilities."
            ),
        },
    )


def run_perturbation_acceptance(
    client: JsonHttpClient,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """Exercise real observations through moving, shared, focal and mixed conditions."""

    configuration = _create_configuration(
        client,
        node_count=4,
        seed=8_604,
        time_scale=1.0,
    )
    configuration["buffer_duration_s"] = 40.0
    configuration["session_name"] = "AQSE four-node controlled perturbation acceptance"
    environment = dict(configuration["environment"])
    environment["dipoles"] = [
        {
            "source_id": "acceptance-moving-local-source",
            "initial_position_m": [-2.5, -0.35, 1.2],
            "velocity_m_per_s": [0.125, 0.0, 0.0],
            "moment_A_m2": [0.0, 0.0, 0.45],
            "minimum_distance_m": 0.1,
            "enabled": True,
        }
    ]
    configuration["environment"] = environment
    created = client.request_json(
        "POST",
        "/api/network/sessions",
        payload=configuration,
        expected=(201,),
    ).body
    session_id = created["status"]["session_id"]
    try:
        initial_view = client.request_json(
            "POST",
            f"/api/demo/analysis/{session_id}/start",
            payload={"acquire_reference": True},
        ).body
        client.request_json("POST", f"/api/network/sessions/{session_id}/start")
        paused = client.request_json("POST", f"/api/network/sessions/{session_id}/pause").body
        reference_and_window_frames = 1_200 - paused["status"]["latest_frame_id"]
        if reference_and_window_frames > 0:
            _step(client, session_id, reference_and_window_frames)
        baseline = _wait_for_observed_window(
            client,
            session_id,
            after_completed_count=0,
            timeout_s=timeout_s,
        )

        before_moving_count = baseline["completed_window_count"]
        _step(client, session_id, 400)
        moving = _wait_for_observed_window(
            client,
            session_id,
            after_completed_count=before_moving_count,
            timeout_s=timeout_s,
        )
        moving_change = _observed_change(baseline, moving)
        moving_means = [float(item["feature_values"][0]) for item in moving["latest_results"]]
        spatial_spread_nt = max(moving_means) - min(moving_means)
        if spatial_spread_nt <= 1.0e-9:
            raise AcceptanceError(
                "moving local source did not produce spatially distinct node responses"
            )
        cases: list[dict[str, Any]] = [
            {
                "case_id": "moving_local_source",
                "scheduled_controls": [],
                "observed_change": moving_change,
                "observed_f0_spatial_spread_nT": spatial_spread_nt,
                "interpretation": (
                    "Spatially different observations are compatible with a moving/local "
                    "source; AQSE v1 does not perform inverse localization or prove cause."
                ),
            }
        ]
        previous = moving
        for case in _perturbation_cases():
            current, scheduled = _schedule_observed_events(
                client,
                session_id,
                event_specs=case["events"],
                analysis=previous,
                timeout_s=timeout_s,
            )
            cases.append(
                {
                    "case_id": case["case_id"],
                    "scheduled_controls": scheduled,
                    "observed_change": _observed_change(previous, current),
                    "interpretation": case["interpretation"],
                }
            )
            previous = current

        before_replay = {
            item["sensor_id"]: _result_evidence(item) for item in previous["latest_results"]
        }
        stopped = client.request_json("POST", f"/api/demo/analysis/{session_id}/stop").body
        replay = client.request_json("POST", f"/api/network/sessions/{session_id}/replay").body
        restarted = client.request_json(
            "POST",
            f"/api/demo/analysis/{session_id}/start",
            payload={"acquire_reference": True},
        ).body

        def fetch_replay() -> dict[str, Any]:
            return client.request_json("GET", f"/api/demo/analysis/{session_id}").body

        replayed = _wait_for(
            "deterministic observation replay",
            fetch_replay,
            lambda value: (
                value.get("latest_analyzed_window_start_s")
                == previous.get("latest_analyzed_window_start_s")
                and len(value.get("latest_results", ())) == 4
            ),
            timeout_s=timeout_s,
        )
        after_replay = {
            item["sensor_id"]: _result_evidence(item) for item in replayed["latest_results"]
        }
        for sensor_id in before_replay:
            left = before_replay[sensor_id]
            right = after_replay[sensor_id]
            for field_name in (
                "feature_values",
                "afse_vector",
                "displayed_class",
                "class_scores",
                "observable_rule_status",
                "uncertain",
            ):
                if left[field_name] != right[field_name]:
                    raise AcceptanceError(f"replay changed {field_name} for {sensor_id}")
        if restarted["worker_epoch"] <= stopped["worker_epoch"]:
            raise AcceptanceError("analysis restart did not advance its stale-result epoch")
        return {
            "session_id": session_id,
            "observation_only": True,
            "prediction_receives_event_commands": False,
            "initial_worker_epoch": initial_view["worker_epoch"],
            "cases": cases,
            "replay": {
                "session_state": replay["status"]["state"],
                "replayed_frame_count": replay["status"]["latest_frame_id"],
                "previous_worker_epoch": stopped["worker_epoch"],
                "new_worker_epoch": restarted["worker_epoch"],
                "features_and_predictions_reproduced": True,
                "stale_worker_generation_rejected": True,
            },
        }
    finally:
        _cleanup_session(client, session_id)


def run_bundle_acceptance(
    client: JsonHttpClient,
    registry: dict[str, Any],
) -> dict[str, Any]:
    active = registry.get("active")
    freeze_id = registry.get("selection_freeze_id")
    if not active or not freeze_id:
        raise AcceptanceError("prepared registry has no active compatible bundle pair")
    payload = {
        "local_bundle_id": active["local_bundle_id"],
        "network_bundle_id": active["network_bundle_id"],
        "selection_freeze_id": freeze_id,
    }
    applied = client.request_json("POST", "/api/demo/bundles/apply", payload=payload).body
    if (
        applied["local_bundle_id"] != payload["local_bundle_id"]
        or applied["network_bundle_id"] != payload["network_bundle_id"]
    ):
        raise AcceptanceError("compatible bundle application was not atomic")
    retried = client.request_json("POST", "/api/demo/bundles/apply", payload=payload).body
    if retried != applied:
        raise AcceptanceError("identical bundle-application retry was not idempotent")
    mismatch = {
        **payload,
        "local_bundle_id": active["network_bundle_id"],
        "network_bundle_id": active["local_bundle_id"],
    }
    rejected = client.request_json(
        "POST",
        "/api/demo/bundles/apply",
        payload=mismatch,
        expected=(409,),
    )
    return {
        "application_id": applied["application_id"],
        "compatible_retry_idempotent": True,
        "compatible_retry_application_id": retried["application_id"],
        "mismatched_pair_status": rejected.status,
        "mismatched_pair_detail": rejected.body.get("detail"),
    }


def run_training_acceptance(
    client: JsonHttpClient,
    registry: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    study_id = registry.get("study_artifact_id")
    if not study_id:
        raise AcceptanceError("prepared registry has no immutable study identifier")
    probe_configuration = _create_configuration(
        client,
        node_count=1,
        seed=8_501,
        time_scale=1.0,
    )
    probe = client.request_json(
        "POST",
        "/api/network/sessions",
        payload=probe_configuration,
        expected=(201,),
    ).body
    probe_id = probe["status"]["session_id"]
    try:
        client.request_json("POST", f"/api/network/sessions/{probe_id}/start")
        initial_frame = client.request_json("GET", f"/api/network/sessions/{probe_id}").body[
            "status"
        ]["latest_frame_id"]
        intent = {
            "intent_id": f"aqse-demo-intent-acceptance-{uuid4().hex}",
            "study_artifact_id": study_id,
            "requested_action": "fit-two-candidate-local-and-network-bundles",
        }
        first = client.request_json(
            "POST",
            "/api/demo/training/jobs",
            payload=intent,
            expected=(201,),
        )
        retry = client.request_json(
            "POST",
            "/api/demo/training/jobs",
            payload=intent,
            expected=(200,),
        )
        if first.body["job_id"] != retry.body["job_id"]:
            raise AcceptanceError("duplicate training intent launched a second job")
        health = client.request_json("GET", "/api/health")
        if health.elapsed_ms >= 1_000.0:
            raise AcceptanceError("lightweight health exceeded one second during training")
        time.sleep(0.15)
        later_frame = client.request_json("GET", f"/api/network/sessions/{probe_id}").body[
            "status"
        ]["latest_frame_id"]
        if later_frame <= initial_frame:
            raise AcceptanceError("sensor acquisition stopped while training held the slot")
        cancelled = client.request_json(
            "POST", f"/api/demo/training/jobs/{first.body['job_id']}/cancel"
        ).body

        def fetch_job() -> dict[str, Any]:
            return client.request_json(
                "GET", f"/api/demo/training/jobs/{first.body['job_id']}"
            ).body

        terminal = _wait_for(
            "bounded training cancellation",
            fetch_job,
            lambda value: value.get("state") in {"cancelled", "completed", "failed"},
            timeout_s=timeout_s,
            interval_s=0.2,
        )
        if terminal["state"] == "failed":
            raise AcceptanceError(f"training job failed: {terminal.get('error')}")
        return {
            "job_id": first.body["job_id"],
            "first_status": first.status,
            "retry_status": retry.status,
            "same_job_for_duplicate_intent": True,
            "cancel_request_stage": cancelled["current_stage"],
            "terminal_state": terminal["state"],
            "health_during_training_ms": health.elapsed_ms,
            "sensor_frames_advanced_during_training": later_frame - initial_frame,
            "automatic_bundle_application": False,
        }
    finally:
        _cleanup_session(client, probe_id)


def _safe_report_path(report_dir: Path, stem: str) -> Path:
    resolved = report_dir.expanduser().resolve()
    if resolved == REPOSITORY_ROOT.resolve() or REPOSITORY_ROOT.resolve() in resolved.parents:
        raise AcceptanceError("acceptance reports must be written outside the Git tree")
    resolved.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return resolved / f"{stem}-{timestamp}.json"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_acceptance(
    *,
    base_url: str,
    frontend_url: str,
    timeout_s: float,
) -> tuple[dict[str, Any], JsonHttpClient]:
    client = JsonHttpClient(base_url=base_url, timeout_s=min(timeout_s, 15.0))
    started_at = datetime.now(timezone.utc)
    health = client.request_json("GET", "/api/health")
    quantum_health = client.request_json("GET", "/api/quantum/health")
    if health.body != {"status": "ok", "service": "AQSE Backend"}:
        raise AcceptanceError(f"unexpected backend health contract: {health.body}")
    if (
        quantum_health.body.get("status") != "ok"
        or quantum_health.body.get("adapter") != "ready"
        or quantum_health.body.get("qubits") != 8
        or quantum_health.body.get("features") != 8
    ):
        raise AcceptanceError(f"unexpected quantum readiness contract: {quantum_health.body}")
    if max(health.elapsed_ms, quantum_health.elapsed_ms) >= 1_000.0:
        raise AcceptanceError("a lightweight health endpoint exceeded one second")
    frontend = _get_text(frontend_url, timeout_s=client.timeout_s)
    if frontend.status != 200 or "AQSE" not in frontend.body:
        raise AcceptanceError("frontend did not return the AQSE application shell")
    registry = client.request_json("GET", "/api/demo/registry").body
    if not registry.get("prepared"):
        raise AcceptanceError(registry.get("preparation_detail", "run make prepare-demo"))
    if registry.get("historical_test_ledger_sha256") != HISTORICAL_TEST_LEDGER_SHA256:
        raise AcceptanceError("historical TEST ledger identity changed")

    bundle_evidence = run_bundle_acceptance(client, registry)
    node_evidence = [
        run_node_acceptance(client, node_count=count, timeout_s=timeout_s) for count in (1, 2, 4, 8)
    ]
    perturbation_evidence = run_perturbation_acceptance(
        client,
        timeout_s=timeout_s,
    )
    invalid_evidence = run_invalid_input_acceptance(client, timeout_s=timeout_s)
    training_evidence = run_training_acceptance(
        client,
        registry,
        timeout_s=max(timeout_s, 120.0),
    )
    registry_after = client.request_json("GET", "/api/demo/registry").body
    active_after = registry_after.get("active") or {}
    if active_after.get("application_id") != bundle_evidence["application_id"]:
        raise AcceptanceError(
            "saved active bundle identity changed during acceptance without explicit apply"
        )
    elapsed_s = (datetime.now(timezone.utc) - started_at).total_seconds()
    return (
        {
            "schema_version": "aqse.demo-acceptance-report.v2",
            "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
            "elapsed_wall_s": elapsed_s,
            "result": "passed",
            "scientific_label": "research / not validated for field deployment",
            "prediction_input_boundary": (
                "observation and analysis REST endpoints only; simulator truth not requested"
            ),
            "service_health": {
                "backend": health.body,
                "backend_elapsed_ms": health.elapsed_ms,
                "quantum": quantum_health.body,
                "quantum_elapsed_ms": quantum_health.elapsed_ms,
                "frontend_http_status": frontend.status,
                "frontend_elapsed_ms": frontend.elapsed_ms,
            },
            "registry": {
                "study_artifact_id": registry["study_artifact_id"],
                "selection_freeze_id": registry["selection_freeze_id"],
                "final_evaluation_id": registry["final_evaluation_id"],
                "historical_test_ledger_sha256": registry["historical_test_ledger_sha256"],
            },
            "bundle_application": bundle_evidence,
            "node_routes": node_evidence,
            "controlled_perturbations": perturbation_evidence,
            "invalid_inputs": invalid_evidence,
            "training": training_evidence,
            "restart_and_persistence": {
                "saved_bundle_loaded_at_acceptance_start": True,
                "active_application_stable_during_run": True,
                "active_application_id": bundle_evidence["application_id"],
                "startup_training_request_sent_by_acceptance": False,
                "process_restart_check": "external-orchestrator-required",
                "process_restart_reason": (
                    "An in-process REST client must not terminate the backend or Docker "
                    "services that are serving its own acceptance run. Restart services "
                    "externally, confirm both health checks, then rerun this script."
                ),
            },
            "http_call_count": len(client.calls),
            "http_calls": client.calls,
        },
        client,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:3000")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    if args.timeout_s <= 0.0:
        parser.error("--timeout-s must be positive")
    try:
        report, _ = run_acceptance(
            base_url=args.base_url,
            frontend_url=args.frontend_url,
            timeout_s=args.timeout_s,
        )
        path = _safe_report_path(args.report_dir, "demo-acceptance")
        _write_report(path, report)
    except (AcceptanceError, OSError, ValueError) as exc:
        print(f"AQSE acceptance failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"result": "passed", "report_path": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
