from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from app.training.canonical import file_sha256
from app.training.storage import artifact_root

from .evaluation import (
    BOOTSTRAP_REPLICATES,
    MAXIMUM_NORMAL_FALSE_POSITIVE_RATE,
    MINIMUM_ADVANTAGE_DELTA,
    MINIMUM_CLASS_RECALL,
    MINIMUM_COVERAGE,
    MINIMUM_NORMAL_RECALL,
    PRIMARY_BALANCED_ACCURACY_THRESHOLD,
    PRIMARY_MACRO_F1_THRESHOLD,
    active_network_model_binding,
    assess_pilot,
    evaluate_frozen_network_model,
    validate_evaluation_binding,
)
from .generation import build_canonical_plans, build_pilot_plans, generate_episode
from .models import FourSensorStudyBuild, FourSensorStudyEpisodePlan
from .protocol import FROZEN_FOUR_SENSOR_STUDY_PROTOCOL
from .storage import (
    load_test_labels,
    load_test_observations,
    mark_evaluation_published,
    read_test_ledger,
    write_final_evaluation,
    write_final_test_dataset,
    write_model_binding,
    write_pilot,
    write_protocol_freeze,
    write_test_authorization,
)

Progress = Callable[[str], None]


@dataclass(frozen=True)
class FourSensorWorkflowResult:
    pilot_path: Path
    pilot_manifest: dict[str, Any]
    pilot_assessment: dict[str, Any]
    protocol_freeze_path: Path
    protocol_freeze_manifest: dict[str, Any]
    dataset_path: Path
    dataset_manifest: dict[str, Any]
    model_binding_path: Path
    model_binding_manifest: dict[str, Any]
    authorization_path: Path
    authorization_manifest: dict[str, Any]
    evaluation_path: Path
    evaluation_manifest: dict[str, Any]
    evaluation: dict[str, Any]
    ledger_entries: tuple[dict[str, Any], ...]
    timings_s: dict[str, float]


def _generate_build(
    plans: Sequence[FourSensorStudyEpisodePlan],
    *,
    kind: str,
    protocol_freeze_digest: str | None,
    report: Progress,
) -> FourSensorStudyBuild:
    observations = []
    labels = []
    started = perf_counter()
    for index, plan in enumerate(plans, start=1):
        observation, label = generate_episode(plan)
        observations.append(observation)
        labels.append(label)
        if index == 1 or index % 5 == 0 or index == len(plans):
            report(
                f"Generated {index}/{len(plans)} {kind.replace('_', ' ')} episodes "
                f"({perf_counter() - started:.1f} s)."
            )
    return FourSensorStudyBuild(
        kind=kind,
        protocol_digest=FROZEN_FOUR_SENSOR_STUDY_PROTOCOL.digest,
        protocol_freeze_digest=protocol_freeze_digest,
        episode_plans=tuple(plans),
        observations=tuple(observations),
        labels=tuple(labels),
    )


def _source_hashes() -> dict[str, str]:
    backend = Path(__file__).resolve().parents[2]
    paths = (
        "app/four_sensor_study/protocol.py",
        "app/four_sensor_study/models.py",
        "app/four_sensor_study/generation.py",
        "app/four_sensor_study/evaluation.py",
        "app/quantum/user_pipeline/tqk8.py",
        "app/quantum/user_pipeline/sampler_qng.py",
        "tests/quantum/test_tqk8.py",
    )
    return {name: file_sha256(backend / name) for name in paths}


def _protocol_publication(model_binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "aqse.four-sensor-study.published-protocol.v1",
        "protocol": FROZEN_FOUR_SENSOR_STUDY_PROTOCOL.model_dump(mode="json"),
        "evaluation_target": {
            "kind": "pre-existing-frozen-AQSE-network-model",
            "model_binding": model_binding,
            "training_permitted": False,
            "selection_permitted": False,
            "refit_permitted": False,
        },
        "predeclared_methods": (
            "frozen_quantum_afse_mlp",
            "raw_state8_mlp",
            "observable_p99_rule",
        ),
        "primary_metrics": ("balanced_accuracy", "macro_f1"),
        "secondary_metrics": (
            "per_class_precision_recall_f1",
            "coverage",
            "normal_false_positive_episode_rate",
            "confusion_matrix",
            "subgroup_by_geometry",
            "subgroup_by_focal_sensor",
        ),
        "uncertainty": {
            "method": "episode-level-stratified-percentile-bootstrap",
            "confidence_level": 0.95,
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "success_criteria": {
            "balanced_accuracy_min": PRIMARY_BALANCED_ACCURACY_THRESHOLD,
            "macro_f1_min": PRIMARY_MACRO_F1_THRESHOLD,
            "each_class_recall_min": MINIMUM_CLASS_RECALL,
            "normal_recall_min": MINIMUM_NORMAL_RECALL,
            "coverage_min": MINIMUM_COVERAGE,
            "normal_false_positive_episode_rate_max": (
                MAXIMUM_NORMAL_FALSE_POSITIVE_RATE
            ),
        },
        "quantum_advantage_rule": {
            "paired_balanced_accuracy_delta_min": MINIMUM_ADVANTAGE_DELTA,
            "paired_delta_95pct_lower_bound_must_exceed": 0.0,
            "coverage_must_not_decrease": True,
            "normal_recall_must_not_decrease": True,
        },
        "test_access_plan": (
            "sealed",
            "observations_opened",
            "labels_opened",
            "evaluation_published",
        ),
        "pilot_use": "quality-only;excluded-from-final-test;no-model-execution",
        "historical_artifact_policy": "never-read-modify-or-reopen",
        "effective_source_hashes": _source_hashes(),
    }


def run_four_sensor_study(
    *,
    root: Path | None = None,
    progress: Progress | None = None,
) -> FourSensorWorkflowResult:
    """Run pilot, freeze, and one 100-episode evaluation in that exact order."""

    artifact_base = (root or artifact_root()).resolve()
    report = progress or (lambda _message: None)
    timings: dict[str, float] = {}
    workflow_started = perf_counter()

    report("Stage 1/6: generating the 20-episode noncanonical pilot.")
    started = perf_counter()
    pilot_build = _generate_build(
        build_pilot_plans(),
        kind="pilot",
        protocol_freeze_digest=None,
        report=report,
    )
    pilot_path, pilot_manifest, _ = write_pilot(pilot_build, root=artifact_base)
    pilot_assessment = assess_pilot(pilot_build.observations, pilot_build.labels)
    if pilot_assessment["accepted"] is not True:
        raise RuntimeError("the noncanonical pilot did not pass the frozen quality gate")
    timings["pilot_generation_and_publication"] = perf_counter() - started
    report(
        f"Pilot accepted and published as {pilot_manifest['artifact_id']}; "
        "its rows are ineligible for final evaluation."
    )

    report("Stage 2/6: binding the pre-existing model and publishing protocol freeze.")
    started = perf_counter()
    model_binding = active_network_model_binding(root=artifact_base)
    publication = _protocol_publication(model_binding)
    protocol_path, protocol_manifest, _ = write_protocol_freeze(
        publication,
        pilot_path=pilot_path,
        acceptance=pilot_assessment,
        root=artifact_base,
    )
    timings["protocol_freeze_publication"] = perf_counter() - started
    report(f"Protocol frozen as {protocol_manifest['artifact_id']}.")

    report("Stage 3/6: generating exactly 100 post-freeze held-out TEST episodes.")
    started = perf_counter()
    final_build = _generate_build(
        build_canonical_plans(
            protocol_freeze_digest=protocol_manifest["content_digest"]
        ),
        kind="final_test",
        protocol_freeze_digest=protocol_manifest["content_digest"],
        report=report,
    )
    dataset_path, dataset_manifest, _ = write_final_test_dataset(
        final_build,
        protocol_freeze_path=protocol_path,
        root=artifact_base,
    )
    timings["final_test_generation_and_publication"] = perf_counter() - started
    report(
        f"Held-out corpus published as {dataset_manifest['artifact_id']}; TEST is sealed."
    )

    report("Stage 4/6: publishing immutable model binding and explicit authorization.")
    started = perf_counter()
    model_path, model_manifest, _ = write_model_binding(
        model_binding,
        protocol_freeze_path=protocol_path,
        dataset_path=dataset_path,
        root=artifact_base,
    )
    authorization_payload = {
        "schema_version": "aqse.four-sensor-study.test-authorization.v1",
        "authorized": True,
        "authority": "explicit project-author instruction in current task",
        "scope": "one 100-episode held-out evaluation only",
        "fit_permitted": False,
        "selection_change_permitted": False,
        "additional_test_runs_permitted": False,
    }
    authorization_path, authorization_manifest, _ = write_test_authorization(
        authorization_payload,
        protocol_freeze_path=protocol_path,
        dataset_path=dataset_path,
        model_binding_path=model_path,
        root=artifact_base,
    )
    timings["binding_and_authorization_publication"] = perf_counter() - started

    if len(read_test_ledger(dataset_path).entries) != 1:
        raise RuntimeError("new TEST ledger was not sealed before semantic access")

    report("Stage 5/6: opening TEST once and executing the frozen model.")
    started = perf_counter()
    test_observations = load_test_observations(
        dataset_path,
        protocol_freeze_path=protocol_path,
        model_binding_path=model_path,
        authorization_path=authorization_path,
    )
    test_labels = load_test_labels(
        dataset_path,
        protocol_freeze_path=protocol_path,
        model_binding_path=model_path,
        authorization_path=authorization_path,
    )
    evaluation = evaluate_frozen_network_model(
        test_observations,
        test_labels,
        model_binding=model_binding,
        protocol_freeze_id=protocol_manifest["artifact_id"],
        dataset_id=dataset_manifest["artifact_id"],
        authorization_id=authorization_manifest["artifact_id"],
        root=artifact_base,
    )
    validate_evaluation_binding(
        evaluation,
        protocol_freeze_id=protocol_manifest["artifact_id"],
        dataset_id=dataset_manifest["artifact_id"],
        authorization_id=authorization_manifest["artifact_id"],
    )
    timings["single_test_access_and_evaluation"] = perf_counter() - started

    report("Stage 6/6: publishing the final result and closing the TEST ledger.")
    started = perf_counter()
    evaluation_path, evaluation_manifest, _ = write_final_evaluation(
        evaluation,
        protocol_freeze_path=protocol_path,
        dataset_path=dataset_path,
        model_binding_path=model_path,
        authorization_path=authorization_path,
        root=artifact_base,
    )
    ledger = mark_evaluation_published(
        dataset_path,
        evaluation_path=evaluation_path,
        protocol_freeze_path=protocol_path,
        model_binding_path=model_path,
        authorization_path=authorization_path,
    )
    if tuple(entry["event"] for entry in ledger.entries) != (
        "sealed",
        "observations_opened",
        "labels_opened",
        "evaluation_published",
    ):
        raise RuntimeError("final TEST ledger did not close in the frozen order")
    timings["evaluation_publication"] = perf_counter() - started
    timings["workflow_total"] = perf_counter() - workflow_started
    report(f"Evaluation published as {evaluation_manifest['artifact_id']}.")

    return FourSensorWorkflowResult(
        pilot_path=pilot_path,
        pilot_manifest=pilot_manifest,
        pilot_assessment=pilot_assessment,
        protocol_freeze_path=protocol_path,
        protocol_freeze_manifest=protocol_manifest,
        dataset_path=dataset_path,
        dataset_manifest=dataset_manifest,
        model_binding_path=model_path,
        model_binding_manifest=model_manifest,
        authorization_path=authorization_path,
        authorization_manifest=authorization_manifest,
        evaluation_path=evaluation_path,
        evaluation_manifest=evaluation_manifest,
        evaluation=evaluation,
        ledger_entries=ledger.entries,
        timings_s=timings,
    )
