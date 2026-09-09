from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np

from app.demo.bundle_models import (
    LOCAL_TASK_ID,
    NETWORK_TASK_ID,
    DemoEvaluationMetrics,
    DemoFinalEvaluation,
    DemoModelSelectionFreeze,
    DemoResearchBundle,
    DemoTaskPartition,
)
from app.demo.bundle_storage import (
    apply_selection_freeze,
    list_bundles,
    load_final_evaluation,
    load_selection_freeze,
    write_bundle,
    write_final_evaluation,
    write_selection_freeze,
)
from app.demo.evaluation import (
    DemoTrainingCancelled,
    assemble_task_partition,
    evaluate_frozen_test,
    fit_demo_task_candidates,
    freeze_model_selection,
)
from app.demo.protocol import FROZEN_NETWORK_DEMO_PROTOCOL
from app.demo.runtime import bundle_runtime_cache
from app.demo.runtime_models import (
    DemoBundleSummary,
    DemoMetricSummary,
    DemoRegistryView,
)
from app.demo.study import build_network_study
from app.demo.study_models import (
    LoadedStudyPartition,
    NetworkStudyManifest,
    StudyPartition,
)
from app.demo.study_storage import (
    HISTORICAL_TEST_LEDGER_SHA256,
    load_development_partition,
    load_new_test_labels,
    load_new_test_observations,
    read_demo_test_ledger,
    verify_network_study,
    write_network_study,
)
from app.demo.training import qiskit_numpy_preflight, qng_wrapper_equivalence
from app.training.storage import artifact_root

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class DemoPreparationResult:
    study_path: Path
    manifest: NetworkStudyManifest
    freeze: DemoModelSelectionFreeze
    final_evaluation: DemoFinalEvaluation
    bundles: tuple[DemoResearchBundle, ...]
    reused_study: bool


def _network_demo_root(root: Path | None = None) -> Path:
    return (root or artifact_root()).resolve() / "network-demo"


def _artifact_directories(parent: Path) -> tuple[Path, ...]:
    if not parent.is_dir() or parent.is_symlink():
        return ()
    return tuple(
        path
        for path in sorted(parent.iterdir(), key=lambda item: item.name)
        if path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
    )


def find_current_study(
    *,
    root: Path | None = None,
) -> tuple[Path, NetworkStudyManifest] | None:
    matches: list[tuple[Path, NetworkStudyManifest]] = []
    for path in _artifact_directories(_network_demo_root(root) / "studies"):
        manifest = verify_network_study(path)
        if manifest.protocol_digest == FROZEN_NETWORK_DEMO_PROTOCOL.digest:
            matches.append((path, manifest))
    if len(matches) > 1:
        identities = {manifest.content_digest for _, manifest in matches}
        if len(identities) > 1:
            raise RuntimeError("multiple incompatible canonical demo studies are present")
    return None if not matches else matches[-1]


def _current_freeze(
    study_artifact_id: str,
    *,
    root: Path | None = None,
) -> DemoModelSelectionFreeze | None:
    matches = [
        value
        for value in (
            load_selection_freeze(path)
            for path in _artifact_directories(
                _network_demo_root(root) / "registry" / "selection-freezes"
            )
        )
        if value.study_artifact_id == study_artifact_id
        and value.protocol_digest == FROZEN_NETWORK_DEMO_PROTOCOL.digest
    ]
    if len({item.content_digest for item in matches}) > 1:
        raise RuntimeError("multiple selection freezes exist for the canonical demo study")
    return None if not matches else matches[-1]


def _current_final_evaluation(
    freeze_id: str,
    *,
    root: Path | None = None,
) -> DemoFinalEvaluation | None:
    matches = [
        value
        for value in (
            load_final_evaluation(path)
            for path in _artifact_directories(
                _network_demo_root(root) / "registry" / "final-evaluations"
            )
        )
        if value.freeze_id == freeze_id
    ]
    if len({item.content_digest for item in matches}) > 1:
        raise RuntimeError("multiple final evaluations exist for one selection freeze")
    return None if not matches else matches[-1]


def _preflight() -> None:
    generator = np.random.default_rng(2_001_004)
    encoded = generator.uniform(-np.pi, np.pi, size=(4, 8)).astype(np.float64)
    labels = (-1, -1, 1, 1)
    qng_wrapper_equivalence(encoded, labels, steps=2)
    qiskit_numpy_preflight(encoded[:2], (-1, 1))


def load_task_partitions(
    study_path: Path,
    manifest: NetworkStudyManifest,
    partition: StudyPartition,
) -> tuple[DemoTaskPartition, DemoTaskPartition]:
    loaded = load_development_partition(
        study_path,
        partition,
        include_labels=True,
    )
    return (
        assemble_task_partition(
            loaded,
            task_id=LOCAL_TASK_ID,
            study_artifact_id=manifest.artifact_id,
            study_content_digest=manifest.content_digest,
        ),
        assemble_task_partition(
            loaded,
            task_id=NETWORK_TASK_ID,
            study_artifact_id=manifest.artifact_id,
            study_content_digest=manifest.content_digest,
        ),
    )


def prepare_demo(
    *,
    root: Path | None = None,
    cancellation: Event | None = None,
    progress: ProgressCallback | None = None,
) -> DemoPreparationResult:
    """Create/reuse the bounded study, select on VALIDATION, then open new TEST once."""

    cancellation = cancellation or Event()
    report = progress or (lambda _message: None)
    located = find_current_study(root=root)
    reused_study = located is not None
    if located is None:
        report("Generating the frozen 160-episode observation study.")
        if cancellation.is_set():
            raise DemoTrainingCancelled("demo preparation cancelled before generation")
        build = build_network_study()
        study_path, manifest, _ = write_network_study(build, root=root)
    else:
        study_path, manifest = located
        report(f"Reusing immutable study {manifest.artifact_id}.")

    freeze = _current_freeze(manifest.artifact_id, root=root)
    if freeze is None:
        report("Running QNG wrapper and Qiskit/NumPy preflight checks.")
        _preflight()
        report("Loading TRAIN and VALIDATION channels; TEST remains sealed.")
        local_train, network_train = load_task_partitions(
            study_path,
            manifest,
            "train",
        )
        local_validation, network_validation = load_task_partitions(
            study_path,
            manifest,
            "validation",
        )
        report("Fitting the two bounded local theta candidates.")
        local_fit = fit_demo_task_candidates(
            local_train,
            local_validation,
            cancellation=cancellation,
        )
        report("Fitting the two bounded network theta candidates.")
        network_fit = fit_demo_task_candidates(
            network_train,
            network_validation,
            cancellation=cancellation,
        )
        for bundle in (*local_fit.bundles, *network_fit.bundles):
            write_bundle(bundle, root=root)
        freeze = freeze_model_selection(local_fit, network_fit)
        write_selection_freeze(freeze, root=root)
        report(f"Selection frozen as {freeze.freeze_id}; TEST is still sealed.")
    else:
        report(f"Reusing selection freeze {freeze.freeze_id}.")

    final = _current_final_evaluation(freeze.freeze_id, root=root)
    if final is None:
        ledger = read_demo_test_ledger(study_path)
        if len(ledger) != 1:
            raise RuntimeError(
                "new-study TEST access already advanced without a persisted final evaluation"
            )
        report("Opening the new study TEST once under the frozen selection.")
        test_observations = load_new_test_observations(
            study_path,
            selection_freeze_id=freeze.freeze_id,
        )
        test_labels = load_new_test_labels(
            study_path,
            selection_freeze_id=freeze.freeze_id,
        )
        loaded_test = LoadedStudyPartition(
            partition="test",
            observations=test_observations.observations,
            labels=test_labels,
        )
        local_test = assemble_task_partition(
            loaded_test,
            task_id=LOCAL_TASK_ID,
            study_artifact_id=manifest.artifact_id,
            study_content_digest=manifest.content_digest,
        )
        network_test = assemble_task_partition(
            loaded_test,
            task_id=NETWORK_TASK_ID,
            study_artifact_id=manifest.artifact_id,
            study_content_digest=manifest.content_digest,
        )
        bundles_by_id = {
            bundle.bundle_id: bundle for bundle in list_bundles(root=root)
        }
        final = evaluate_frozen_test(
            freeze,
            bundles_by_id,
            local_test=local_test,
            network_test=network_test,
            query_acquisition_id=(
                f"{manifest.artifact_id}:single-frozen-test-evaluation"
            ),
        )
        write_final_evaluation(final, root=root)
        report(f"Single TEST evaluation persisted as {final.evaluation_id}.")
    else:
        report(f"Reusing final evaluation {final.evaluation_id}; TEST not reopened.")

    apply_selection_freeze(freeze, root=root)
    bundle_runtime_cache.invalidate()
    report("Selected local/network bundles applied atomically.")
    return DemoPreparationResult(
        study_path=study_path,
        manifest=manifest,
        freeze=freeze,
        final_evaluation=final,
        bundles=list_bundles(root=root),
        reused_study=reused_study,
    )


def _metric_summary(metric: DemoEvaluationMetrics) -> DemoMetricSummary:
    return DemoMetricSummary(
        partition=metric.partition,
        task_id=metric.task_id,
        balanced_accuracy=metric.balanced_accuracy,
        macro_f1=metric.macro_f1,
        coverage=metric.coverage,
        sample_count=metric.sample_count,
        uncertain_count=metric.uncertain_count,
        heuristic_ood_count=metric.heuristic_ood_count,
    )


def demo_registry_view(*, root: Path | None = None) -> DemoRegistryView:
    located = find_current_study(root=root)
    bundles = list_bundles(root=root)
    active_set = bundle_runtime_cache.get() if root is None else None
    active = None if active_set is None else active_set.pointer
    freeze = None
    final = None
    if located is not None:
        freeze = _current_freeze(located[1].artifact_id, root=root)
        if freeze is not None:
            final = _current_final_evaluation(freeze.freeze_id, root=root)
    active_ids = (
        set()
        if active is None
        else {active.local_bundle_id, active.network_bundle_id}
    )
    summaries = tuple(
        DemoBundleSummary(
            bundle_id=bundle.bundle_id,
            task_id=bundle.task_id,
            profile_id=bundle.profile_id,
            theta_candidate_name=bundle.theta_candidate.candidate_name,
            theta_id=bundle.afse.theta_id,
            accepted_qng_updates=bundle.theta_candidate.accepted_updates,
            afse_method_id=bundle.afse.method_id,
            afse_reference_size=bundle.afse.reference_size,
            afse_ridge_lambda=bundle.afse.ridge_lambda,
            classifier_model_id=bundle.classifier.model_id,
            class_order=bundle.class_order,
            validation=_metric_summary(bundle.validation_metrics),
            raw_baseline_validation=_metric_summary(
                bundle.raw_baseline_validation_metrics
            ),
            active=bundle.bundle_id in active_ids,
        )
        for bundle in bundles
    )
    prepared = located is not None and freeze is not None and final is not None
    detail = (
        "Prepared study, frozen selection and single TEST evaluation are available."
        if prepared
        else "Run make prepare-demo once to create the bounded research artifacts."
    )
    return DemoRegistryView(
        prepared=prepared,
        study_artifact_id=None if located is None else located[1].artifact_id,
        study_content_digest=None if located is None else located[1].content_digest,
        selection_freeze_id=None if freeze is None else freeze.freeze_id,
        final_evaluation_id=None if final is None else final.evaluation_id,
        historical_test_ledger_sha256=HISTORICAL_TEST_LEDGER_SHA256,
        active=active,
        bundles=summaries,
        final_metrics=(
            ()
            if final is None
            else tuple(_metric_summary(metric) for metric in final.task_metrics)
        ),
        preparation_detail=detail,
    )
