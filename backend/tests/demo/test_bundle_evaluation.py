from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.demo.bundle_models import (
    LOCAL_TASK_ID,
    NETWORK_TASK_ID,
    BundleQueryContext,
    BundleRuntime,
    DemoModelSelectionFreeze,
    DemoTaskExample,
    DemoTaskPartition,
    TaskSelection,
    canonical_digest,
)
from app.demo.bundle_storage import (
    apply_selection_freeze,
    list_bundles,
    load_active_bundle_set,
    load_bundle,
    load_final_evaluation,
    load_selection_freeze,
    write_bundle,
    write_final_evaluation,
    write_selection_freeze,
)
from app.demo.evaluation import (
    DemoTaskFitResult,
    evaluate_frozen_test,
    fit_demo_task_candidates,
    freeze_model_selection,
    select_task_candidate,
)
from app.demo.training import DemoQngCandidate, canonical_demo_theta0
from app.features.state8 import state8_profile, state8_profile_fingerprint

STUDY_ID = "aqse-network-study-fixture"
STUDY_DIGEST = "a" * 64


def _partition(
    task_id: str,
    partition: str,
    *,
    rows_per_class: int,
    ordinal_offset: int,
) -> DemoTaskPartition:
    local = task_id == LOCAL_TASK_ID
    profile_id = "aqse.local-state8.v1" if local else "aqse.network-state8.v1"
    classes = (
        ("NORMAL", "CHANGE_DETECTED")
        if local
        else (
            "NORMAL",
            "ENVIRONMENT_COMPATIBLE",
            "DEVICE_COMPATIBLE",
            "MIXED_OR_AMBIGUOUS",
        )
    )
    generator = np.random.default_rng(900 + ordinal_offset + (0 if local else 100))
    examples: list[DemoTaskExample] = []
    for class_index, label in enumerate(classes):
        center = -1.5 + class_index
        for row_index in range(rows_per_class):
            ordinal = ordinal_offset + (class_index * rows_per_class) + row_index
            features = generator.normal(center, 0.12, size=8)
            # Preserve the physical State8 domains used by the observable rule:
            # dispersion and the network spatial residual are non-negative.
            features[1] = abs(features[1])
            if not local:
                features[5] = abs(features[5])
            examples.append(
                DemoTaskExample(
                    sample_id=f"{task_id}-{partition}-{ordinal:03d}",
                    episode_id=f"episode-{task_id}-{partition}-{ordinal:03d}",
                    generative_lineage_id=(
                        f"lineage-{task_id}-{partition}-{ordinal:03d}"
                    ),
                    node_count=(1 + ordinal % 8) if local else (3 + ordinal % 6),
                    label=label,
                    feature_values=tuple(float(value) for value in features),
                    eligible=True,
                )
            )
    profile = state8_profile(profile_id)
    return DemoTaskPartition(
        study_artifact_id=STUDY_ID,
        study_content_digest=STUDY_DIGEST,
        partition=partition,
        task_id=task_id,
        profile_id=profile_id,
        profile_fingerprint=state8_profile_fingerprint(profile),
        class_order=classes,
        examples=tuple(examples),
    )


def _fake_qng(
    encoded_features,
    binary_labels,
    *,
    sample_ids,
    lineage_ids,
    profile_id,
    **_kwargs,
) -> DemoQngCandidate:
    assert encoded_features.shape[1] == 8
    assert set(binary_labels) == {-1, 1}
    theta0 = canonical_demo_theta0()
    final = theta0 + np.linspace(0.001, 0.016, 16)
    return DemoQngCandidate(
        candidate_id=f"fixture-qng-{profile_id}",
        eligible=True,
        profile_id=profile_id,
        bank_sample_ids=tuple(sample_ids),
        bank_lineage_ids=tuple(lineage_ids),
        theta0=tuple(float(value) for value in theta0),
        final_theta=tuple(float(value) for value in final),
        accepted_updates=1,
        stop_reason="FIXTURE_ACCEPTED_UPDATE",
        initial_loss=0.5,
        final_loss=0.4,
        steps=(),
        total_wall_time_ms=1.0,
    )


@pytest.fixture(scope="module")
def fitted_results() -> tuple[
    DemoTaskFitResult,
    DemoTaskFitResult,
    DemoTaskPartition,
    DemoTaskPartition,
]:
    local_train = _partition(
        LOCAL_TASK_ID,
        "train",
        rows_per_class=4,
        ordinal_offset=0,
    )
    local_validation = _partition(
        LOCAL_TASK_ID,
        "validation",
        rows_per_class=2,
        ordinal_offset=100,
    )
    network_train = _partition(
        NETWORK_TASK_ID,
        "train",
        rows_per_class=4,
        ordinal_offset=200,
    )
    network_validation = _partition(
        NETWORK_TASK_ID,
        "validation",
        rows_per_class=2,
        ordinal_offset=300,
    )
    with patch("app.demo.evaluation.train_demo_qng", _fake_qng):
        local = fit_demo_task_candidates(local_train, local_validation)
        network = fit_demo_task_candidates(network_train, network_validation)
    return local, network, local_validation, network_validation


def test_two_complete_candidates_use_validation_only_and_theta0_tie_break(
    fitted_results,
) -> None:
    local, network, _, _ = fitted_results

    assert [item.theta_candidate.candidate_name for item in local.bundles] == [
        "theta0",
        "protected_qng",
    ]
    assert [item.theta_candidate.candidate_name for item in network.bundles] == [
        "theta0",
        "protected_qng",
    ]
    assert all(item.fitted_on_partition == "TRAIN" for item in (*local.bundles, *network.bundles))
    assert all(
        item.validation_metrics.partition == "validation"
        for item in (*local.bundles, *network.bundles)
    )
    assert all(
        item.raw_feature_baseline.model_role == "raw_feature_baseline"
        for item in (*local.bundles, *network.bundles)
    )

    qng = local.bundles[1]
    tied_qng = qng.model_copy(
        update={"validation_metrics": local.bundles[0].validation_metrics}
    )
    tied = DemoTaskFitResult(
        task_id=LOCAL_TASK_ID,
        train_dataset_id=local.train_dataset_id,
        train_dataset_digest=local.train_dataset_digest,
        bundles=(tied_qng, local.bundles[0]),
        qng_candidate=local.qng_candidate,
    )
    assert select_task_candidate(tied).selected_candidate_name == "theta0"


def test_bundle_runtime_allows_new_acquisition_but_rejects_incompatible_space(
    fitted_results,
) -> None:
    local, _, local_validation, _ = fitted_results
    bundle = local.bundles[0]
    runtime = BundleRuntime.from_artifact(bundle)
    features = np.asarray(
        [item.feature_values for item in local_validation.examples],
        dtype=np.float64,
    )
    sample_ids = tuple(item.sample_id for item in local_validation.examples)
    context = BundleQueryContext(
        bundle_id=bundle.bundle_id,
        task_id=bundle.task_id,
        profile_id=bundle.profile_id,
        profile_fingerprint=bundle.profile_fingerprint,
        query_acquisition_id="live-session-that-is-not-the-training-dataset",
    )

    result = runtime.predict(features, sample_ids=sample_ids, context=context)

    assert result.query_acquisition_id != bundle.fitted_on_dataset_id
    assert result.sample_ids == sample_ids
    assert len(result.afse_vectors) == len(features)
    wrong = context.model_copy(update={"bundle_id": local.bundles[1].bundle_id})
    with pytest.raises(ValueError, match="incompatible"):
        runtime.predict(features, sample_ids=sample_ids, context=wrong)


def test_atomic_registry_reload_idempotence_and_symlink_rejection(
    tmp_path: Path,
    fitted_results,
) -> None:
    local, network, _, _ = fitted_results
    freeze = freeze_model_selection(local, network)
    for bundle in (*local.bundles, *network.bundles):
        _, reused = write_bundle(bundle, root=tmp_path)
        assert not reused
        _, reused = write_bundle(bundle, root=tmp_path)
        assert reused
    freeze_path, reused = write_selection_freeze(freeze, root=tmp_path)
    assert not reused
    assert load_selection_freeze(freeze_path) == freeze
    assert len(list_bundles(root=tmp_path)) == 4

    first = apply_selection_freeze(freeze, root=tmp_path)
    second = apply_selection_freeze(freeze, root=tmp_path)
    assert second == first
    assert first.generation == first.revision == 1

    reloaded = load_active_bundle_set(root=tmp_path)
    assert reloaded is not None
    assert reloaded.pointer == first
    assert reloaded.local.artifact.bundle_id == freeze.selections[0].selected_bundle_id
    assert reloaded.network.artifact.bundle_id == freeze.selections[1].selected_bundle_id

    link = tmp_path / "bundle-link"
    link.symlink_to(
        tmp_path
        / "network-demo"
        / "registry"
        / "bundles"
        / local.bundles[0].bundle_id,
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="real directory"):
        load_bundle(link)


def test_application_rejects_old_or_foreign_bundle(
    tmp_path: Path,
    fitted_results,
) -> None:
    local, network, _, _ = fitted_results
    original_freeze = freeze_model_selection(local, network)
    selected_local = next(
        item
        for item in local.bundles
        if item.bundle_id == original_freeze.selections[0].selected_bundle_id
    )
    foreign_payload = selected_local.model_dump(
        mode="json",
        exclude={"bundle_id", "content_digest"},
    )
    foreign_payload["study_artifact_id"] = "aqse-network-study-old"
    foreign_digest = canonical_digest(foreign_payload)
    foreign = type(selected_local).model_validate(
        {
            **foreign_payload,
            "bundle_id": f"aqse-demo-bundle-{foreign_digest[:16]}",
            "content_digest": foreign_digest,
        }
    )
    for bundle in (*local.bundles, *network.bundles, foreign):
        write_bundle(bundle, root=tmp_path)
    local_selection = original_freeze.selections[0]
    foreign_evidence = tuple(
        item.model_copy(
            update={"bundle_id": foreign.bundle_id}
            if item.bundle_id == local_selection.selected_bundle_id
            else {}
        )
        for item in local_selection.candidate_evidence
    )
    foreign_selection = TaskSelection(
        task_id=LOCAL_TASK_ID,
        candidate_evidence=foreign_evidence,
        selected_bundle_id=foreign.bundle_id,
        selected_candidate_name=local_selection.selected_candidate_name,
    )
    freeze_payload = original_freeze.model_dump(
        mode="json",
        exclude={"freeze_id", "content_digest"},
    )
    freeze_payload["selections"][0] = foreign_selection.model_dump(mode="json")
    freeze_digest = canonical_digest(freeze_payload)
    foreign_freeze = DemoModelSelectionFreeze.model_validate(
        {
            **freeze_payload,
            "freeze_id": f"aqse-demo-freeze-{freeze_digest[:16]}",
            "content_digest": freeze_digest,
        }
    )
    write_selection_freeze(foreign_freeze, root=tmp_path)

    with pytest.raises(ValueError, match="old or incompatible"):
        apply_selection_freeze(foreign_freeze, root=tmp_path)


def test_final_test_evaluation_uses_frozen_selection_without_fitting(
    tmp_path: Path,
    fitted_results,
) -> None:
    local, network, local_validation, network_validation = fitted_results
    freeze = freeze_model_selection(local, network)
    bundles = {item.bundle_id: item for item in (*local.bundles, *network.bundles)}
    local_test = local_validation.model_copy(update={"partition": "test"})
    network_test = network_validation.model_copy(update={"partition": "test"})

    with (
        patch("app.demo.evaluation.fit_mlp_classifier", side_effect=AssertionError),
        patch("app.demo.evaluation.fit_nystrom_afse", side_effect=AssertionError),
        patch("app.demo.evaluation.train_demo_qng", side_effect=AssertionError),
    ):
        final = evaluate_frozen_test(
            freeze,
            bundles,
            local_test=local_test,
            network_test=network_test,
            query_acquisition_id="single-authorized-test-acquisition",
        )

    assert final.freeze_id == freeze.freeze_id
    assert final.selection_changed is False
    assert final.refit_performed is False
    assert [item.task_id for item in final.task_metrics] == [
        LOCAL_TASK_ID,
        NETWORK_TASK_ID,
    ]
    assert all(item.partition == "test" for item in final.task_metrics)
    path, reused = write_final_evaluation(final, root=tmp_path)
    assert not reused
    assert load_final_evaluation(path) == final
    assert write_final_evaluation(final, root=tmp_path) == (path, True)
