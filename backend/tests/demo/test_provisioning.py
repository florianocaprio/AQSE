from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.demo import provisioning


def test_validation_preflight_failure_leaves_new_test_ledger_untouched(
    tmp_path,
    monkeypatch,
) -> None:
    study_path = tmp_path / "network-demo" / "studies" / "study-fixture"
    study_path.mkdir(parents=True)
    ledger_path = study_path / "test-access-ledger.jsonl"
    original = b"sealed-new-test-ledger\n"
    ledger_path.write_bytes(original)
    manifest = SimpleNamespace(artifact_id="study-fixture")
    freeze = SimpleNamespace(freeze_id="aqse-demo-freeze-aaaaaaaaaaaaaaaa")

    monkeypatch.setattr(
        provisioning,
        "find_current_study",
        lambda **_kwargs: (study_path, manifest),
    )
    monkeypatch.setattr(
        provisioning,
        "_current_freeze",
        lambda *_args, **_kwargs: freeze,
    )

    def fail_validation(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("selected bundle is missing before TEST access")

    monkeypatch.setattr(
        provisioning,
        "_validated_selected_bundles",
        fail_validation,
    )

    with pytest.raises(RuntimeError, match="missing before TEST"):
        provisioning.prepare_demo(root=tmp_path)

    assert ledger_path.read_bytes() == original


def test_pretest_validation_rejects_freeze_not_bound_to_study(monkeypatch) -> None:
    manifest = SimpleNamespace(
        artifact_id="aqse-network-study-1111111111111111",
        content_digest="1" * 64,
        protocol_digest=provisioning.FROZEN_NETWORK_DEMO_PROTOCOL.digest,
    )
    freeze = SimpleNamespace(
        study_artifact_id=manifest.artifact_id,
        study_content_digest="2" * 64,
        protocol_digest=manifest.protocol_digest,
    )
    monkeypatch.setattr(
        provisioning,
        "list_bundles",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not load")),
    )

    with pytest.raises(RuntimeError, match="not bound"):
        provisioning._validated_selected_bundles(
            SimpleNamespace(),
            manifest,
            freeze,
        )


def test_pretest_validation_replays_raw_baseline_and_comparison(monkeypatch) -> None:
    protocol_digest = provisioning.FROZEN_NETWORK_DEMO_PROTOCOL.digest
    manifest = SimpleNamespace(
        artifact_id="aqse-network-study-1111111111111111",
        content_digest="1" * 64,
        protocol_digest=protocol_digest,
    )
    selections = (
        SimpleNamespace(task_id="aqse.local-change.v1", selected_bundle_id="local"),
        SimpleNamespace(task_id="aqse.network-pattern.v1", selected_bundle_id="network"),
    )
    freeze = SimpleNamespace(
        study_artifact_id=manifest.artifact_id,
        study_content_digest=manifest.content_digest,
        protocol_digest=protocol_digest,
        selections=selections,
    )
    bundles = tuple(
        SimpleNamespace(
            bundle_id=selection.selected_bundle_id,
            task_id=selection.task_id,
            study_artifact_id=manifest.artifact_id,
            study_content_digest=manifest.content_digest,
            protocol_digest=protocol_digest,
            validation_metrics=f"{selection.task_id}:afse",
            raw_baseline_validation_metrics=f"{selection.task_id}:raw",
            validation_comparison=f"{selection.task_id}:comparison",
        )
        for selection in selections
    )
    partitions = tuple(
        SimpleNamespace(task_id=selection.task_id) for selection in selections
    )
    monkeypatch.setattr(provisioning, "list_bundles", lambda **_kwargs: bundles)
    monkeypatch.setattr(
        provisioning,
        "load_task_partitions",
        lambda *_args, **_kwargs: partitions,
    )

    def replay(bundle, _partition, **_kwargs):  # type: ignore[no-untyped-def]
        return (
            bundle.validation_metrics,
            bundle.raw_baseline_validation_metrics,
            "tampered-comparison",
        )

    monkeypatch.setattr(provisioning, "evaluate_bundle_evidence", replay)

    with pytest.raises(RuntimeError, match="complete frozen VALIDATION"):
        provisioning._validated_selected_bundles(
            SimpleNamespace(),
            manifest,
            freeze,
        )


def test_persisted_final_requires_complete_matching_test_ledger(monkeypatch) -> None:
    manifest = SimpleNamespace(
        artifact_id="aqse-network-study-1111111111111111",
        content_digest="1" * 64,
    )
    freeze = SimpleNamespace(freeze_id="aqse-demo-freeze-aaaaaaaaaaaaaaaa")
    final = SimpleNamespace(
        freeze_id=freeze.freeze_id,
        study_artifact_id=manifest.artifact_id,
        study_content_digest=manifest.content_digest,
        query_acquisition_id=(
            f"{manifest.artifact_id}:single-frozen-test-evaluation"
        ),
    )
    matching = tuple(
        SimpleNamespace(
            sequence=index,
            selection_freeze_id=None if index == 0 else freeze.freeze_id,
        )
        for index in range(3)
    )
    monkeypatch.setattr(
        provisioning,
        "read_demo_test_ledger",
        lambda _path: matching,
    )

    provisioning._validate_final_and_ledger(
        SimpleNamespace(), manifest, freeze, final
    )

    monkeypatch.setattr(
        provisioning,
        "read_demo_test_ledger",
        lambda _path: matching[:2],
    )
    with pytest.raises(RuntimeError, match="complete matching TEST access ledger"):
        provisioning._validate_final_and_ledger(
            SimpleNamespace(), manifest, freeze, final
        )


def test_persisted_final_rejects_study_or_freeze_identity_mismatch(monkeypatch) -> None:
    manifest = SimpleNamespace(
        artifact_id="aqse-network-study-1111111111111111",
        content_digest="1" * 64,
    )
    freeze = SimpleNamespace(freeze_id="aqse-demo-freeze-aaaaaaaaaaaaaaaa")
    final = SimpleNamespace(
        freeze_id="aqse-demo-freeze-bbbbbbbbbbbbbbbb",
        study_artifact_id=manifest.artifact_id,
        study_content_digest=manifest.content_digest,
        query_acquisition_id=(
            f"{manifest.artifact_id}:single-frozen-test-evaluation"
        ),
    )
    monkeypatch.setattr(
        provisioning,
        "read_demo_test_ledger",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("ledger must not be read after identity mismatch")
        ),
    )

    with pytest.raises(RuntimeError, match="not bound"):
        provisioning._validate_final_and_ledger(
            SimpleNamespace(), manifest, freeze, final
        )


def test_selection_freeze_summary_retains_reloadable_bundle_pair_identity() -> None:
    freeze = SimpleNamespace(
        freeze_id="aqse-demo-freeze-aaaaaaaaaaaaaaaa",
        study_artifact_id="aqse-knowledge-study-example",
        study_content_digest="1" * 64,
        selections=(
            SimpleNamespace(
                task_id="aqse.local-change.v1",
                selected_bundle_id="aqse-demo-bundle-1111111111111111",
            ),
            SimpleNamespace(
                task_id="aqse.network-pattern.v1",
                selected_bundle_id="aqse-demo-bundle-2222222222222222",
            ),
        ),
    )

    summary = provisioning._selection_freeze_summary(freeze)

    assert summary.freeze_id == freeze.freeze_id
    assert summary.local_bundle_id == "aqse-demo-bundle-1111111111111111"
    assert summary.network_bundle_id == "aqse-demo-bundle-2222222222222222"
