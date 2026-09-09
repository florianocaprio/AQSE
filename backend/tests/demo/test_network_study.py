from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

import app.demo.study_storage as study_storage
from app.demo.protocol import FROZEN_NETWORK_DEMO_PROTOCOL, SCENARIOS
from app.demo.study import generate_network_study_episode
from app.demo.study_models import NetworkStudyBuild, NetworkStudyEpisodePlan

EXPECTED_PROTOCOL_DIGEST = (
    "986c362dfe0feed39bba2d959e9904c7c6903f541a010704d3f9c444eff6c099"
)


def _install_opaque_historical_ledger_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> bytes:
    payload = b"opaque historical ledger fixture; no TEST values\n"
    ledger = (
        root
        / study_storage.HISTORICAL_DATASET_ID
        / "test-access-ledger.jsonl"
    )
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(payload)
    monkeypatch.setattr(
        study_storage,
        "HISTORICAL_TEST_LEDGER_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    return payload


def test_protocol_digest_is_deterministic_and_frozen() -> None:
    assert FROZEN_NETWORK_DEMO_PROTOCOL.digest == EXPECTED_PROTOCOL_DIGEST
    assert (
        FROZEN_NETWORK_DEMO_PROTOCOL.model_copy().digest
        == FROZEN_NETWORK_DEMO_PROTOCOL.digest
    )


def test_plans_cover_exactly_160_episodes_and_the_frozen_split(
    network_study_plans: tuple[NetworkStudyEpisodePlan, ...],
) -> None:
    assert len(network_study_plans) == 160
    assert len({plan.episode_id for plan in network_study_plans}) == 160
    assert len({plan.generative_lineage_id for plan in network_study_plans}) == 160
    assert Counter(plan.partition for plan in network_study_plans) == {
        "train": 96,
        "validation": 32,
        "test": 32,
    }
    for scenario in SCENARIOS:
        for node_count in range(1, 9):
            cell = [
                plan
                for plan in network_study_plans
                if plan.scenario == scenario and plan.node_count == node_count
            ]
            assert len(cell) == 5
            assert Counter(plan.partition for plan in cell) == {
                "train": 3,
                "validation": 1,
                "test": 1,
            }


def test_focal_node_selection_is_independent_of_scenario(
    network_study_plans: tuple[NetworkStudyEpisodePlan, ...],
) -> None:
    focal_by_design: dict[tuple[int, int], set[str]] = {}
    for plan in network_study_plans:
        focal_by_design.setdefault(
            (plan.node_count, plan.replicate_index),
            set(),
        ).add(plan.focal_sensor_id)

    assert all(len(focal_ids) == 1 for focal_ids in focal_by_design.values())


def test_fixture_and_canonical_generation_domains_and_identities_are_disjoint(
    network_study_plans: tuple[NetworkStudyEpisodePlan, ...],
    real_fixture_episode_plan: NetworkStudyEpisodePlan,
) -> None:
    canonical_ids = {plan.episode_id for plan in network_study_plans}
    canonical_seeds = {plan.episode_seed for plan in network_study_plans}

    assert real_fixture_episode_plan.generation_domain == (
        "aqse-network-demo-v1/test-fixture-generation/v1"
    )
    assert (
        real_fixture_episode_plan.generation_domain
        != FROZEN_NETWORK_DEMO_PROTOCOL.generation_domain_separator
    )
    assert real_fixture_episode_plan.episode_id not in canonical_ids
    assert real_fixture_episode_plan.episode_seed not in canonical_seeds


def test_generation_plans_cover_sham_spatial_device_and_ambiguous_modes(
    network_study_plans: tuple[NetworkStudyEpisodePlan, ...],
) -> None:
    by_key = {
        (plan.scenario, plan.node_count, plan.replicate_index): plan
        for plan in network_study_plans
    }
    normal = by_key[("NORMAL", 3, 0)].network_configuration
    environment = by_key[("ENVIRONMENT_COMPATIBLE", 3, 0)].network_configuration
    device_event_ids = {
        by_key[("DEVICE_COMPATIBLE", 3, replicate)]
        .network_configuration["events"][0]["event_id"]
        for replicate in range(5)
    }
    ambiguous_common = by_key[("MIXED_OR_AMBIGUOUS", 3, 0)].network_configuration
    ambiguous_shared = by_key[("MIXED_OR_AMBIGUOUS", 3, 1)].network_configuration

    assert normal["events"][0]["event_id"] == "normal-sham-event"
    assert normal["events"][0]["field_offset_T"] == [0.0, 0.0, 0.0]
    moving_source = environment["environment"]["dipoles"][0]
    assert moving_source["source_id"] == "moving-spatial-source"
    assert moving_source["velocity_m_per_s"][0] > 0.0
    assert device_event_ids == {
        "focal-device-bias",
        "focal-device-drift",
        "focal-device-noise",
        "focal-device-thermal-window",
    }
    assert ambiguous_common["events"][0]["kind"] == "world_field_offset"
    assert ambiguous_shared["events"][0]["kind"] == "shared_instrument_offset"
    assert any(
        [node["sensor_id"] for node in plan.network_configuration["nodes"]]
        != sorted(node["sensor_id"] for node in plan.network_configuration["nodes"])
        for plan in network_study_plans
    )


def test_real_episode_produces_one_valid_frozen_local_and_network_window(
    real_fixture_episode_plan: NetworkStudyEpisodePlan,
) -> None:
    observation, label = generate_network_study_episode(real_fixture_episode_plan)

    assert observation.local_feature.quality.valid_for_quantum
    assert observation.network_feature is not None
    assert observation.network_feature.quality.valid_for_quantum
    assert observation.local_feature.start_time_s == 18.0
    assert observation.local_feature.end_exclusive_time_s == 22.0
    assert observation.network_feature.start_time_s == 18.0
    assert observation.network_feature.end_exclusive_time_s == 22.0
    assert label.network_label == "MIXED_OR_AMBIGUOUS"


def test_observation_label_and_generation_channels_are_structurally_separate(
    fixture_network_study_build: NetworkStudyBuild,
) -> None:
    observation = fixture_network_study_build.observations[0].model_dump(mode="json")
    label = fixture_network_study_build.labels[0].model_dump(mode="json")
    generation = fixture_network_study_build.episode_plans[0].model_dump(mode="json")

    assert not {
        "scenario",
        "local_label",
        "network_label",
        "episode_seed",
        "network_configuration",
    } & observation.keys()
    assert not {"local_feature", "network_feature", "episode_seed"} & label.keys()
    assert {"scenario", "episode_seed", "network_configuration"} <= generation.keys()


def test_historical_ledger_hash_check_uses_only_an_opaque_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _install_opaque_historical_ledger_fixture(tmp_path, monkeypatch)

    observed = study_storage.verify_historical_test_ledger(tmp_path)

    assert observed == hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(study_storage, "HISTORICAL_TEST_LEDGER_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="changed"):
        study_storage.verify_historical_test_ledger(tmp_path)


def test_study_storage_is_atomic_idempotent_and_keeps_test_physically_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_network_study_build: NetworkStudyBuild,
) -> None:
    _install_opaque_historical_ledger_fixture(tmp_path, monkeypatch)

    path, manifest, reused = study_storage.write_network_study(
        fixture_network_study_build,
        root=tmp_path,
    )
    first_files = {
        item.name: item.read_bytes()
        for item in path.iterdir()
        if item.name != "test-access-ledger.jsonl"
    }
    repeated_path, repeated_manifest, repeated = study_storage.write_network_study(
        fixture_network_study_build,
        root=tmp_path,
    )

    assert not reused
    assert repeated
    assert repeated_path == path
    assert repeated_manifest == manifest
    assert first_files == {
        item.name: item.read_bytes()
        for item in repeated_path.iterdir()
        if item.name != "test-access-ledger.jsonl"
    }
    assert {item.name for item in path.iterdir()} == study_storage.STUDY_FILES
    assert not tuple(path.parent.glob(f".{manifest.artifact_id}.*"))
    assert study_storage.verify_network_study(path) == manifest


def test_development_loader_never_decodes_test_and_generic_test_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_network_study_build: NetworkStudyBuild,
) -> None:
    _install_opaque_historical_ledger_fixture(tmp_path, monkeypatch)
    path, _, _ = study_storage.write_network_study(
        fixture_network_study_build,
        root=tmp_path,
    )
    decoded_observation_partitions: list[str] = []
    decoded_label_partitions: list[str] = []
    original_observations = study_storage._decode_observations
    original_labels = study_storage._decode_labels

    def observe_partition(path: Path, partition: study_storage.StudyPartition):
        decoded_observation_partitions.append(partition)
        return original_observations(path, partition)

    def observe_labels(path: Path, partition: study_storage.StudyPartition):
        decoded_label_partitions.append(partition)
        return original_labels(path, partition)

    monkeypatch.setattr(study_storage, "_decode_observations", observe_partition)
    monkeypatch.setattr(study_storage, "_decode_labels", observe_labels)

    loaded = study_storage.load_development_partition(
        path,
        "train",
        include_labels=True,
    )

    assert len(loaded.observations) == len(loaded.labels or ()) == 96
    assert decoded_observation_partitions == ["train"]
    assert decoded_label_partitions == ["train"]
    with pytest.raises(PermissionError, match="one-time frozen access"):
        study_storage.load_development_partition(path, "test", include_labels=False)
    assert decoded_observation_partitions == ["train"]
    assert decoded_label_partitions == ["train"]


def test_new_study_test_ledger_advances_once_from_one_to_two_to_three_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_network_study_build: NetworkStudyBuild,
) -> None:
    _install_opaque_historical_ledger_fixture(tmp_path, monkeypatch)
    path, _, _ = study_storage.write_network_study(
        fixture_network_study_build,
        root=tmp_path,
    )
    freeze_id = "aqse-network-selection-freeze-fixture"

    genesis = study_storage.read_demo_test_ledger(path)
    observations = study_storage.load_new_test_observations(
        path,
        selection_freeze_id=freeze_id,
    )
    after_observations = study_storage.read_demo_test_ledger(path)
    labels = study_storage.load_new_test_labels(
        path,
        selection_freeze_id=freeze_id,
    )
    final = study_storage.read_demo_test_ledger(path)

    assert len(genesis) == 1
    assert len(observations.observations) == 32
    assert len(after_observations) == 2
    assert len(labels) == 32
    assert len(final) == 3
    assert tuple(entry.sequence for entry in final) == (0, 1, 2)
    assert final[1].previous_entry_sha256 == final[0].entry_sha256
    assert final[2].previous_entry_sha256 == final[1].entry_sha256
    assert final[1].selection_freeze_id == final[2].selection_freeze_id == freeze_id
    with pytest.raises(PermissionError, match="already advanced"):
        study_storage.load_new_test_observations(
            path,
            selection_freeze_id=freeze_id,
        )
