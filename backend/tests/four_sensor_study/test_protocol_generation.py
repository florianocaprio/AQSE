from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from app.features.state8_models import NETWORK_STATE8_PROFILE_ID, STATE8_WINDOW_SAMPLES
from app.four_sensor_study.generation import (
    build_canonical_plans,
    build_pilot_plans,
    generate_episode,
)
from app.four_sensor_study.models import FourSensorStudyObservation
from app.four_sensor_study.protocol import (
    FROZEN_FOUR_SENSOR_STUDY_PROTOCOL,
    GEOMETRY_IDS,
    SCENARIOS,
    SENSOR_IDS,
)
from app.network.models import NetworkSessionConfiguration
from app.training.canonical import canonical_json_bytes

FIXTURE_PROTOCOL_FREEZE_DIGEST = hashlib.sha256(
    b"aqse-four-sensor-study pytest protocol freeze"
).hexdigest()


@pytest.fixture(scope="module")
def pilot_plans():
    return build_pilot_plans()


@pytest.fixture(scope="module")
def canonical_plans():
    return build_canonical_plans(
        protocol_freeze_digest=FIXTURE_PROTOCOL_FREEZE_DIGEST,
    )


def _assert_exact_four_sensor_configuration(plan) -> None:
    configuration = NetworkSessionConfiguration.model_validate(
        plan.network_configuration
    )
    configured_sensor_ids = tuple(node.sensor_id for node in configuration.nodes)

    assert plan.node_count == 4
    assert len(configured_sensor_ids) == 4
    assert set(configured_sensor_ids) == set(SENSOR_IDS)
    assert plan.focal_sensor_id in configured_sensor_ids


def test_protocol_digest_is_deterministic_and_covers_the_frozen_payload() -> None:
    protocol = FROZEN_FOUR_SENSOR_STUDY_PROTOCOL
    expected = hashlib.sha256(
        canonical_json_bytes(protocol.model_dump(mode="json"))
    ).hexdigest()

    assert protocol.digest == expected
    assert protocol.model_copy().digest == expected
    assert len(expected) == 64
    assert protocol.pilot.total_episodes == 20
    assert protocol.canonical.total_episodes == 100
    assert protocol.canonical.partition == "test"
    assert protocol.canonical.fit_policy == "no-fit-no-selection-no-refit"


def test_pilot_has_twenty_independent_balanced_noncanonical_plans(
    pilot_plans,
    canonical_plans,
) -> None:
    assert len(pilot_plans) == 20
    assert Counter(plan.scenario for plan in pilot_plans) == {
        scenario: 5 for scenario in SCENARIOS
    }
    assert Counter(plan.partition for plan in pilot_plans) == {"pilot": 20}
    assert Counter(plan.geometry_id for plan in pilot_plans) == {
        geometry_id: 5 for geometry_id in GEOMETRY_IDS
    }
    assert Counter(plan.focal_sensor_id for plan in pilot_plans) == {
        sensor_id: 5 for sensor_id in SENSOR_IDS
    }
    assert len({plan.episode_id for plan in pilot_plans}) == 20
    assert len({plan.generative_lineage_id for plan in pilot_plans}) == 20
    assert len({plan.episode_seed for plan in pilot_plans}) == 20

    pilot_domains = {plan.generation_domain for plan in pilot_plans}
    canonical_domains = {plan.generation_domain for plan in canonical_plans}
    assert pilot_domains == {
        FROZEN_FOUR_SENSOR_STUDY_PROTOCOL.pilot_generation_domain_separator
    }
    assert canonical_domains == {
        FROZEN_FOUR_SENSOR_STUDY_PROTOCOL.canonical_generation_domain_separator
    }
    assert pilot_domains.isdisjoint(canonical_domains)
    assert {plan.episode_id for plan in pilot_plans}.isdisjoint(
        plan.episode_id for plan in canonical_plans
    )
    assert {plan.generative_lineage_id for plan in pilot_plans}.isdisjoint(
        plan.generative_lineage_id for plan in canonical_plans
    )
    assert {plan.episode_seed for plan in pilot_plans}.isdisjoint(
        plan.episode_seed for plan in canonical_plans
    )


def test_final_plans_cover_exactly_one_hundred_test_episodes(
    canonical_plans,
) -> None:
    assert len(canonical_plans) == 100
    assert Counter(plan.partition for plan in canonical_plans) == {"test": 100}
    assert Counter(plan.scenario for plan in canonical_plans) == {
        scenario: 25 for scenario in SCENARIOS
    }
    assert Counter(plan.geometry_id for plan in canonical_plans) == {
        geometry_id: 25 for geometry_id in GEOMETRY_IDS
    }
    assert Counter(plan.focal_sensor_id for plan in canonical_plans) == {
        sensor_id: 25 for sensor_id in SENSOR_IDS
    }
    global_geometry_focal_counts = Counter(
        (plan.geometry_id, plan.focal_sensor_id) for plan in canonical_plans
    )
    assert len(global_geometry_focal_counts) == len(GEOMETRY_IDS) * len(SENSOR_IDS)
    assert max(global_geometry_focal_counts.values()) - min(
        global_geometry_focal_counts.values()
    ) <= 1
    assert len({plan.episode_id for plan in canonical_plans}) == 100
    assert len({plan.generative_lineage_id for plan in canonical_plans}) == 100
    assert len({plan.episode_seed for plan in canonical_plans}) == 100
    assert all(
        plan.protocol_freeze_digest == FIXTURE_PROTOCOL_FREEZE_DIGEST
        for plan in canonical_plans
    )

    for scenario in SCENARIOS:
        scenario_plans = tuple(
            plan for plan in canonical_plans if plan.scenario == scenario
        )
        geometry_counts = Counter(plan.geometry_id for plan in scenario_plans)
        focal_counts = Counter(plan.focal_sensor_id for plan in scenario_plans)
        geometry_focal_counts = Counter(
            (plan.geometry_id, plan.focal_sensor_id) for plan in scenario_plans
        )

        assert set(geometry_counts) == set(GEOMETRY_IDS)
        assert set(focal_counts) == set(SENSOR_IDS)
        assert set(geometry_focal_counts) == {
            (geometry_id, sensor_id)
            for geometry_id in GEOMETRY_IDS
            for sensor_id in SENSOR_IDS
        }
        assert max(geometry_counts.values()) - min(geometry_counts.values()) <= 1
        assert max(focal_counts.values()) - min(focal_counts.values()) <= 1
        assert max(geometry_focal_counts.values()) - min(
            geometry_focal_counts.values()
        ) <= 1

    for plan in canonical_plans:
        _assert_exact_four_sensor_configuration(plan)


def test_final_plan_generation_requires_the_published_freeze_digest() -> None:
    with pytest.raises(TypeError):
        build_canonical_plans()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        build_canonical_plans(protocol_freeze_digest="")


def test_predictor_observation_schema_excludes_labels_seed_configuration_and_truth() -> None:
    forbidden = {
        "scenario",
        "label",
        "local_label",
        "network_label",
        "truth",
        "truth_label",
        "generative_truth",
        "episode_seed",
        "generation_seed",
        "network_configuration",
        "events",
    }

    assert forbidden.isdisjoint(FourSensorStudyObservation.model_fields)
    assert forbidden.isdisjoint(
        FourSensorStudyObservation.model_json_schema().get("properties", {})
    )


def test_one_real_pilot_episode_has_a_complete_valid_network_state8_window(
    pilot_plans,
) -> None:
    plan = pilot_plans[0]
    observation, _label = generate_episode(plan)
    feature = observation.network_feature
    domain = FROZEN_FOUR_SENSOR_STUDY_PROTOCOL.simulation

    _assert_exact_four_sensor_configuration(plan)
    assert observation.node_count == 4
    assert feature.profile_id == NETWORK_STATE8_PROFILE_ID
    assert feature.sensor_id == plan.focal_sensor_id
    assert feature.quality.valid_for_quantum
    assert feature.quality.received_sample_count == STATE8_WINDOW_SAMPLES
    assert feature.quality.usable_sample_count == STATE8_WINDOW_SAMPLES
    assert len(feature.source_frame_ids) == STATE8_WINDOW_SAMPLES
    assert feature.start_time_s == domain.supervised_window_start_s
    assert feature.end_exclusive_time_s == domain.supervised_window_end_s
    assert len(feature.peer_sensor_ids) == 3
    assert set(feature.peer_sensor_ids) == set(SENSOR_IDS) - {plan.focal_sensor_id}
    assert len(observation.network_replay_features) == 17
    assert observation.network_replay_features[0].start_time_s == 8.0
    assert observation.network_replay_features[-1].end_exclusive_time_s == 28.0
    assert forbidden_observation_keys(observation) == set()


def forbidden_observation_keys(observation: FourSensorStudyObservation) -> set[str]:
    forbidden = {
        "scenario",
        "label",
        "local_label",
        "network_label",
        "truth",
        "truth_label",
        "episode_seed",
        "network_configuration",
        "events",
    }
    return forbidden.intersection(observation.model_dump(mode="json"))
