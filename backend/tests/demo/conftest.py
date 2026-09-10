from __future__ import annotations

import hashlib

import pytest

from app.demo.protocol import FROZEN_NETWORK_DEMO_PROTOCOL
from app.demo.study import _study_configuration, build_network_study_plans
from app.demo.study_models import (
    REPLAY_WINDOW_STARTS_S,
    NetworkStudyBuild,
    NetworkStudyEpisodePlan,
    NetworkStudyLabel,
    NetworkStudyObservation,
)
from app.features.state8 import state8_profile, state8_profile_fingerprint
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
    State8FeatureQuality,
    State8FeatureRecord,
)

FIXTURE_GENERATION_DOMAIN = "aqse-network-demo-v1/test-fixture-generation/v1"


@pytest.fixture(scope="session")
def network_study_plans() -> tuple[NetworkStudyEpisodePlan, ...]:
    return build_network_study_plans()


@pytest.fixture(scope="session")
def real_fixture_episode_plan() -> NetworkStudyEpisodePlan:
    """Non-canonical TRAIN-only plan for exercising the real simulator path."""

    seed = 91_000_001
    configuration, amplitude, focal_extra, focal = _study_configuration(
        scenario="MIXED_OR_AMBIGUOUS",
        node_count=3,
        replicate=1,
        episode_seed=seed,
        generation_domain=FIXTURE_GENERATION_DOMAIN,
    )
    return NetworkStudyEpisodePlan(
        episode_id="network-episode-1111111111111111",
        generative_lineage_id="network-lineage-1111111111111111",
        generation_domain=FIXTURE_GENERATION_DOMAIN,
        partition="train",
        scenario="MIXED_OR_AMBIGUOUS",
        node_count=3,
        replicate_index=1,
        episode_seed=seed,
        focal_sensor_id=focal,
        perturbation_north_nt=amplitude,
        focal_extra_nt=focal_extra,
        network_configuration=configuration.model_dump(mode="json"),
    )


def _feature_record(
    plan: NetworkStudyEpisodePlan,
    *,
    network: bool,
    start_time_s: float = 18.0,
) -> State8FeatureRecord:
    profile_id = NETWORK_STATE8_PROFILE_ID if network else LOCAL_STATE8_PROFILE_ID
    profile = state8_profile(profile_id)
    peer_ids = tuple(
        sensor_id
        for sensor_id in (f"S{index}" for index in range(1, plan.node_count + 1))
        if sensor_id != plan.focal_sensor_id
    )
    reference_suffix = hashlib.sha256(plan.episode_id.encode("utf-8")).hexdigest()[:16]
    return State8FeatureRecord(
        window_id=(
            f"{plan.episode_id}:{profile_id}:{plan.focal_sensor_id}:"
            f"{int(start_time_s * 100)}:{int((start_time_s + 4.0) * 100)}"
        ),
        session_id=plan.episode_id,
        sensor_id=plan.focal_sensor_id,
        profile_id=profile_id,
        profile_fingerprint=state8_profile_fingerprint(profile),
        reference_id=f"aqse-state8-reference-{reference_suffix}",
        start_time_s=start_time_s,
        end_exclusive_time_s=start_time_s + 4.0,
        source_frame_ids=tuple(
            range(
                int(start_time_s * 100) + 1,
                int((start_time_s + 4.0) * 100) + 1,
            )
        ),
        peer_sensor_ids=peer_ids if network else (),
        values=(1.0, 0.5, 0.1, -2.0, 0.0, 0.25, 0.75, 1.0),
        quality=State8FeatureQuality(
            valid_for_quantum=True,
            flags=(),
            per_feature_valid=(True, True, True, True, True, True, True, True),
            received_sample_count=400,
            usable_sample_count=400,
        ),
    )


@pytest.fixture(scope="session")
def fixture_network_study_build(
    network_study_plans: tuple[NetworkStudyEpisodePlan, ...],
) -> NetworkStudyBuild:
    observations: list[NetworkStudyObservation] = []
    labels: list[NetworkStudyLabel] = []
    for plan in network_study_plans:
        local_replay = tuple(
            _feature_record(plan, network=False, start_time_s=start)
            for start in REPLAY_WINDOW_STARTS_S
        )
        network_replay = (
            tuple(
                _feature_record(plan, network=True, start_time_s=start)
                for start in REPLAY_WINDOW_STARTS_S
            )
            if plan.node_count >= 3
            else ()
        )
        observations.append(
            NetworkStudyObservation(
                episode_id=plan.episode_id,
                generative_lineage_id=plan.generative_lineage_id,
                partition=plan.partition,
                node_count=plan.node_count,
                focal_sensor_id=plan.focal_sensor_id,
                local_feature=local_replay[10],
                network_feature=(
                    network_replay[10]
                    if plan.node_count >= 3
                    else None
                ),
                local_replay_features=local_replay,
                network_replay_features=network_replay,
            )
        )
        labels.append(
            NetworkStudyLabel(
                episode_id=plan.episode_id,
                generative_lineage_id=plan.generative_lineage_id,
                partition=plan.partition,
                local_label=(
                    "NORMAL" if plan.scenario == "NORMAL" else "CHANGE_DETECTED"
                ),
                network_label=plan.scenario,
            )
        )
    return NetworkStudyBuild(
        protocol_digest=FROZEN_NETWORK_DEMO_PROTOCOL.digest,
        episode_plans=network_study_plans,
        observations=tuple(observations),
        labels=tuple(labels),
    )
