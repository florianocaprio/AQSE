from __future__ import annotations

import pytest

from app.training.generation import DatasetBuild, build_episode_plans, generate_episode
from app.training.models import DatasetPartition, SplitAssignment


@pytest.fixture(scope="session")
def generated_episodes():
    specifications = build_episode_plans(
        master_seed=9191,
        episodes_per_class=2,
        purpose="test-fixture",
    )
    return tuple(generate_episode(item) for item in specifications)


@pytest.fixture(scope="session")
def small_pilot_build(generated_episodes) -> DatasetBuild:
    assignments = tuple(
        SplitAssignment(
            episode_id=episode.plan.episode_id,
            lineage_id=episode.plan.lineage_id,
            partition=DatasetPartition.PILOT,
        )
        for episode in generated_episodes
    )
    return DatasetBuild(
        kind="pilot",
        master_seed=9191,
        split_seed=None,
        episodes=generated_episodes,
        assignments=assignments,
    )


@pytest.fixture(scope="session")
def small_development_build(generated_episodes) -> DatasetBuild:
    partitions = (
        DatasetPartition.TRAIN,
        DatasetPartition.TRAIN,
        DatasetPartition.VALIDATION,
        DatasetPartition.TEST,
    )
    assignments = tuple(
        SplitAssignment(
            episode_id=episode.plan.episode_id,
            lineage_id=episode.plan.lineage_id,
            partition=partition,
        )
        for episode, partition in zip(generated_episodes, partitions, strict=True)
    )
    return DatasetBuild(
        kind="development",
        master_seed=9191,
        split_seed=9292,
        episodes=generated_episodes,
        assignments=assignments,
    )
