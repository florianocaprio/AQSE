from __future__ import annotations

from collections import Counter

import pytest

from app.training.generation import build_episode_plans
from app.training.models import (
    DatasetPartition,
    ObservationInterval,
    SplitAssignment,
)
from app.training.splits import (
    extend_assignments_by_lineage,
    stratified_lineage_split,
    validate_no_cross_partition_duplicates,
    validate_no_cross_partition_raw_overlaps,
)


def test_split_is_exact_stratified_and_lineage_grouped() -> None:
    labels = tuple(
        item.label
        for item in build_episode_plans(
            master_seed=100,
            episodes_per_class=60,
            purpose="split",
        )
    )
    assignments = stratified_lineage_split(labels, split_seed=101)
    targets = {item.episode_id: item.target for item in labels}
    counts = Counter((item.partition, targets[item.episode_id]) for item in assignments)
    assert counts == {
        (DatasetPartition.TRAIN, -1): 36,
        (DatasetPartition.TRAIN, 1): 36,
        (DatasetPartition.VALIDATION, -1): 12,
        (DatasetPartition.VALIDATION, 1): 12,
        (DatasetPartition.TEST, -1): 12,
        (DatasetPartition.TEST, 1): 12,
    }
    assert assignments == stratified_lineage_split(labels, split_seed=101)


def test_replays_and_paired_descendants_follow_parent_lineage() -> None:
    base = (
        SplitAssignment(
            episode_id="episode-0000000000000001",
            lineage_id="lineage-0000000000000001",
            partition=DatasetPartition.VALIDATION,
        ),
    )
    extended = extend_assignments_by_lineage(
        base,
        {"episode-0000000000000002": "lineage-0000000000000001"},
    )
    assert {item.partition for item in extended} == {DatasetPartition.VALIDATION}


def test_duplicate_observations_across_partitions_are_rejected() -> None:
    assignments = (
        SplitAssignment(episode_id="a", lineage_id="la", partition="train"),
        SplitAssignment(episode_id="b", lineage_id="lb", partition="test"),
    )
    with pytest.raises(ValueError, match="duplicate observation"):
        validate_no_cross_partition_duplicates(assignments, {"a": "same", "b": "same"})


def test_raw_overlap_is_lineage_aware_and_relative_time_can_repeat() -> None:
    assignments = (
        SplitAssignment(episode_id="a", lineage_id="la", partition="train"),
        SplitAssignment(episode_id="b", lineage_id="lb", partition="test"),
    )
    independent = (
        ObservationInterval(
            episode_id="a", lineage_id="la", sensor_id="M1", start_index=0, end_index=100
        ),
        ObservationInterval(
            episode_id="b", lineage_id="lb", sensor_id="M1", start_index=0, end_index=100
        ),
    )
    validate_no_cross_partition_raw_overlaps(assignments, independent)

    leaking_assignments = (
        assignments[0],
        SplitAssignment(episode_id="b", lineage_id="la", partition="test"),
    )
    leaking_intervals = (
        independent[0],
        ObservationInterval(
            episode_id="b", lineage_id="la", sensor_id="M1", start_index=50, end_index=150
        ),
    )
    with pytest.raises(ValueError, match="overlapping raw intervals"):
        validate_no_cross_partition_raw_overlaps(leaking_assignments, leaking_intervals)
