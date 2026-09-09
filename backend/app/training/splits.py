from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from app.training.models import (
    DatasetPartition,
    EpisodeLabel,
    ObservationInterval,
    SplitAssignment,
)

EXPECTED_PER_CLASS = {
    DatasetPartition.TRAIN: 36,
    DatasetPartition.VALIDATION: 12,
    DatasetPartition.TEST: 12,
}


def stratified_lineage_split(
    labels: tuple[EpisodeLabel, ...],
    *,
    split_seed: int,
) -> tuple[SplitAssignment, ...]:
    """Assign 60 episodes per class exactly 60/20/20 by lineage."""

    by_target: dict[int, list[EpisodeLabel]] = defaultdict(list)
    for label in labels:
        by_target[label.target].append(label)
    if set(by_target) != {-1, 1}:
        raise ValueError("development labels must contain targets -1 and +1")
    if any(len(items) != 60 for items in by_target.values()):
        raise ValueError("development split requires exactly 60 episodes per class")

    assignments: list[SplitAssignment] = []
    for target in (-1, 1):
        ordered = sorted(by_target[target], key=lambda item: item.lineage_id)
        rng = np.random.default_rng(np.random.SeedSequence([split_seed, 0 if target == -1 else 1]))
        shuffled = [ordered[index] for index in rng.permutation(len(ordered))]
        cursor = 0
        for partition in (
            DatasetPartition.TRAIN,
            DatasetPartition.VALIDATION,
            DatasetPartition.TEST,
        ):
            count = EXPECTED_PER_CLASS[partition]
            for label in shuffled[cursor : cursor + count]:
                assignments.append(
                    SplitAssignment(
                        episode_id=label.episode_id,
                        lineage_id=label.lineage_id,
                        partition=partition,
                    )
                )
            cursor += count

    result = tuple(sorted(assignments, key=lambda item: item.episode_id))
    validate_split(result, labels)
    return result


def validate_split(
    assignments: tuple[SplitAssignment, ...],
    labels: tuple[EpisodeLabel, ...],
) -> None:
    label_ids = [item.episode_id for item in labels]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("episode labels must have unique identifiers")
    if len(assignments) != len(labels):
        raise ValueError("every labeled episode must have exactly one split assignment")
    if len({item.episode_id for item in assignments}) != len(assignments):
        raise ValueError("an episode cannot appear in more than one partition")

    labels_by_id = {item.episode_id: item for item in labels}
    if {item.episode_id for item in assignments} != set(labels_by_id):
        raise ValueError("split assignments and labels must reference the same episodes")
    lineage_partitions: dict[str, set[DatasetPartition]] = defaultdict(set)
    for assignment in assignments:
        if labels_by_id[assignment.episode_id].lineage_id != assignment.lineage_id:
            raise ValueError("split assignment lineage does not match its label")
        lineage_partitions[assignment.lineage_id].add(assignment.partition)
    leaking = [lineage for lineage, parts in lineage_partitions.items() if len(parts) > 1]
    if leaking:
        raise ValueError(f"lineages cross partitions: {', '.join(sorted(leaking))}")

    targets = {label.episode_id: label.target for label in labels}
    counts = Counter(
        (assignment.partition, targets[assignment.episode_id]) for assignment in assignments
    )
    for partition, expected in EXPECTED_PER_CLASS.items():
        for target in (-1, 1):
            if counts[(partition, target)] != expected:
                raise ValueError(
                    f"partition {partition.value} target {target} must contain {expected} episodes"
                )


def validate_no_cross_partition_duplicates(
    assignments: tuple[SplitAssignment, ...],
    observation_digests: dict[str, str],
) -> None:
    """Reject byte-equivalent observations assigned to different partitions."""

    partitions = {item.episode_id: item.partition for item in assignments}
    digest_owners: dict[str, tuple[str, DatasetPartition]] = {}
    for episode_id, digest in sorted(observation_digests.items()):
        partition = partitions[episode_id]
        owner = digest_owners.get(digest)
        if owner is not None and owner[1] is not partition:
            raise ValueError(
                "duplicate observation payload crosses partitions: "
                f"{owner[0]} ({owner[1].value}) and {episode_id} ({partition.value})"
            )
        digest_owners[digest] = (episode_id, partition)


def validate_no_cross_partition_raw_overlaps(
    assignments: tuple[SplitAssignment, ...],
    intervals: tuple[ObservationInterval, ...],
) -> None:
    """Reject overlapping source intervals from one lineage across partitions.

    Relative time equality between independent episode lineages is intentionally
    not an overlap.
    """

    partitions = {item.episode_id: item.partition for item in assignments}
    if len({item.episode_id for item in intervals}) != len(intervals):
        raise ValueError("every episode must have one unambiguous raw source interval")
    if {item.episode_id for item in intervals} != set(partitions):
        raise ValueError("raw source intervals must reference every assigned episode")
    ordered = sorted(
        intervals,
        key=lambda item: (item.source_id or item.lineage_id, item.sensor_id, item.start_index),
    )
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            left_source = left.source_id or left.lineage_id
            right_source = right.source_id or right.lineage_id
            if (left_source, left.sensor_id) != (right_source, right.sensor_id):
                break
            if right.start_index >= left.end_index:
                break
            if partitions[left.episode_id] is not partitions[right.episode_id]:
                raise ValueError(
                    f"overlapping raw intervals cross partitions for source {left_source}"
                )


def extend_assignments_by_lineage(
    assignments: tuple[SplitAssignment, ...],
    derived_episode_lineages: dict[str, str],
) -> tuple[SplitAssignment, ...]:
    """Place replay/paired descendants in the partition of their lineage."""

    lineage_partition: dict[str, DatasetPartition] = {}
    existing_ids = {item.episode_id for item in assignments}
    for item in assignments:
        previous = lineage_partition.setdefault(item.lineage_id, item.partition)
        if previous is not item.partition:
            raise ValueError(f"parent lineage crosses partitions: {item.lineage_id}")
    extended = list(assignments)
    for episode_id, lineage_id in sorted(derived_episode_lineages.items()):
        if episode_id in existing_ids:
            raise ValueError(f"derived episode already has an assignment: {episode_id}")
        try:
            partition = lineage_partition[lineage_id]
        except KeyError as exc:
            raise ValueError(f"unknown parent lineage: {lineage_id}") from exc
        extended.append(
            SplitAssignment(
                episode_id=episode_id,
                lineage_id=lineage_id,
                partition=partition,
            )
        )
        existing_ids.add(episode_id)
    return tuple(extended)
