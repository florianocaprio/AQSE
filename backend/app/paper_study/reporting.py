from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import PermutationControlResult, ReplicatedStudyResult


def build_aggregate_report(
    study: ReplicatedStudyResult,
    *,
    permutation: PermutationControlResult | None = None,
    positive_control: ReplicatedStudyResult | None = None,
) -> dict[str, Any]:
    """Build an unrounded JSON-ready report without asserting significance.

    Every aggregate performance or kernel statistic is represented by a point
    estimate and its 95% bootstrap interval.  Wilcoxon p-values are inferential
    test outputs rather than effect-size intervals; they must not be read as a
    probability that either hypothesis is true.
    """

    def aggregate_payload(value: ReplicatedStudyResult) -> dict[str, Any]:
        return {
            "base_seed": value.base_seed,
            "replica_count": value.replica_count,
            "qng_steps": value.qng_steps,
            "aggregate": value.aggregate.model_dump(mode="json"),
            "scientific_scope": value.scientific_scope,
        }

    return {
        "schema_version": "aqse.paper-aggregate-report.v1",
        "scientific_scope": (
            "simulator-only replicated evidence; null results retained; no field or "
            "quantum-advantage claim"
        ),
        "primary_study": aggregate_payload(study),
        "negative_permutation_control": (
            None
            if permutation is None
            else {
                "replica_count": permutation.replica_count,
                "chance_value": permutation.chance_value,
                "balanced_accuracy": {
                    name: interval.model_dump(mode="json")
                    for name, interval in permutation.balanced_accuracy.items()
                },
                "compatible_with_chance": permutation.compatible_with_chance,
            }
        ),
        "positive_mechanism_control": (
            None if positive_control is None else aggregate_payload(positive_control)
        ),
        "reporting_policy": {
            "rounding": "none",
            "interval": "95% percentile bootstrap over independent replicas",
            "paired_test": "two-sided Wilcoxon signed-rank with Pratt zero handling",
        },
    }


def write_aggregate_report(
    path: Path,
    study: ReplicatedStudyResult,
    *,
    permutation: PermutationControlResult | None = None,
    positive_control: ReplicatedStudyResult | None = None,
) -> Path:
    """Write a report atomically; publication alone proves no scientific claim."""

    report = build_aggregate_report(
        study,
        permutation=permutation,
        positive_control=positive_control,
    )
    payload = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return path
