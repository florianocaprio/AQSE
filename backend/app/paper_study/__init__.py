"""Replicated AQSE experiments for paper-oriented statistical evaluation.

This package orchestrates the protected TQK8 implementation without changing
its circuit or QNG definitions.  Its outputs are simulation evidence only and
do not establish quantum advantage or field validity.
"""

from .datasets import ReplicatedDataset, positive_control_dataset
from .diagnostics import kernel_diagnostics
from .reporting import build_aggregate_report, write_aggregate_report
from .statistics import aggregate_replicas
from .study import permutation_control, run_replicated_study

__all__ = (
    "ReplicatedDataset",
    "aggregate_replicas",
    "build_aggregate_report",
    "kernel_diagnostics",
    "permutation_control",
    "positive_control_dataset",
    "run_replicated_study",
    "write_aggregate_report",
)
