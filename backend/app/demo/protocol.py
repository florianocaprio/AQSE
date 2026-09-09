from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field

from app.training.canonical import canonical_json_bytes
from app.training.models import FrozenModel

ScenarioName = Literal[
    "NORMAL",
    "ENVIRONMENT_COMPATIBLE",
    "DEVICE_COMPATIBLE",
    "MIXED_OR_AMBIGUOUS",
]

STUDY_ID = "aqse-network-demo-v1"
STUDY_SCHEMA_VERSION = "aqse.network-demo.protocol.v1"
SCENARIOS: tuple[ScenarioName, ...] = (
    "NORMAL",
    "ENVIRONMENT_COMPATIBLE",
    "DEVICE_COMPATIBLE",
    "MIXED_OR_AMBIGUOUS",
)
NODE_COUNTS = tuple(range(1, 9))
REPLICATES_PER_CELL = 5
SPLIT_COUNTS_PER_CELL = {"train": 3, "validation": 1, "test": 1}


class StudySeeds(FrozenModel):
    generation: Literal[2001001] = 2_001_001
    split: Literal[2001002] = 2_001_002
    bank_and_landmarks: Literal[2001003] = 2_001_003
    theta: Literal[2001004] = 2_001_004
    neural: Literal[2001005] = 2_001_005
    paired_bootstrap: Literal[2001006] = 2_001_006


class SimulationDomain(FrozenModel):
    sampling_rate_hz: Literal[100.0] = 100.0
    episode_duration_s: Literal[28.0] = 28.0
    reference_start_s: Literal[0.0] = 0.0
    reference_end_s: Literal[8.0] = 8.0
    supervised_window_start_s: Literal[18.0] = 18.0
    supervised_window_end_s: Literal[22.0] = 22.0
    analysis_window_s: Literal[4.0] = 4.0
    analysis_hop_s: Literal[1.0] = 1.0
    event_start_s: Literal[14.0] = 14.0
    event_duration_s: Literal[10.0] = 10.0
    base_white_noise_nt_range: tuple[float, float] = (0.15, 0.60)
    base_ou_noise_nt_range: tuple[float, float] = (0.05, 0.25)
    base_temperature_offset_k_range: tuple[float, float] = (-1.5, 1.5)
    perturbation_north_nt_range: tuple[float, float] = (6.0, 18.0)
    focal_extra_nt_range: tuple[float, float] = (2.0, 6.0)
    mobile_source_initial_x_m_range: tuple[float, float] = (-4.2, -3.8)
    mobile_source_y_m_range: tuple[float, float] = (-0.35, 0.35)
    mobile_source_height_m_range: tuple[float, float] = (1.0, 1.4)
    mobile_source_velocity_x_m_per_s_range: tuple[float, float] = (0.19, 0.21)
    focal_noise_multiplier_range: tuple[float, float] = (4.0, 8.0)
    focal_temperature_ramp_k_per_s_range: tuple[float, float] = (0.04, 0.08)
    focal_thermal_bias_nt_per_k_range: tuple[float, float] = (2.0, 5.0)


class QuantumStudyBudget(FrozenModel):
    theta_candidates: tuple[str, str] = ("theta0", "protected_qng")
    maximum_qng_updates: Literal[10] = 10
    maximum_qng_bank_rows: Literal[32] = 32
    minimum_examples_per_binary_class: Literal[4] = 4
    learning_rate: Literal[0.2] = 0.2
    damping: Literal[0.001] = 0.001
    maximum_step_norm: Literal[0.4] = 0.4
    backend: Literal["numpy_statevector"] = "numpy_statevector"


class AfsePolicy(FrozenModel):
    method_id: Literal["aqse.afse.nystrom-ridge32.v1"] = (
        "aqse.afse.nystrom-ridge32.v1"
    )
    requested_landmarks: Literal[32] = 32
    ridge_lambda: Literal[1e-6] = 1.0e-6
    negative_eigenvalue_relative_tolerance: Literal[1e-10] = 1.0e-10
    ood_policy: Literal["train-residual-p99-heuristic"] = (
        "train-residual-p99-heuristic"
    )


class NeuralPolicy(FrozenModel):
    architecture: tuple[int, int] = (32, 16)
    activation: Literal["tanh"] = "tanh"
    solver: Literal["lbfgs"] = "lbfgs"
    alpha: Literal[0.001] = 0.001
    maximum_iterations: Literal[500] = 500
    maximum_function_evaluations: Literal[15000] = 15_000
    early_stopping: Literal[False] = False
    top_score_threshold: Literal[0.70] = 0.70
    top_two_margin_threshold: Literal[0.15] = 0.15


class NetworkDemoProtocol(FrozenModel):
    schema_version: Literal["aqse.network-demo.protocol.v1"] = STUDY_SCHEMA_VERSION
    study_id: Literal["aqse-network-demo-v1"] = STUDY_ID
    scientific_label: Literal["research / not validated for field deployment"] = (
        "research / not validated for field deployment"
    )
    generation_domain_separator: Literal[
        "aqse-network-demo-v1/canonical-generation/v2"
    ] = "aqse-network-demo-v1/canonical-generation/v2"
    scenarios: tuple[ScenarioName, ...] = SCENARIOS
    node_counts: tuple[int, ...] = NODE_COUNTS
    replicates_per_cell: Literal[5] = REPLICATES_PER_CELL
    split_counts_per_cell: dict[str, int] = Field(
        default_factory=lambda: dict(SPLIT_COUNTS_PER_CELL)
    )
    focal_node_policy: Literal["predeclared-from-node-count-and-replicate"] = (
        "predeclared-from-node-count-and-replicate"
    )
    normal_event_policy: Literal["zero-amplitude-sham-on-common-schedule"] = (
        "zero-amplitude-sham-on-common-schedule"
    )
    environment_policy: Literal["moving-spatial-dipole"] = "moving-spatial-dipole"
    device_modes_by_replicate: tuple[str, ...] = (
        "bias",
        "drift",
        "noise",
        "thermal",
        "bias",
    )
    ambiguous_modes_by_replicate: tuple[str, ...] = (
        "common_field",
        "shared_instrument",
        "common_field",
        "shared_instrument",
        "common_field",
    )
    mixed_focal_component_replicates: tuple[int, ...] = (2, 3, 4)
    local_task_classes: tuple[str, str] = ("NORMAL", "CHANGE_DETECTED")
    network_task_classes: tuple[ScenarioName, ...] = SCENARIOS
    network_minimum_nodes: Literal[3] = 3
    two_node_policy: Literal["local-result-with-attribution-ambiguity"] = (
        "local-result-with-attribution-ambiguity"
    )
    seeds: StudySeeds = Field(default_factory=StudySeeds)
    simulation: SimulationDomain = Field(default_factory=SimulationDomain)
    quantum: QuantumStudyBudget = Field(default_factory=QuantumStudyBudget)
    afse: AfsePolicy = Field(default_factory=AfsePolicy)
    neural: NeuralPolicy = Field(default_factory=NeuralPolicy)
    selection_rule: tuple[str, ...] = (
        "maximum_validation_balanced_accuracy",
        "maximum_validation_macro_f1",
        "prefer_theta0_on_tie",
    )
    final_evaluation_rule: Literal[
        "single-new-test-evaluation-after-selection-freeze"
    ] = "single-new-test-evaluation-after-selection-freeze"
    historical_test_policy: Literal["never-read-or-modify"] = "never-read-or-modify"

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()


FROZEN_NETWORK_DEMO_PROTOCOL = NetworkDemoProtocol()
