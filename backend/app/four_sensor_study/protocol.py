from __future__ import annotations

import hashlib
from collections import Counter
from typing import Literal

from pydantic import Field, model_validator

from app.training.canonical import canonical_json_bytes
from app.training.models import FrozenModel

ScenarioName = Literal[
    "NORMAL",
    "ENVIRONMENT_COMPATIBLE",
    "DEVICE_COMPATIBLE",
    "MIXED_OR_AMBIGUOUS",
]
GeometryId = Literal[
    "square",
    "rectangle",
    "rotated_square",
    "irregular",
]
Vector3 = tuple[float, float, float]

STUDY_ID = "aqse-four-sensor-study-v1"
STUDY_SCHEMA_VERSION = "aqse.four-sensor-study.protocol.v1"
SCENARIOS: tuple[ScenarioName, ...] = (
    "NORMAL",
    "ENVIRONMENT_COMPATIBLE",
    "DEVICE_COMPATIBLE",
    "MIXED_OR_AMBIGUOUS",
)
GEOMETRY_IDS: tuple[GeometryId, ...] = (
    "square",
    "rectangle",
    "rotated_square",
    "irregular",
)
SENSOR_IDS = ("S1", "S2", "S3", "S4")

_GEOMETRY_POSITIONS: dict[GeometryId, tuple[Vector3, Vector3, Vector3, Vector3]] = {
    "square": (
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
    ),
    "rectangle": (
        (-0.75, -0.4, 0.0),
        (0.75, -0.4, 0.0),
        (0.75, 0.4, 0.0),
        (-0.75, 0.4, 0.0),
    ),
    "rotated_square": (
        (0.0, -0.7071067811865476, 0.0),
        (0.7071067811865476, 0.0, 0.0),
        (0.0, 0.7071067811865476, 0.0),
        (-0.7071067811865476, 0.0, 0.0),
    ),
    "irregular": (
        (-0.65, -0.45, 0.0),
        (0.58, -0.32, 0.0),
        (0.66, 0.61, 0.0),
        (-0.44, 0.53, 0.0),
    ),
}


class GeometryDefinition(FrozenModel):
    geometry_id: GeometryId
    positions_m: tuple[Vector3, Vector3, Vector3, Vector3]

    @model_validator(mode="after")
    def validate_geometry(self) -> GeometryDefinition:
        if len(set(self.positions_m)) != 4:
            raise ValueError("a four-sensor geometry requires four distinct positions")
        if any(position[2] != 0.0 for position in self.positions_m):
            raise ValueError("the frozen four-sensor geometries must remain planar")
        return self


class StudySeeds(FrozenModel):
    pilot_generation: Literal[3001001] = 3_001_001
    canonical_generation: Literal[3001002] = 3_001_002


class PilotDesign(FrozenModel):
    episodes_per_scenario: Literal[5] = 5
    total_episodes: Literal[20] = 20
    geometry_schedule: Literal["(scenario_index+replicate)%4"] = (
        "(scenario_index+replicate)%4"
    )
    focal_sensor_schedule: Literal["(scenario_index+2*replicate)%4"] = (
        "(scenario_index+2*replicate)%4"
    )
    disposition: Literal["noncanonical-structural-feasibility-only"] = (
        "noncanonical-structural-feasibility-only"
    )
    minimum_primary_coverage: Literal[1.0] = 1.0
    minimum_replay_coverage: Literal[1.0] = 1.0


class CanonicalTestDesign(FrozenModel):
    episodes_per_scenario: Literal[25] = 25
    total_episodes: Literal[100] = 100
    partition: Literal["test"] = "test"
    geometry_schedule: Literal["(scenario_index+replicate)%4"] = (
        "(scenario_index+replicate)%4"
    )
    focal_sensor_schedule: Literal[
        "(scenario_index+2*replicate+floor(replicate/4))%4"
    ] = "(scenario_index+2*replicate+floor(replicate/4))%4"
    model_policy: Literal["evaluate-already-frozen-existing-model-only"] = (
        "evaluate-already-frozen-existing-model-only"
    )
    fit_policy: Literal["no-fit-no-selection-no-refit"] = (
        "no-fit-no-selection-no-refit"
    )
    evaluation_policy: Literal["single-final-held-out-evaluation"] = (
        "single-final-held-out-evaluation"
    )
    minimum_primary_coverage: Literal[1.0] = 1.0
    minimum_replay_coverage: Literal[1.0] = 1.0

    @model_validator(mode="after")
    def validate_balance_and_no_identity_leakage(self) -> CanonicalTestDesign:
        geometry_counts: Counter[int] = Counter()
        focal_counts: Counter[int] = Counter()
        geometry_focal_counts: Counter[tuple[int, int]] = Counter()
        for scenario_index in range(len(SCENARIOS)):
            per_scenario_geometry: Counter[int] = Counter()
            per_scenario_focal: Counter[int] = Counter()
            for replicate in range(self.episodes_per_scenario):
                geometry_index = (scenario_index + replicate) % len(GEOMETRY_IDS)
                focal_index = (
                    scenario_index + 2 * replicate + replicate // 4
                ) % len(SENSOR_IDS)
                geometry_counts[geometry_index] += 1
                focal_counts[focal_index] += 1
                geometry_focal_counts[(geometry_index, focal_index)] += 1
                per_scenario_geometry[geometry_index] += 1
                per_scenario_focal[focal_index] += 1
            if set(per_scenario_geometry.values()) != {6, 7}:
                raise ValueError("geometry identity is not balanced within each scenario")
            if set(per_scenario_focal.values()) != {6, 7}:
                raise ValueError("focal identity is not balanced within each scenario")
        if set(geometry_counts.values()) != {25}:
            raise ValueError("each geometry must appear exactly 25 times")
        if set(focal_counts.values()) != {25}:
            raise ValueError("each focal sensor must appear exactly 25 times")
        if len(geometry_focal_counts) != 16 or set(geometry_focal_counts.values()) != {
            6,
            7,
        }:
            raise ValueError("geometry/focal cells must have balanced 6/7 occupancy")
        return self


# Kept as a public alias because generation/storage APIs use the term
# "canonical" for the post-freeze corpus. It is intentionally TEST-only.
CanonicalDesign = CanonicalTestDesign


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

    @model_validator(mode="after")
    def validate_timeline_and_ranges(self) -> SimulationDomain:
        if self.reference_start_s != 0.0 or self.reference_end_s > self.event_start_s:
            raise ValueError("the observed reference must finish before the event starts")
        if self.event_start_s + self.event_duration_s > self.episode_duration_s:
            raise ValueError("the frozen event extends beyond the episode")
        if not (
            self.event_start_s
            <= self.supervised_window_start_s
            < self.supervised_window_end_s
            <= self.event_start_s + self.event_duration_s
        ):
            raise ValueError("the supervised window must remain inside the event interval")
        bounded_ranges = (
            self.base_white_noise_nt_range,
            self.base_ou_noise_nt_range,
            self.base_temperature_offset_k_range,
            self.perturbation_north_nt_range,
            self.focal_extra_nt_range,
            self.mobile_source_initial_x_m_range,
            self.mobile_source_y_m_range,
            self.mobile_source_height_m_range,
            self.mobile_source_velocity_x_m_per_s_range,
            self.focal_noise_multiplier_range,
            self.focal_temperature_ramp_k_per_s_range,
            self.focal_thermal_bias_nt_per_k_range,
        )
        if any(lower > upper for lower, upper in bounded_ranges):
            raise ValueError("simulation parameter ranges must be ordered")
        return self


def _default_geometries() -> tuple[GeometryDefinition, ...]:
    return tuple(
        GeometryDefinition(
            geometry_id=geometry_id,
            positions_m=_GEOMETRY_POSITIONS[geometry_id],
        )
        for geometry_id in GEOMETRY_IDS
    )


class FourSensorStudyProtocol(FrozenModel):
    schema_version: Literal["aqse.four-sensor-study.protocol.v1"] = (
        STUDY_SCHEMA_VERSION
    )
    study_id: Literal["aqse-four-sensor-study-v1"] = STUDY_ID
    scientific_label: Literal["research / not validated for field deployment"] = (
        "research / not validated for field deployment"
    )
    node_count: Literal[4] = 4
    sensor_ids: tuple[str, str, str, str] = SENSOR_IDS
    scenarios: tuple[ScenarioName, ...] = SCENARIOS
    geometries: tuple[GeometryDefinition, ...] = Field(default_factory=_default_geometries)
    pilot_generation_domain_separator: Literal[
        "aqse-four-sensor-study-v1/pilot-generation/v1"
    ] = "aqse-four-sensor-study-v1/pilot-generation/v1"
    canonical_generation_domain_separator: Literal[
        "aqse-four-sensor-study-v1/final-test-generation/v1"
    ] = "aqse-four-sensor-study-v1/final-test-generation/v1"
    focal_node_policy: Literal["balanced-predeclared-before-simulation"] = (
        "balanced-predeclared-before-simulation"
    )
    normal_event_policy: Literal["zero-amplitude-sham-on-common-schedule"] = (
        "zero-amplitude-sham-on-common-schedule"
    )
    environment_policy: Literal["scheduled-moving-spatial-dipole"] = (
        "scheduled-moving-spatial-dipole"
    )
    device_modes: tuple[str, str, str, str] = ("bias", "drift", "noise", "thermal")
    mixed_policy: Literal["moving-spatial-dipole-plus-focal-device-bias"] = (
        "moving-spatial-dipole-plus-focal-device-bias"
    )
    corpus_policy: Literal["100-final-test-only-no-development-split"] = (
        "100-final-test-only-no-development-split"
    )
    historical_artifact_policy: Literal["never-read-modify-or-reopen"] = (
        "never-read-modify-or-reopen"
    )
    seeds: StudySeeds = Field(default_factory=StudySeeds)
    pilot: PilotDesign = Field(default_factory=PilotDesign)
    canonical: CanonicalTestDesign = Field(default_factory=CanonicalTestDesign)
    simulation: SimulationDomain = Field(default_factory=SimulationDomain)

    @model_validator(mode="after")
    def validate_frozen_design(self) -> FourSensorStudyProtocol:
        if self.sensor_ids != SENSOR_IDS:
            raise ValueError("the frozen study requires sensor IDs S1 through S4")
        if self.scenarios != SCENARIOS:
            raise ValueError("the frozen study requires the four declared scenarios")
        if tuple(item.geometry_id for item in self.geometries) != GEOMETRY_IDS:
            raise ValueError("the frozen study requires the four geometries in canonical order")
        for geometry in self.geometries:
            if geometry.positions_m != _GEOMETRY_POSITIONS[geometry.geometry_id]:
                raise ValueError(f"geometry {geometry.geometry_id} differs from its freeze")
        if (
            self.pilot_generation_domain_separator
            == self.canonical_generation_domain_separator
        ):
            raise ValueError("pilot and final-test generation domains must remain disjoint")
        return self

    def geometry(self, geometry_id: GeometryId) -> GeometryDefinition:
        return next(item for item in self.geometries if item.geometry_id == geometry_id)

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()


FROZEN_FOUR_SENSOR_STUDY_PROTOCOL = FourSensorStudyProtocol()
