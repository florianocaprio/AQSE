from __future__ import annotations

from collections import Counter
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from app.features.state8_models import (
    NETWORK_STATE8_PROFILE_ID,
    STATE8_WINDOW_SAMPLES,
    State8FeatureRecord,
    State8Values,
)
from app.network.models import EventKind, NetworkSessionConfiguration, TemperatureDriverKind
from app.training.models import FrozenModel

from .protocol import (
    FROZEN_FOUR_SENSOR_STUDY_PROTOCOL,
    GEOMETRY_IDS,
    SCENARIOS,
    SENSOR_IDS,
    GeometryId,
    ScenarioName,
)

StudyPartition = Literal["pilot", "test"]
StudyKind = Literal["pilot", "final_test"]
REPLAY_WINDOW_STARTS_S = tuple(float(value) for value in range(8, 25))


class FourSensorStudyEpisodePlan(FrozenModel):
    """Private generative plan; it never crosses the predictor boundary."""

    schema_version: Literal["aqse.four-sensor-study.episode-plan.v1"] = (
        "aqse.four-sensor-study.episode-plan.v1"
    )
    episode_id: str = Field(pattern=r"^four-sensor-episode-[a-f0-9]{16}$")
    generative_lineage_id: str = Field(
        pattern=r"^four-sensor-lineage-[a-f0-9]{16}$"
    )
    generation_domain: str = Field(min_length=1, max_length=128)
    partition: StudyPartition
    scenario: ScenarioName
    geometry_id: GeometryId
    node_count: Literal[4] = 4
    replicate_index: int = Field(ge=0, lt=25)
    episode_seed: int = Field(ge=0, le=2**32 - 1)
    focal_sensor_id: str = Field(pattern=r"^S[1-4]$")
    perturbation_north_nt: float = Field(ge=6.0, le=18.0)
    focal_extra_nt: float = Field(ge=2.0, le=6.0)
    protocol_freeze_digest: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    network_configuration: dict[str, Any]

    @model_validator(mode="after")
    def validate_generation_configuration(self) -> FourSensorStudyEpisodePlan:
        protocol = FROZEN_FOUR_SENSOR_STUDY_PROTOCOL
        if self.partition == "pilot":
            if self.replicate_index >= protocol.pilot.episodes_per_scenario:
                raise ValueError("pilot replicate is outside the predeclared design")
            if self.generation_domain != protocol.pilot_generation_domain_separator:
                raise ValueError("pilot plan uses an incompatible generation domain")
            if self.protocol_freeze_digest is not None:
                raise ValueError("noncanonical pilot plans precede protocol publication")
        else:
            if self.generation_domain != protocol.canonical_generation_domain_separator:
                raise ValueError("final TEST plan uses an incompatible generation domain")
            if self.protocol_freeze_digest is None:
                raise ValueError("final TEST plans require a published protocol freeze")

        configuration = NetworkSessionConfiguration.model_validate(
            self.network_configuration
        )
        if configuration.random_seed != self.episode_seed:
            raise ValueError("episode seed differs from the simulator configuration")
        if len(configuration.nodes) != 4:
            raise ValueError("four-sensor study plans require exactly four nodes")
        nodes_by_id = {node.sensor_id: node for node in configuration.nodes}
        if set(nodes_by_id) != set(SENSOR_IDS):
            raise ValueError("four-sensor study plans require exactly S1 through S4")
        if self.focal_sensor_id not in nodes_by_id:
            raise ValueError("focal sensor is absent from the simulator configuration")
        expected_positions = dict(
            zip(
                SENSOR_IDS,
                protocol.geometry(self.geometry_id).positions_m,
                strict=True,
            )
        )
        if {
            sensor_id: node.position_m for sensor_id, node in nodes_by_id.items()
        } != expected_positions:
            raise ValueError("simulator node positions differ from the declared geometry")
        simulation = protocol.simulation
        if configuration.sampling_rate_Hz != simulation.sampling_rate_hz:
            raise ValueError("four-sensor study must use the frozen 100 Hz rate")
        if configuration.buffer_duration_s != simulation.episode_duration_s:
            raise ValueError("four-sensor study must use the frozen 28 s duration")
        active_dipoles = tuple(
            item for item in configuration.environment.dipoles if item.enabled
        )
        events = configuration.events
        if self.scenario in {"ENVIRONMENT_COMPATIBLE", "MIXED_OR_AMBIGUOUS"}:
            if len(active_dipoles) != 1:
                raise ValueError("spatial-change scenarios require one active dipole")
            if (
                active_dipoles[0].active_start_time_s != simulation.event_start_s
                or active_dipoles[0].active_duration_s != simulation.event_duration_s
            ):
                raise ValueError("dipole timing differs from the frozen event window")
        elif active_dipoles:
            raise ValueError("normal/device scenarios cannot contain an active dipole")
        if self.scenario == "NORMAL":
            if (
                len(events) != 1
                or events[0].kind is not EventKind.WORLD_FIELD_OFFSET
                or events[0].field_offset_T != (0.0, 0.0, 0.0)
            ):
                raise ValueError("NORMAL requires exactly one zero-amplitude sham event")
        elif self.scenario == "ENVIRONMENT_COMPATIBLE" and events:
            raise ValueError("environment-only episodes cannot contain device events")
        elif self.scenario == "DEVICE_COMPATIBLE":
            mode = protocol.device_modes[
                self.replicate_index % len(protocol.device_modes)
            ]
            expected_kind = {
                "bias": EventKind.NODE_BIAS,
                "drift": EventKind.NODE_DRIFT,
                "noise": EventKind.NODE_NOISE_BURST,
            }.get(mode)
            if mode == "thermal":
                focal = nodes_by_id[self.focal_sensor_id]
                driver = focal.errors.temperature_driver
                if (
                    events
                    or driver is None
                    or driver.kind is not TemperatureDriverKind.RAMP
                    or driver.start_time_s != simulation.event_start_s
                    or driver.ramp_duration_s != simulation.event_duration_s
                ):
                    raise ValueError("thermal device episode differs from its freeze")
            elif (
                len(events) != 1
                or events[0].kind is not expected_kind
                or events[0].target_sensor_ids != (self.focal_sensor_id,)
            ):
                raise ValueError("device episode differs from its frozen mode")
        elif self.scenario == "MIXED_OR_AMBIGUOUS" and (
            len(events) != 1
            or events[0].kind is not EventKind.NODE_BIAS
            or events[0].target_sensor_ids != (self.focal_sensor_id,)
        ):
            raise ValueError("mixed episode requires one focal device-bias component")
        return self


class FourSensorStudyObservation(FrozenModel):
    """Predictor channel: observed network State8 only, never generator truth."""

    schema_version: Literal["aqse.four-sensor-study.observation.v1"] = (
        "aqse.four-sensor-study.observation.v1"
    )
    episode_id: str = Field(pattern=r"^four-sensor-episode-[a-f0-9]{16}$")
    generative_lineage_id: str = Field(
        pattern=r"^four-sensor-lineage-[a-f0-9]{16}$"
    )
    partition: StudyPartition
    node_count: Literal[4] = 4
    network_feature: State8FeatureRecord
    network_replay_features: tuple[State8FeatureRecord, ...]

    @model_validator(mode="after")
    def validate_observed_features(self) -> FourSensorStudyObservation:
        feature = self.network_feature
        if feature.profile_id != NETWORK_STATE8_PROFILE_ID:
            raise ValueError("primary feature has an incompatible State8 profile")
        if feature.session_id != self.episode_id:
            raise ValueError("primary feature belongs to a different episode")
        if len(feature.peer_sensor_ids) != 3:
            raise ValueError("four-sensor network features require exactly three peers")
        if (
            not feature.quality.valid_for_quantum
            or feature.quality.received_sample_count != STATE8_WINDOW_SAMPLES
            or feature.quality.usable_sample_count != STATE8_WINDOW_SAMPLES
            or len(feature.source_frame_ids) != STATE8_WINDOW_SAMPLES
            or any(value is None for value in feature.values)
        ):
            raise ValueError("primary feature is not a complete eligible State8 window")

        starts = tuple(item.start_time_s for item in self.network_replay_features)
        if starts != REPLAY_WINDOW_STARTS_S:
            raise ValueError("replay trace must contain the frozen 17 causal windows")
        for item in self.network_replay_features:
            if (
                item.profile_id != NETWORK_STATE8_PROFILE_ID
                or item.profile_fingerprint != feature.profile_fingerprint
                or item.reference_id != feature.reference_id
                or item.session_id != self.episode_id
                or item.sensor_id != feature.sensor_id
                or item.peer_sensor_ids != feature.peer_sensor_ids
                or item.end_exclusive_time_s != item.start_time_s + 4.0
                or len(item.source_frame_ids) != STATE8_WINDOW_SAMPLES
                or not item.quality.valid_for_quantum
                or item.quality.received_sample_count != STATE8_WINDOW_SAMPLES
                or item.quality.usable_sample_count != STATE8_WINDOW_SAMPLES
                or any(value is None for value in item.values)
            ):
                raise ValueError("replay feature is incomplete or incompatible")
        if feature != self.network_replay_features[10]:
            raise ValueError("primary feature differs from the 18 s replay window")
        return self

    @property
    def predictor_values(self) -> tuple[float, float, float, float, float, float, float, float]:
        """The only covariates admitted to the already-frozen predictor."""

        return cast(
            tuple[float, float, float, float, float, float, float, float],
            cast(State8Values, self.network_feature.values),
        )


class FourSensorStudyLabel(FrozenModel):
    """Independent supervisory/audit channel, never accepted by inference."""

    schema_version: Literal["aqse.four-sensor-study.label.v1"] = (
        "aqse.four-sensor-study.label.v1"
    )
    episode_id: str = Field(pattern=r"^four-sensor-episode-[a-f0-9]{16}$")
    generative_lineage_id: str = Field(
        pattern=r"^four-sensor-lineage-[a-f0-9]{16}$"
    )
    partition: StudyPartition
    scenario: ScenarioName
    geometry_id: GeometryId
    focal_sensor_id: str = Field(pattern=r"^S[1-4]$")
    event_start_s: Literal[14.0] = 14.0


class FourSensorStudyBuild(FrozenModel):
    schema_version: Literal["aqse.four-sensor-study.build.v1"] = (
        "aqse.four-sensor-study.build.v1"
    )
    kind: StudyKind
    protocol_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    protocol_freeze_digest: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    episode_plans: tuple[FourSensorStudyEpisodePlan, ...]
    observations: tuple[FourSensorStudyObservation, ...]
    labels: tuple[FourSensorStudyLabel, ...]

    @model_validator(mode="after")
    def validate_alignment_and_design(self) -> FourSensorStudyBuild:
        protocol = FROZEN_FOUR_SENSOR_STUDY_PROTOCOL
        if self.protocol_digest != protocol.digest:
            raise ValueError("study build protocol digest is incompatible")
        expected_count = 20 if self.kind == "pilot" else 100
        expected_partition: StudyPartition = (
            "pilot" if self.kind == "pilot" else "test"
        )
        if not (
            len(self.episode_plans)
            == len(self.observations)
            == len(self.labels)
            == expected_count
        ):
            raise ValueError("study build channels have an incompatible row count")
        if self.kind == "pilot" and self.protocol_freeze_digest is not None:
            raise ValueError("pilot build must precede protocol-freeze publication")
        if self.kind == "final_test" and self.protocol_freeze_digest is None:
            raise ValueError("final TEST build requires a protocol freeze digest")

        plans_by_id = {item.episode_id: item for item in self.episode_plans}
        observations_by_id = {item.episode_id: item for item in self.observations}
        labels_by_id = {item.episode_id: item for item in self.labels}
        if not all(
            len(channel) == expected_count
            for channel in (plans_by_id, observations_by_id, labels_by_id)
        ):
            raise ValueError("episode identities must be unique in every channel")
        if set(plans_by_id) != set(observations_by_id) or set(plans_by_id) != set(
            labels_by_id
        ):
            raise ValueError("plan, observation and label channels do not align")
        if len({item.generative_lineage_id for item in self.episode_plans}) != expected_count:
            raise ValueError("every episode requires an independent generative lineage")

        for episode_id, plan in plans_by_id.items():
            observation = observations_by_id[episode_id]
            label = labels_by_id[episode_id]
            if any(
                item.generative_lineage_id != plan.generative_lineage_id
                or item.partition != expected_partition
                for item in (observation, label)
            ):
                raise ValueError("study channels differ in lineage or partition")
            if (
                plan.partition != expected_partition
                or label.scenario != plan.scenario
                or label.geometry_id != plan.geometry_id
                or label.focal_sensor_id != plan.focal_sensor_id
                or observation.network_feature.sensor_id != plan.focal_sensor_id
            ):
                raise ValueError("study channels differ from the predeclared plan")
            if plan.protocol_freeze_digest != self.protocol_freeze_digest:
                raise ValueError("episode plan uses a different protocol freeze")

        design_cells = {
            (item.scenario, item.replicate_index) for item in self.episode_plans
        }
        if len(design_cells) != expected_count:
            raise ValueError("scenario/replicate design cells must be unique")
        for plan in self.episode_plans:
            scenario_index = SCENARIOS.index(plan.scenario)
            expected_geometry = GEOMETRY_IDS[
                (scenario_index + plan.replicate_index) % len(GEOMETRY_IDS)
            ]
            focal_index = scenario_index + 2 * plan.replicate_index
            if self.kind == "final_test":
                focal_index += plan.replicate_index // 4
            expected_focal = SENSOR_IDS[focal_index % len(SENSOR_IDS)]
            if (
                plan.geometry_id != expected_geometry
                or plan.focal_sensor_id != expected_focal
            ):
                raise ValueError("plan nuisance allocation differs from its schedule")

        scenario_counts = Counter(item.scenario for item in self.labels)
        geometry_counts = Counter(item.geometry_id for item in self.labels)
        focal_counts = Counter(item.focal_sensor_id for item in self.labels)
        expected_per_scenario = 5 if self.kind == "pilot" else 25
        expected_per_nuisance = 5 if self.kind == "pilot" else 25
        if scenario_counts != Counter({item: expected_per_scenario for item in SCENARIOS}):
            raise ValueError("study scenario allocation differs from its design")
        if geometry_counts != Counter({item: expected_per_nuisance for item in GEOMETRY_IDS}):
            raise ValueError("study geometry allocation differs from its design")
        if focal_counts != Counter({item: expected_per_nuisance for item in SENSOR_IDS}):
            raise ValueError("study focal-sensor allocation differs from its design")
        return self
