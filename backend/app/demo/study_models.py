from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from app.demo.protocol import FROZEN_NETWORK_DEMO_PROTOCOL, ScenarioName
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
    STATE8_WINDOW_SAMPLES,
    State8FeatureRecord,
)
from app.network.models import NetworkSessionConfiguration
from app.training.models import FrozenModel

StudyPartition = Literal["train", "validation", "test"]


class NetworkStudyEpisodePlan(FrozenModel):
    schema_version: Literal["aqse.network-demo.episode-plan.v1"] = (
        "aqse.network-demo.episode-plan.v1"
    )
    episode_id: str = Field(pattern=r"^network-episode-[a-f0-9]{16}$")
    generative_lineage_id: str = Field(pattern=r"^network-lineage-[a-f0-9]{16}$")
    generation_domain: str = Field(min_length=1, max_length=128)
    partition: StudyPartition
    scenario: ScenarioName
    node_count: int = Field(ge=1, le=8)
    replicate_index: int = Field(ge=0, lt=5)
    episode_seed: int = Field(ge=0, le=2**32 - 1)
    focal_sensor_id: str = Field(pattern=r"^S[1-8]$")
    perturbation_north_nt: float = Field(ge=6.0, le=18.0)
    focal_extra_nt: float = Field(ge=2.0, le=6.0)
    network_configuration: dict[str, Any]

    @model_validator(mode="after")
    def validate_generation_configuration(self) -> NetworkStudyEpisodePlan:
        configuration = NetworkSessionConfiguration.model_validate(
            self.network_configuration
        )
        sensor_ids = tuple(node.sensor_id for node in configuration.nodes)
        if configuration.random_seed != self.episode_seed:
            raise ValueError("episode seed differs from the simulator configuration")
        if len(sensor_ids) != self.node_count or self.focal_sensor_id not in sensor_ids:
            raise ValueError("episode node/focal design differs from its configuration")
        simulation = FROZEN_NETWORK_DEMO_PROTOCOL.simulation
        if configuration.sampling_rate_Hz != simulation.sampling_rate_hz:
            raise ValueError("network study simulator must use the frozen 100 Hz rate")
        if configuration.buffer_duration_s != simulation.episode_duration_s:
            raise ValueError("network study simulator must use the frozen 28 s duration")
        return self


class NetworkStudyObservation(FrozenModel):
    """Predictive channel: no scenario, label, seed or generator truth."""

    schema_version: Literal["aqse.network-demo.observation.v1"] = (
        "aqse.network-demo.observation.v1"
    )
    episode_id: str = Field(pattern=r"^network-episode-[a-f0-9]{16}$")
    generative_lineage_id: str = Field(pattern=r"^network-lineage-[a-f0-9]{16}$")
    partition: StudyPartition
    node_count: int = Field(ge=1, le=8)
    focal_sensor_id: str = Field(pattern=r"^S[1-8]$")
    local_feature: State8FeatureRecord
    network_feature: State8FeatureRecord | None

    @model_validator(mode="after")
    def validate_profiles(self) -> NetworkStudyObservation:
        if self.local_feature.profile_id != LOCAL_STATE8_PROFILE_ID:
            raise ValueError("local feature has an incompatible profile")
        if (
            self.local_feature.session_id != self.episode_id
            or self.local_feature.sensor_id != self.focal_sensor_id
        ):
            raise ValueError("local feature identity differs from its observation")
        if (
            not self.local_feature.quality.valid_for_quantum
            or self.local_feature.quality.received_sample_count
            != STATE8_WINDOW_SAMPLES
            or self.local_feature.quality.usable_sample_count != STATE8_WINDOW_SAMPLES
            or len(self.local_feature.source_frame_ids) != STATE8_WINDOW_SAMPLES
        ):
            raise ValueError("local feature is not a complete eligible study window")
        if (
            self.local_feature.start_time_s
            != FROZEN_NETWORK_DEMO_PROTOCOL.simulation.supervised_window_start_s
            or self.local_feature.end_exclusive_time_s
            != FROZEN_NETWORK_DEMO_PROTOCOL.simulation.supervised_window_end_s
        ):
            raise ValueError("local feature differs from the frozen supervised window")
        if self.node_count < 3 and self.network_feature is not None:
            raise ValueError("network feature cannot exist with fewer than three nodes")
        if self.node_count >= 3 and self.network_feature is None:
            raise ValueError("eligible network episodes require a network feature")
        if self.network_feature is not None:
            if self.network_feature.profile_id != NETWORK_STATE8_PROFILE_ID:
                raise ValueError("network feature has an incompatible profile")
            if (
                self.network_feature.session_id != self.episode_id
                or self.network_feature.sensor_id != self.focal_sensor_id
            ):
                raise ValueError("network feature identity differs from its observation")
            if len(self.network_feature.peer_sensor_ids) != self.node_count - 1:
                raise ValueError("network feature peer count differs from node_count")
            if (
                not self.network_feature.quality.valid_for_quantum
                or self.network_feature.quality.received_sample_count
                != STATE8_WINDOW_SAMPLES
                or self.network_feature.quality.usable_sample_count
                != STATE8_WINDOW_SAMPLES
                or len(self.network_feature.source_frame_ids) != STATE8_WINDOW_SAMPLES
            ):
                raise ValueError("network feature is not a complete eligible study window")
            if (
                self.network_feature.reference_id != self.local_feature.reference_id
                or self.network_feature.start_time_s
                != self.local_feature.start_time_s
                or self.network_feature.end_exclusive_time_s
                != self.local_feature.end_exclusive_time_s
            ):
                raise ValueError("local and network feature contexts are not aligned")
        return self


class NetworkStudyLabel(FrozenModel):
    """Independent supervisory channel; never accepted by inference."""

    schema_version: Literal["aqse.network-demo.label.v1"] = (
        "aqse.network-demo.label.v1"
    )
    episode_id: str = Field(pattern=r"^network-episode-[a-f0-9]{16}$")
    generative_lineage_id: str = Field(pattern=r"^network-lineage-[a-f0-9]{16}$")
    partition: StudyPartition
    local_label: Literal["NORMAL", "CHANGE_DETECTED"]
    network_label: ScenarioName


class NetworkStudyBuild(FrozenModel):
    protocol_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    episode_plans: tuple[NetworkStudyEpisodePlan, ...]
    observations: tuple[NetworkStudyObservation, ...]
    labels: tuple[NetworkStudyLabel, ...]

    @model_validator(mode="after")
    def validate_alignment(self) -> NetworkStudyBuild:
        if self.protocol_digest != FROZEN_NETWORK_DEMO_PROTOCOL.digest:
            raise ValueError("network study protocol digest is incompatible")
        if len(self.episode_plans) != 160:
            raise ValueError("the canonical network study requires 160 episode plans")
        if len(self.observations) != 160 or len(self.labels) != 160:
            raise ValueError("each network study channel must contain 160 rows")
        plan_ids = [item.episode_id for item in self.episode_plans]
        observation_ids = [item.episode_id for item in self.observations]
        label_ids = [item.episode_id for item in self.labels]
        if any(len(set(items)) != 160 for items in (plan_ids, observation_ids, label_ids)):
            raise ValueError("episode identities must be unique within every channel")
        if set(plan_ids) != set(observation_ids) or set(plan_ids) != set(label_ids):
            raise ValueError("study plan, observation and label channels must align")
        if len({item.generative_lineage_id for item in self.episode_plans}) != 160:
            raise ValueError("every study episode requires an independent lineage")
        observations_by_id = {item.episode_id: item for item in self.observations}
        labels_by_id = {item.episode_id: item for item in self.labels}
        counts = {"train": 0, "validation": 0, "test": 0}
        cells: dict[tuple[str, int], dict[str, int]] = {}
        design_cells: set[tuple[str, int, int]] = set()
        for item in self.episode_plans:
            if (
                item.generation_domain
                != FROZEN_NETWORK_DEMO_PROTOCOL.generation_domain_separator
            ):
                raise ValueError("canonical study plan uses a non-canonical domain")
            design_cell = (item.scenario, item.node_count, item.replicate_index)
            if design_cell in design_cells:
                raise ValueError("scenario/node/replicate design cells must be unique")
            design_cells.add(design_cell)
            counts[item.partition] += 1
            cell = cells.setdefault(
                (item.scenario, item.node_count),
                {"train": 0, "validation": 0, "test": 0},
            )
            cell[item.partition] += 1
            observation = observations_by_id[item.episode_id]
            label = labels_by_id[item.episode_id]
            for channel in (observation, label):
                if (
                    channel.generative_lineage_id != item.generative_lineage_id
                    or channel.partition != item.partition
                ):
                    raise ValueError(
                        "study plan, observation and label lineage/partition must align"
                    )
            if (
                observation.node_count != item.node_count
                or observation.focal_sensor_id != item.focal_sensor_id
            ):
                raise ValueError("observation context differs from its generation plan")
            expected_local_label = (
                "NORMAL" if item.scenario == "NORMAL" else "CHANGE_DETECTED"
            )
            if (
                label.local_label != expected_local_label
                or label.network_label != item.scenario
            ):
                raise ValueError("label channel differs from its generation plan")
        if counts != {"train": 96, "validation": 32, "test": 32}:
            raise ValueError("study partitions must contain 96/32/32 episodes")
        if len(cells) != 32 or len(design_cells) != 160:
            raise ValueError("study design must cover all scenario/node/replicate cells")
        if any(value != {"train": 3, "validation": 1, "test": 1} for value in cells.values()):
            raise ValueError("each scenario/node-count cell must use a 3/1/1 split")
        return self


class StudyFileRecord(FrozenModel):
    relative_path: str
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class NetworkStudyManifest(FrozenModel):
    schema_version: Literal["aqse.network-demo.manifest.v1"] = (
        "aqse.network-demo.manifest.v1"
    )
    artifact_id: str = Field(pattern=r"^aqse-network-study-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    protocol_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_profile_fingerprints: dict[str, str]
    partition_counts: dict[str, int]
    files: tuple[StudyFileRecord, ...]
    historical_test_ledger_sha256: Literal[
        "e0d4898141d8be07f4a4f1af7582791eae61994ced52bb7b3dba30a7ddda4b70"
    ] = "e0d4898141d8be07f4a4f1af7582791eae61994ced52bb7b3dba30a7ddda4b70"


class DemoTestLedgerEntry(FrozenModel):
    schema_version: Literal["aqse.network-demo.test-ledger.v1"] = (
        "aqse.network-demo.test-ledger.v1"
    )
    study_artifact_id: str = Field(pattern=r"^aqse-network-study-[a-f0-9]{16}$")
    sequence: int = Field(ge=0, le=2)
    event: Literal["sealed", "opened"]
    reason: str = Field(min_length=1, max_length=256)
    selection_freeze_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    previous_entry_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    entry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class LoadedStudyPartition(FrozenModel):
    partition: StudyPartition
    observations: tuple[NetworkStudyObservation, ...]
    labels: tuple[NetworkStudyLabel, ...] | None = None
