from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class NoiseRegime(str, Enum):
    NOMINAL = "nominal"
    ELEVATED = "elevated"


class DatasetPartition(str, Enum):
    PILOT = "pilot"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class EpisodePlan(FrozenModel):
    episode_id: str = Field(pattern=r"^episode-[a-f0-9]{16}$")
    lineage_id: str = Field(pattern=r"^lineage-[a-f0-9]{16}$")
    ordinal: int = Field(ge=0)
    generation_seed: int = Field(ge=0, le=2**32 - 1)
    signal_seed: int = Field(ge=0, le=2**32 - 1)
    amplitude_nt: float = Field(ge=6.0, le=10.0)
    frequency_hz: float = Field(ge=4.0, le=8.0)
    phase_rad: float = Field(ge=-3.141592653589793, lt=3.141592653589793)
    temperature_k: float = Field(ge=292.15, le=294.15)
    white_noise_std_nt: float = Field(ge=0.25, le=3.0)


class EpisodeLabel(FrozenModel):
    episode_id: str
    lineage_id: str
    target: Literal[-1, 1]
    origin_policy: Literal["aqse.white-noise-regime.v1"] = "aqse.white-noise-regime.v1"


class LabeledEpisodePlan(FrozenModel):
    plan: EpisodePlan
    label: EpisodeLabel

    @model_validator(mode="after")
    def validate_identity(self) -> LabeledEpisodePlan:
        if self.plan.episode_id != self.label.episode_id:
            raise ValueError("plan and label episode identifiers must match")
        if self.plan.lineage_id != self.label.lineage_id:
            raise ValueError("plan and label lineage identifiers must match")
        return self


class SplitAssignment(FrozenModel):
    episode_id: str
    lineage_id: str
    partition: DatasetPartition


class ObservationInterval(FrozenModel):
    episode_id: str
    lineage_id: str
    sensor_id: str
    start_index: int = Field(ge=0)
    end_index: int = Field(gt=0)
    source_id: str | None = None
    relation: Literal["original", "replay", "paired"] = "original"

    @model_validator(mode="after")
    def validate_interval(self) -> ObservationInterval:
        if self.end_index <= self.start_index:
            raise ValueError("observation interval end must follow its start")
        return self


class WindowDisposition(FrozenModel):
    window_id: str
    episode_id: str
    lineage_id: str
    start_index: int = Field(ge=0)
    end_index: int = Field(gt=0)
    valid_for_quantum: bool
    rejection_reasons: tuple[str, ...] = ()


class EpisodeCoverage(FrozenModel):
    episode_id: str
    lineage_id: str
    proposed_windows: int = Field(gt=0)
    accepted_windows: int = Field(ge=0)
    rejected_windows: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    rejection_reasons: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_window_accounting(self) -> EpisodeCoverage:
        if self.accepted_windows + self.rejected_windows != self.proposed_windows:
            raise ValueError("accepted and rejected windows must cover every proposal")
        return self


class PartitionSummary(FrozenModel):
    partition: DatasetPartition
    episode_count: int = Field(ge=0)
    lineage_count: int = Field(ge=0)
    proposed_window_count: int = Field(ge=0)
    accepted_window_count: int | None = Field(default=None, ge=0)
    rejected_window_count: int | None = Field(default=None, ge=0)


class FileRecord(FrozenModel):
    relative_path: str
    byte_count: int = Field(ge=0)
    file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    order: Literal["C"] | None = None
    endianness: Literal["little", "not-applicable"] | None = None


class LegacyDatasetManifest(FrozenModel):
    schema_version: Literal["aqse.dataset-manifest.v1"] = "aqse.dataset-manifest.v1"
    dataset_id: str
    kind: Literal["pilot", "development"]
    scientific_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    generation_policy: Literal["aqse.milestone-1d1.white-noise.v1"] = (
        "aqse.milestone-1d1.white-noise.v1"
    )
    split_policy: Literal["pilot-only", "stratified-lineage-60-20-20"]
    fixed_epoch_utc: Literal["2026-01-01T00:00:00Z"] = "2026-01-01T00:00:00Z"
    partition_summaries: tuple[PartitionSummary, ...]
    files: tuple[FileRecord, ...]
    test_state: Literal["not-applicable", "sealed"]
    expected_encoding: Literal["aqse.tqk8.encoding.phase-direct.v1"] = (
        "aqse.tqk8.encoding.phase-direct.v1"
    )
    scaler_id: None = None
    theta_id: None = None
    reference_bank_id: None = None
    model_id: None = None
    fit_state: Literal["not-fitted"] = "not-fitted"


class DatasetManifest(FrozenModel):
    schema_version: Literal["aqse.dataset-manifest.v2"] = "aqse.dataset-manifest.v2"
    dataset_id: str
    kind: Literal["pilot", "development"]
    scientific_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    numeric_content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    software_provenance_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    generation_policy: Literal["aqse.milestone-1d1.white-noise.v1"] = (
        "aqse.milestone-1d1.white-noise.v1"
    )
    split_policy: Literal["pilot-only", "stratified-lineage-60-20-20"]
    fixed_epoch_utc: Literal["2026-01-01T00:00:00Z"] = "2026-01-01T00:00:00Z"
    partition_summaries: tuple[PartitionSummary, ...]
    files: tuple[FileRecord, ...]
    test_state: Literal["not-applicable", "sealed"]
    expected_encoding: Literal["aqse.tqk8.encoding.phase-direct.v1"] = (
        "aqse.tqk8.encoding.phase-direct.v1"
    )
    scaler_id: None = None
    theta_id: None = None
    reference_bank_id: None = None
    model_id: None = None
    fit_state: Literal["not-fitted"] = "not-fitted"


class RawSourceReference(FrozenModel):
    source_id: str = Field(min_length=1, max_length=160)
    sensor_id: str = Field(min_length=1, max_length=128)
    start_index: int = Field(ge=0)
    end_index: int = Field(gt=0)
    relation: Literal["original", "replay", "paired"] = "original"

    @model_validator(mode="after")
    def validate_interval(self) -> RawSourceReference:
        if self.end_index <= self.start_index:
            raise ValueError("raw source interval end must follow its start")
        return self


class ArchivedWindowRecord(FrozenModel):
    window_id: str
    acquisition_id: str
    sensor_id: str
    start_index: int = Field(ge=0)
    end_index: int = Field(gt=0)
    start_time_s: float
    end_time_s: float
    center_time_s: float
    quality: dict[str, Any]


class ObservableProvenance(FrozenModel):
    schema_version: Literal["aqse.observable-provenance.v1"] = "aqse.observable-provenance.v1"
    field_unit: Literal["T"] = "T"
    temperature_unit: Literal["K"] = "K"
    time_unit: Literal["s"] = "s"
    calibration_id: Literal["identity-calibration.v1"] = "identity-calibration.v1"
    pose_id: Literal["world-aligned-origin.v1"] = "world-aligned-origin.v1"
    feature_profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    software_provenance_ref: Literal["provenance/software.json"] = "provenance/software.json"


class PartitionMetadataV2(FrozenModel):
    schema_version: Literal["aqse.partition-metadata.v2"] = "aqse.partition-metadata.v2"
    partition: DatasetPartition
    episode_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    profile: dict[str, Any]
    observable_provenance: ObservableProvenance
    raw_sources: tuple[RawSourceReference, ...]
    coverage: tuple[EpisodeCoverage, ...]
    windows: tuple[tuple[ArchivedWindowRecord, ...], ...]


class SoftwareProvenance(FrozenModel):
    schema_version: Literal["aqse.software-provenance.v1"] = "aqse.software-provenance.v1"
    algorithm_ids: tuple[str, ...]
    unit_conversions: dict[str, str]
    calibration_id: Literal["identity-calibration.v1"] = "identity-calibration.v1"
    pose_id: Literal["world-aligned-origin.v1"] = "world-aligned-origin.v1"
    repository_base_sha: str
    repository_dirty: bool | None
    source_hashes: dict[str, str]
    runtime_versions: dict[str, str]


class ExecutionMetadata(FrozenModel):
    schema_version: Literal["aqse.dataset-execution.v1"] = "aqse.dataset-execution.v1"
    dataset_id: str
    artifact_path: str
    written_at_utc: str
    duration_ms: float = Field(ge=0.0)
    peak_memory_bytes: int = Field(ge=0)
    total_bytes: int = Field(ge=0)


class TestAccessAuthorization(FrozenModel):
    reason: str = Field(min_length=1, max_length=300)
    actor: str = Field(min_length=1, max_length=160)
    fixture_only: bool = False


class TestAccessLedgerEntry(FrozenModel):
    schema_version: Literal["aqse.test-access-ledger.v2"] = "aqse.test-access-ledger.v2"
    sequence: int = Field(ge=0)
    event: Literal["sealed", "opened"]
    dataset_id: str
    scientific_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    occurred_at_utc: str
    actor: str
    reason: str
    fixture_only: bool
    previous_entry_sha256: str | None
    entry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
