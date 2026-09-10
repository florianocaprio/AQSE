from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from app.training.models import DatasetPartition, FrozenModel

ENCODING_POLICY_ID = "aqse.tqk8.encoding.phase-direct.v1"
PHASE_WRAP_CONVENTION = "[-pi,pi);+pi->-pi"


class FittingPopulation(FrozenModel):
    population_id: str = Field(pattern=r"^aqse-fitting-population-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_partition: Literal[DatasetPartition.TRAIN] = DatasetPartition.TRAIN
    episode_count: int = Field(gt=0)
    accepted_window_count: int = Field(gt=0)


class TQKCompatibility(FrozenModel):
    circuit_id: Literal["aqse.author-tqk8.v1"] = "aqse.author-tqk8.v1"
    tqk8_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_count: Literal[8] = 8
    qubit_count: Literal[8] = 8
    parameter_count: Literal[16] = 16


class EncoderScientificArtifact(FrozenModel):
    schema_version: Literal["aqse.training-encoder.v1"] = "aqse.training-encoder.v1"
    artifact_id: str = Field(pattern=r"^aqse-encoder-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    encoding_policy_id: Literal["aqse.tqk8.encoding.phase-direct.v1"] = ENCODING_POLICY_ID
    source_dataset_id: str
    source_dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_partition: Literal[DatasetPartition.TRAIN] = DatasetPartition.TRAIN
    feature_profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_profile: dict[str, Any]
    feature_order: tuple[str, ...]
    feature_units: tuple[str, ...]
    phase_index: Literal[1] = 1
    phase_reference: Literal["window_start"] = "window_start"
    phase_wrap_convention: Literal["[-pi,pi);+pi->-pi"] = PHASE_WRAP_CONVENTION
    scaler_algorithm: Literal["protected.AngleScaler.fit.v1"] = (
        "protected.AngleScaler.fit.v1"
    )
    scaler_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    scaler_id: str = Field(pattern=r"^aqse-angle-scaler-[a-f0-9]{16}$")
    fitting_population: FittingPopulation
    fitted_mean: tuple[float, ...]
    fitted_scale: tuple[float, ...]
    phase_scaler_statistics_used: Literal[False] = False
    tqk_compatibility: TQKCompatibility
    effective_source_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_vector_contract(self) -> EncoderScientificArtifact:
        if len(self.feature_order) != 8 or len(self.feature_units) != 8:
            raise ValueError("encoder feature order and units must contain exactly 8 entries")
        if len(self.fitted_mean) != 8 or len(self.fitted_scale) != 8:
            raise ValueError("encoder fitted statistics must contain exactly 8 values")
        if any(value <= 0.0 for value in self.fitted_scale):
            raise ValueError("encoder fitted scales must be positive")
        if not self.effective_source_hashes:
            raise ValueError("encoder effective source hashes are required")
        return self


class EncoderExecutionMetadata(FrozenModel):
    schema_version: Literal["aqse.training-encoder-execution.v1"] = (
        "aqse.training-encoder-execution.v1"
    )
    artifact_id: str
    artifact_path: str
    created_at_utc: str
    repository_base_sha: str
    repository_dirty: bool | None
    runtime_versions: dict[str, str]


class EncodingInputContract(FrozenModel):
    source_dataset_id: str
    source_dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_profile: dict[str, Any]
    phase_index: Literal[1] = 1
    phase_reference: str
    encoding_policy_id: str
    scaler_id: str
    tqk_compatibility: TQKCompatibility


class BankRowReference(FrozenModel):
    episode_id: str
    lineage_id: str
    partition: Literal[DatasetPartition.TRAIN, DatasetPartition.VALIDATION]
    window_ordinal: Literal[9] = 9
    window_id: str


class QuantumBankArtifact(FrozenModel):
    schema_version: Literal["aqse.quantum-bank.v1"] = "aqse.quantum-bank.v1"
    artifact_id: str = Field(pattern=r"^aqse-(train|validation)-bank-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    bank_kind: Literal["train", "validation"]
    source_dataset_id: str
    source_dataset_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_partition: Literal[DatasetPartition.TRAIN, DatasetPartition.VALIDATION]
    encoder_artifact_id: str = Field(pattern=r"^aqse-encoder-[a-f0-9]{16}$")
    selection_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    selection_policy: Literal["aqse.fixed-central-window.v1"] = (
        "aqse.fixed-central-window.v1"
    )
    selection_seed: int | None
    window_ordinal: Literal[9] = 9
    row_count: int = Field(gt=0)
    lineage_count: int = Field(gt=0)
    class_balance: dict[str, int] | None
    rows: tuple[BankRowReference, ...]

    @model_validator(mode="after")
    def validate_rows(self) -> QuantumBankArtifact:
        if len(self.rows) != self.row_count:
            raise ValueError("quantum bank row count does not match its references")
        if len({item.lineage_id for item in self.rows}) != self.lineage_count:
            raise ValueError("quantum bank lineage count does not match its references")
        if len({item.episode_id for item in self.rows}) != self.row_count:
            raise ValueError("quantum bank must contain one row per distinct episode")
        if any(item.partition is not self.source_partition for item in self.rows):
            raise ValueError("quantum bank row partition is inconsistent")
        return self
