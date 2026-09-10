from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

STATE8_SAMPLING_RATE_HZ = 100.0
STATE8_REFERENCE_DURATION_S = 8.0
STATE8_WINDOW_DURATION_S = 4.0
STATE8_HOP_DURATION_S = 1.0
STATE8_REFERENCE_SAMPLES = 800
STATE8_WINDOW_SAMPLES = 400
STATE8_HOP_SAMPLES = 100

LOCAL_STATE8_PROFILE_ID = "aqse.local-state8.v1"
NETWORK_STATE8_PROFILE_ID = "aqse.network-state8.v1"

State8ProfileId = Literal[
    "aqse.local-state8.v1",
    "aqse.network-state8.v1",
]
State8Scope = Literal["local", "network"]
NullableFiniteFloat = Annotated[float | None, Field(allow_inf_nan=False)]
State8Values = tuple[
    NullableFiniteFloat,
    NullableFiniteFloat,
    NullableFiniteFloat,
    NullableFiniteFloat,
    NullableFiniteFloat,
    NullableFiniteFloat,
    NullableFiniteFloat,
    NullableFiniteFloat,
]

COMMON_FEATURE_NAMES = (
    "mean_north_anomaly",
    "north_anomaly_mad",
    "north_anomaly_slope",
    "lower_to_upper_band_power_db",
    "mean_temperature_delta",
)
LOCAL_FEATURE_NAMES = (
    *COMMON_FEATURE_NAMES,
    "successive_difference_rms",
    "lag1_pearson_correlation",
    "received_sample_fraction",
)
NETWORK_FEATURE_NAMES = (
    *COMMON_FEATURE_NAMES,
    "peer_median_residual_rms",
    "signed_mean_peer_correlation",
    "received_sample_fraction",
)
STATE8_FEATURE_UNITS = (
    "nT",
    "nT",
    "nT/s",
    "dB",
    "K",
    "nT",
    "dimensionless",
    "dimensionless",
)


class FrozenState8Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class State8FeatureProfile(FrozenState8Model):
    """Frozen observable feature semantics for one state8 model family."""

    schema_version: Literal["aqse.state8-feature-profile.v1"] = (
        "aqse.state8-feature-profile.v1"
    )
    profile_id: State8ProfileId
    scope: State8Scope
    extractor_version: Literal["1.0.0"] = "1.0.0"
    sensor_type: Literal["magnetometer_vector3"] = "magnetometer_vector3"
    coordinate_frame: Literal["world_ned_north"] = "world_ned_north"
    measurement_source: Literal["ObservationFrame"] = "ObservationFrame"
    calibration_policy: Literal["declared_version_frozen"] = (
        "declared_version_frozen"
    )
    sampling_rate_hz: Literal[100.0] = STATE8_SAMPLING_RATE_HZ
    reference_duration_s: Literal[8.0] = STATE8_REFERENCE_DURATION_S
    window_duration_s: Literal[4.0] = STATE8_WINDOW_DURATION_S
    hop_duration_s: Literal[1.0] = STATE8_HOP_DURATION_S
    reference_samples: Literal[800] = STATE8_REFERENCE_SAMPLES
    window_samples: Literal[400] = STATE8_WINDOW_SAMPLES
    hop_samples: Literal[100] = STATE8_HOP_SAMPLES
    lower_band_hz: tuple[Literal[0.25], Literal[2.0]] = (0.25, 2.0)
    upper_band_hz: tuple[Literal[2.0], Literal[20.0]] = (2.0, 20.0)
    band_boundary_policy: Literal["lower:[0.25,2);upper:[2,20]"] = (
        "lower:[0.25,2);upper:[2,20]"
    )
    spectral_window: Literal["hann"] = "hann"
    spectral_detrend: Literal["linear_ols"] = "linear_ols"
    spectral_power_floor: Literal[1e-12] = 1.0e-12
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...] = STATE8_FEATURE_UNITS
    minimum_network_nodes: int

    @model_validator(mode="after")
    def validate_profile_semantics(self) -> State8FeatureProfile:
        expected_scope = (
            "local" if self.profile_id == LOCAL_STATE8_PROFILE_ID else "network"
        )
        expected_names = (
            LOCAL_FEATURE_NAMES
            if self.profile_id == LOCAL_STATE8_PROFILE_ID
            else NETWORK_FEATURE_NAMES
        )
        expected_nodes = 1 if expected_scope == "local" else 3
        if self.scope != expected_scope:
            raise ValueError("state8 profile ID and scope are incompatible")
        if self.feature_names != expected_names:
            raise ValueError("state8 feature names are incompatible with the profile")
        if self.feature_units != STATE8_FEATURE_UNITS:
            raise ValueError("state8 feature units differ from the frozen contract")
        if self.minimum_network_nodes != expected_nodes:
            raise ValueError("state8 minimum node count differs from the profile")
        return self


class State8NodeReference(FrozenState8Model):
    sensor_id: str = Field(min_length=1, max_length=128)
    calibration_version: str | None = None
    expected_sample_count: Literal[800] = STATE8_REFERENCE_SAMPLES
    usable_sample_count: int = Field(ge=0, le=STATE8_REFERENCE_SAMPLES)
    north_reference_nt: NullableFiniteFloat
    temperature_reference_k: NullableFiniteFloat
    valid: bool
    flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_reference_availability(self) -> State8NodeReference:
        values_available = (
            self.north_reference_nt is not None
            and self.temperature_reference_k is not None
            and self.calibration_version is not None
        )
        if self.valid != values_available:
            raise ValueError("reference validity and nullable values are inconsistent")
        if self.valid and self.usable_sample_count != self.expected_sample_count:
            raise ValueError("a valid reference must contain every expected sample")
        if not self.valid and not self.flags:
            raise ValueError("an invalid reference must explain its invalidity")
        return self


class State8ObservedReference(FrozenState8Model):
    schema_version: Literal["aqse.state8-observed-reference.v1"] = (
        "aqse.state8-observed-reference.v1"
    )
    reference_id: str = Field(pattern=r"^aqse-state8-reference-[a-f0-9]{16}$")
    content_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    session_id: str
    reference_start_time_s: float
    reference_end_exclusive_s: float
    sampling_rate_hz: Literal[100.0] = STATE8_SAMPLING_RATE_HZ
    source_frame_ids: tuple[int, ...]
    node_order: tuple[str, ...]
    nodes: tuple[State8NodeReference, ...]

    @model_validator(mode="after")
    def validate_node_identity(self) -> State8ObservedReference:
        if not self.node_order or tuple(sorted(self.node_order)) != self.node_order:
            raise ValueError("reference node_order must be non-empty and sorted")
        if len(set(self.node_order)) != len(self.node_order):
            raise ValueError("reference node IDs must be unique")
        if tuple(node.sensor_id for node in self.nodes) != self.node_order:
            raise ValueError("reference nodes must follow node_order")
        if len(self.source_frame_ids) > STATE8_REFERENCE_SAMPLES:
            raise ValueError("reference contains too many source frames")
        return self


class State8FeatureQuality(FrozenState8Model):
    valid_for_quantum: bool
    flags: tuple[str, ...]
    per_feature_valid: tuple[bool, bool, bool, bool, bool, bool, bool, bool]
    expected_sample_count: Literal[400] = STATE8_WINDOW_SAMPLES
    received_sample_count: int = Field(ge=0, le=STATE8_WINDOW_SAMPLES)
    usable_sample_count: int = Field(ge=0, le=STATE8_WINDOW_SAMPLES)

    @model_validator(mode="after")
    def validate_quality_summary(self) -> State8FeatureQuality:
        if self.valid_for_quantum != all(self.per_feature_valid):
            raise ValueError("state8 quantum validity must match per-feature validity")
        if self.usable_sample_count > self.received_sample_count:
            raise ValueError("usable samples cannot exceed received samples")
        if self.valid_for_quantum and self.flags:
            raise ValueError("valid state8 records cannot carry invalidity flags")
        return self


class State8FeatureRecord(FrozenState8Model):
    schema_version: Literal["aqse.state8-feature-record.v1"] = (
        "aqse.state8-feature-record.v1"
    )
    window_id: str
    session_id: str
    sensor_id: str
    profile_id: State8ProfileId
    profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    reference_id: str = Field(pattern=r"^aqse-state8-reference-[a-f0-9]{16}$")
    start_time_s: float
    end_exclusive_time_s: float
    source_frame_ids: tuple[int, ...]
    peer_sensor_ids: tuple[str, ...]
    values: State8Values
    quality: State8FeatureQuality

    @model_validator(mode="after")
    def validate_values_and_quality(self) -> State8FeatureRecord:
        available = tuple(value is not None for value in self.values)
        if available != self.quality.per_feature_valid:
            raise ValueError("nullable feature values and validity mask are inconsistent")
        if self.end_exclusive_time_s <= self.start_time_s:
            raise ValueError("state8 window end must follow its start")
        if self.profile_id == LOCAL_STATE8_PROFILE_ID and self.peer_sensor_ids:
            raise ValueError("local state8 records cannot name peer sensors")
        if self.profile_id == NETWORK_STATE8_PROFILE_ID:
            if self.sensor_id in self.peer_sensor_ids:
                raise ValueError("network state8 peer set cannot contain the focal node")
            if tuple(sorted(self.peer_sensor_ids)) != self.peer_sensor_ids:
                raise ValueError("network state8 peer IDs must be sorted")
        return self


class State8ExtractionResponse(FrozenState8Model):
    profile: State8FeatureProfile
    profile_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    reference: State8ObservedReference
    node_order: tuple[str, ...]
    records: tuple[State8FeatureRecord, ...]

    @model_validator(mode="after")
    def validate_response_identity(self) -> State8ExtractionResponse:
        if self.node_order != self.reference.node_order:
            raise ValueError("extraction node order differs from its observed reference")
        if any(record.profile_id != self.profile.profile_id for record in self.records):
            raise ValueError("extraction contains a record from an incompatible profile")
        if any(
            record.profile_fingerprint != self.profile_fingerprint
            for record in self.records
        ):
            raise ValueError("extraction contains a mismatched profile fingerprint")
        return self
