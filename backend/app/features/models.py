from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.preprocessing.features import (
    MAGNETOMETER_FEATURE_NAMES,
    MAGNETOMETER_FEATURE_UNITS,
)
from app.sensors.models import MAX_ACQUISITION_SAMPLES, FeatureVector

FeatureChannel = Literal["x", "y", "z", "magnitude"]
FieldUnit = Literal["T", "nT"]
QualityStatus = Literal["valid", "warning", "invalid"]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class FeatureWindowConfiguration(BaseModel):
    """Causal complete-window configuration; no future padding is permitted."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    duration_s: float = Field(default=1.0, gt=0.0, le=120.0)
    overlap_fraction: float = Field(default=0.5, ge=0.0, lt=1.0)
    minimum_cycles: Literal[2.0] = 2.0
    minimum_peak_prominence_db: Literal[6.0] = 6.0
    minimum_snr_db: Literal[0.0] = 0.0
    reject_clipped: Literal[True] = True


class MeasuredVectorSeries(BaseModel):
    """Observable-only vector series accepted by the feature service.

    Generator truth is intentionally absent from this schema.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    acquisition_id: str = Field(min_length=1, max_length=160)
    sensor_id: str = Field(min_length=1, max_length=128)
    sampling_rate_hz: float = Field(gt=0.0, le=100_000.0)
    time_s: list[FiniteFloat] = Field(min_length=16, max_length=MAX_ACQUISITION_SAMPLES)
    measured_field: list[tuple[FiniteFloat, FiniteFloat, FiniteFloat]] = Field(
        min_length=16,
        max_length=MAX_ACQUISITION_SAMPLES,
    )
    field_unit: FieldUnit = "nT"
    temperature_k: list[FiniteFloat] = Field(
        min_length=16,
        max_length=MAX_ACQUISITION_SAMPLES,
    )
    saturation_mask: list[tuple[bool, bool, bool]] = Field(
        min_length=16,
        max_length=MAX_ACQUISITION_SAMPLES,
    )

    @model_validator(mode="after")
    def validate_synchronized_observations(self) -> MeasuredVectorSeries:
        lengths = {
            len(self.time_s),
            len(self.measured_field),
            len(self.temperature_k),
            len(self.saturation_mask),
        }
        if len(lengths) != 1:
            raise ValueError("all measured observation arrays must have the same length")
        if any(right <= left for left, right in zip(self.time_s, self.time_s[1:])):
            raise ValueError("time_s must be strictly increasing")
        expected_period = 1.0 / self.sampling_rate_hz
        tolerance = max(1.0e-12, expected_period * 1.0e-9)
        if any(
            abs((right - left) - expected_period) > tolerance
            for left, right in zip(self.time_s, self.time_s[1:])
        ):
            raise ValueError(
                "time_s must be uniformly spaced and consistent with sampling_rate_hz"
            )
        return self


class FeatureProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    profile_id: str = "aqse.magnetometer.harmonic.v1"
    extractor_version: str = "1.0.0"
    sensor_type: str = "magnetometer_vector3"
    readout: str = "measured_field"
    channel: FeatureChannel
    feature_names: tuple[str, ...] = MAGNETOMETER_FEATURE_NAMES
    feature_units: tuple[str, ...] = MAGNETOMETER_FEATURE_UNITS
    sampling_rate_hz: float = Field(gt=0.0, le=100_000.0)
    window_duration_s: float = Field(gt=0.0, le=120.0)
    overlap_fraction: float = Field(ge=0.0, lt=1.0)
    window_samples: int = Field(ge=16, le=MAX_ACQUISITION_SAMPLES)
    hop_samples: int = Field(ge=1, le=MAX_ACQUISITION_SAMPLES)
    phase_reference: str = "window_start"
    quality_policy_version: str = "aqse.harmonic-quality.v1"
    minimum_cycles: Literal[2.0] = 2.0
    minimum_peak_prominence_db: Literal[6.0] = 6.0
    minimum_snr_db: Literal[0.0] = 0.0
    maximum_saturation_fraction: Literal[0.0] = 0.0

    @model_validator(mode="after")
    def validate_profile_contract(self) -> FeatureProfile:
        if self.profile_id != "aqse.magnetometer.harmonic.v1":
            raise ValueError("unsupported feature profile")
        if self.sensor_type != "magnetometer_vector3":
            raise ValueError("feature profile sensor_type must be magnetometer_vector3")
        if self.readout != "measured_field":
            raise ValueError("feature profile readout must be measured_field")
        if self.quality_policy_version != "aqse.harmonic-quality.v1":
            raise ValueError("unsupported feature quality policy")
        if self.feature_names != MAGNETOMETER_FEATURE_NAMES:
            raise ValueError("feature profile names must match the canonical 8D contract")
        if self.feature_units != MAGNETOMETER_FEATURE_UNITS:
            raise ValueError("feature profile units must match the canonical 8D contract")
        if not 1 <= self.hop_samples <= self.window_samples:
            raise ValueError("hop_samples must be between 1 and window_samples")
        expected_window_samples = int(
            round(self.window_duration_s * self.sampling_rate_hz)
        )
        if self.window_samples != expected_window_samples:
            raise ValueError(
                "window_samples must match window_duration_s and sampling_rate_hz"
            )
        expected_hop = max(
            1,
            int(round(self.window_samples * (1.0 - self.overlap_fraction))),
        )
        if self.hop_samples != expected_hop:
            raise ValueError("hop_samples must match window_samples and overlap_fraction")
        return self


class FeatureQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    status: QualityStatus
    valid_for_quantum: bool
    flags: tuple[str, ...]
    per_feature_valid: tuple[bool, bool, bool, bool, bool, bool, bool, bool]
    sample_count: int
    saturation_fraction: float = Field(ge=0.0, le=1.0)
    cycles_in_window: float = Field(ge=0.0)
    peak_prominence_db: float = Field(ge=-300.0, le=300.0)
    signal_standard_deviation_nt: float = Field(ge=0.0)


class WindowFeatureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    window_id: str
    acquisition_id: str
    sensor_id: str
    start_index: int = Field(ge=0)
    end_index: int = Field(gt=0)
    start_time_s: float
    end_time_s: float
    center_time_s: float
    features: FeatureVector
    quality: FeatureQuality
    provenance_token: str


class FeatureExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    series: MeasuredVectorSeries
    channel: FeatureChannel = "magnitude"
    window: FeatureWindowConfiguration = Field(
        default_factory=FeatureWindowConfiguration
    )


class FeatureExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    profile: FeatureProfile
    windows: list[WindowFeatureRecord]
    valid_window_count: int = Field(ge=0)
    invalid_window_count: int = Field(ge=0)
    discarded_sample_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
