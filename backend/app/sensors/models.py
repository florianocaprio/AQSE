from __future__ import annotations

from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    computed_field,
    model_validator,
)

MAX_ACQUISITION_SAMPLES = 20_000
MIN_ACQUISITION_SAMPLES = 16
FEATURE_VECTOR_SIZE = 8
MAX_ABSOLUTE_FIELD_NT = 1.0e12
MAX_ABSOLUTE_PHASE_RAD = 1.0e6
MAX_TEMPERATURE_K = 1.0e6


class SensorConfiguration(BaseModel):
    """Fields shared by configurable, finite-duration sensor sources."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    sensor_id: str = Field(
        default="magnetometer-001",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    duration: float = Field(default=2.0, gt=0.0, le=300.0)
    sampling_rate: float = Field(default=200.0, gt=0.0, le=100_000.0)
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @property
    def number_of_samples(self) -> int:
        """Return the number of uniformly spaced samples in the acquisition."""

        return int(round(self.duration * self.sampling_rate))

    @model_validator(mode="after")
    def validate_sample_count(self) -> SensorConfiguration:
        sample_count = self.number_of_samples
        if sample_count < MIN_ACQUISITION_SAMPLES:
            raise ValueError(
                "duration and sampling_rate must produce at least "
                f"{MIN_ACQUISITION_SAMPLES} samples"
            )
        if sample_count > MAX_ACQUISITION_SAMPLES:
            raise ValueError(
                "duration and sampling_rate may produce at most "
                f"{MAX_ACQUISITION_SAMPLES} samples"
            )
        return self


class MagnetometerConfiguration(SensorConfiguration):
    """Configuration for the synthetic quantum magnetometer demonstrator.

    Magnetic-field values use nanotesla (nT), time uses seconds (s), frequency
    uses hertz (Hz), drift uses nT/s, phase uses radians and temperature uses
    kelvin (K).
    """

    background_field: float = Field(
        default=50_000.0,
        ge=-MAX_ABSOLUTE_FIELD_NT,
        le=MAX_ABSOLUTE_FIELD_NT,
    )
    amplitude: float = Field(default=100.0, ge=0.0, le=MAX_ABSOLUTE_FIELD_NT)
    frequency: float = Field(default=8.0, gt=0.0)
    phase: float = Field(
        default=0.0,
        ge=-MAX_ABSOLUTE_PHASE_RAD,
        le=MAX_ABSOLUTE_PHASE_RAD,
    )
    drift_rate: float = Field(
        default=0.5,
        ge=-MAX_ABSOLUTE_FIELD_NT,
        le=MAX_ABSOLUTE_FIELD_NT,
    )
    noise_std: float = Field(default=2.0, ge=0.0, le=MAX_ABSOLUTE_FIELD_NT)
    temperature: float = Field(default=293.15, gt=0.0, le=MAX_TEMPERATURE_K)
    anomaly_enabled: bool = False
    anomaly_time: float | None = 1.0
    anomaly_amplitude: float = Field(
        default=40.0,
        ge=-MAX_ABSOLUTE_FIELD_NT,
        le=MAX_ABSOLUTE_FIELD_NT,
    )

    @model_validator(mode="after")
    def validate_magnetometer_parameters(self) -> MagnetometerConfiguration:
        if self.frequency >= self.sampling_rate / 2.0:
            raise ValueError("frequency must be below the Nyquist frequency")

        if self.anomaly_enabled:
            if self.anomaly_time is None:
                raise ValueError("anomaly_time is required when anomaly_enabled is true")
            if not 0.0 <= self.anomaly_time < self.duration:
                raise ValueError(
                    "anomaly_time must be within the acquisition duration"
                )
        return self


class SensorAcquisition(BaseModel):
    """Generic raw sensor acquisition with JSON-serializable metadata."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    sensor_id: str
    sensor_type: str
    timestamp: datetime
    sampling_rate: float = Field(gt=0.0)
    time: list[float] = Field(min_length=2)
    signal: list[float] = Field(min_length=2)
    time_unit: str
    physical_unit: str
    configuration: dict[str, JsonValue]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @computed_field(return_type=int)
    @property
    def sample_count(self) -> int:
        return len(self.signal)

    @model_validator(mode="after")
    def validate_raw_vectors(self) -> SensorAcquisition:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")
        if len(self.time) != len(self.signal):
            raise ValueError("time and signal vectors must have the same length")
        if any(right <= left for left, right in zip(self.time, self.time[1:])):
            raise ValueError("time values must be strictly increasing")
        return self


class FeatureVector(BaseModel):
    """Ordered, eight-dimensional feature vector for an AQSE pipeline."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    names: tuple[str, ...]
    values: tuple[float, ...]
    units: tuple[str, ...]

    @model_validator(mode="after")
    def validate_dimensions(self) -> FeatureVector:
        if len(self.names) != FEATURE_VECTOR_SIZE:
            raise ValueError(f"feature names must contain exactly {FEATURE_VECTOR_SIZE} items")
        if len(self.values) != FEATURE_VECTOR_SIZE:
            raise ValueError(f"feature values must contain exactly {FEATURE_VECTOR_SIZE} items")
        if len(self.units) != FEATURE_VECTOR_SIZE:
            raise ValueError(f"feature units must contain exactly {FEATURE_VECTOR_SIZE} items")
        if len(set(self.names)) != FEATURE_VECTOR_SIZE:
            raise ValueError("feature names must be unique")
        return self


class FrequencySpectrum(BaseModel):
    """One-sided power spectral density for a uniformly sampled signal."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    frequencies: list[float]
    power_spectral_density: list[float]
    frequency_unit: str = "Hz"
    power_unit: str = "nT²/Hz"

    @model_validator(mode="after")
    def validate_spectrum_vectors(self) -> FrequencySpectrum:
        if len(self.frequencies) != len(self.power_spectral_density):
            raise ValueError(
                "frequencies and power_spectral_density must have the same length"
            )
        if not self.frequencies:
            raise ValueError("frequency spectrum cannot be empty")
        return self


class MagnetometerSimulationResponse(BaseModel):
    acquisition: SensorAcquisition
    spectrum: FrequencySpectrum
    features: FeatureVector


class SensorDescriptor(BaseModel):
    sensor_type: str
    display_name: str
    simulation: bool
    physical_unit: str


class SensorCatalogResponse(BaseModel):
    sensors: list[SensorDescriptor]
