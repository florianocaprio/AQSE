from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from math import pi
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.features.models import (
    FeatureExtractionRequest,
    FeatureExtractionResponse,
    FeatureWindowConfiguration,
    MeasuredVectorSeries,
)
from app.features.windowed import extract_windowed_magnetometer_features
from app.network.models import (
    EnvironmentConfiguration,
    MeasurementMode,
    NetworkSessionConfiguration,
    NodeErrorConfiguration,
    PeriodicFieldConfiguration,
    SensorNodeConfiguration,
)
from app.network.simulation import NetworkSimulator
from app.training.canonical import canonical_json_bytes, scientific_digest
from app.training.models import (
    DatasetPartition,
    EpisodeCoverage,
    EpisodeLabel,
    EpisodePlan,
    LabeledEpisodePlan,
    NoiseRegime,
    SplitAssignment,
    WindowDisposition,
)
from app.training.splits import stratified_lineage_split

APPROVED_PILOT_SEED = 1_001_001
APPROVED_DEVELOPMENT_SEED = 1_001_002
APPROVED_SPLIT_SEED = 1_001_003
FIXED_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
SAMPLE_RATE_HZ = 100.0
DURATION_S = 10.0
SAMPLE_COUNT = 1_000
FIELD_NT = (20_000.0, 0.0, 45_000.0)
FEATURE_WINDOW = FeatureWindowConfiguration(duration_s=1.0, overlap_fraction=0.5)


@dataclass(frozen=True)
class GeneratedEpisode:
    plan: EpisodePlan
    label: EpisodeLabel
    time_s: NDArray[np.float64]
    measured_field_T: NDArray[np.float64]
    temperature_K: NDArray[np.float64]
    saturation_mask: NDArray[np.bool_]
    features: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    windows: tuple[WindowDisposition, ...]
    coverage: EpisodeCoverage
    feature_response: FeatureExtractionResponse

    @property
    def observation_digest(self) -> str:
        return scientific_digest(
            {"episode_id": self.plan.episode_id, "sensor_id": "M1"},
            {
                "time_s": self.time_s,
                "measured_field_T": self.measured_field_T,
                "temperature_K": self.temperature_K,
                "saturation_mask": self.saturation_mask,
            },
        )


@dataclass(frozen=True)
class DatasetBuild:
    kind: str
    master_seed: int
    split_seed: int | None
    episodes: tuple[GeneratedEpisode, ...]
    assignments: tuple[SplitAssignment, ...]

    @property
    def labels(self) -> tuple[EpisodeLabel, ...]:
        return tuple(episode.label for episode in self.episodes)

    @property
    def plans(self) -> tuple[EpisodePlan, ...]:
        return tuple(episode.plan for episode in self.episodes)

    @property
    def feature_profile_fingerprint(self) -> str:
        profile = self.episodes[0].feature_response.profile.model_dump(mode="json")
        return hashlib.sha256(canonical_json_bytes(profile)).hexdigest()


def _derived_seed(master_seed: int, namespace: str) -> int:
    payload = f"aqse-1d1:{master_seed}:{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _identity(prefix: str, master_seed: int, namespace: str) -> str:
    digest = hashlib.sha256(f"{master_seed}:{namespace}".encode()).hexdigest()[:16]
    return f"{prefix}-{digest}"


def build_episode_plans(
    *,
    master_seed: int,
    episodes_per_class: int,
    purpose: str,
) -> tuple[LabeledEpisodePlan, ...]:
    """Freeze lineages and interventions before generating any observations."""

    plans: list[LabeledEpisodePlan] = []
    for regime in (NoiseRegime.NOMINAL, NoiseRegime.ELEVATED):
        target = -1 if regime is NoiseRegime.NOMINAL else 1
        noise_bounds = (0.25, 0.75) if target == -1 else (1.5, 3.0)
        for ordinal in range(episodes_per_class):
            namespace = f"{purpose}:{regime.value}:{ordinal}"
            signal_seed = _derived_seed(master_seed, f"{namespace}:nuisance")
            generation_seed = _derived_seed(master_seed, f"{namespace}:measurement")
            nuisance_rng = np.random.default_rng(signal_seed)
            intervention_rng = np.random.default_rng(
                _derived_seed(master_seed, f"{namespace}:intervention")
            )
            episode_id = _identity("episode", master_seed, namespace)
            lineage_id = _identity("lineage", master_seed, f"{namespace}:lineage")
            plan = EpisodePlan(
                episode_id=episode_id,
                lineage_id=lineage_id,
                ordinal=ordinal,
                generation_seed=generation_seed,
                signal_seed=signal_seed,
                amplitude_nt=float(nuisance_rng.uniform(6.0, 10.0)),
                frequency_hz=float(nuisance_rng.uniform(4.0, 8.0)),
                phase_rad=float(nuisance_rng.uniform(-pi, pi)),
                temperature_k=float(nuisance_rng.uniform(292.15, 294.15)),
                white_noise_std_nt=float(intervention_rng.uniform(*noise_bounds)),
            )
            label = EpisodeLabel(
                episode_id=episode_id,
                lineage_id=lineage_id,
                target=target,
            )
            plans.append(LabeledEpisodePlan(plan=plan, label=label))
    return tuple(plans)


def _network_configuration(plan: EpisodePlan) -> NetworkSessionConfiguration:
    nt_to_t = 1.0e-9
    temperature = plan.temperature_k
    return NetworkSessionConfiguration(
        session_name=f"AQSE 1D.1 {plan.episode_id}",
        random_seed=plan.generation_seed,
        sampling_rate_Hz=SAMPLE_RATE_HZ,
        ui_refresh_rate_Hz=5.0,
        buffer_duration_s=DURATION_S,
        environment=EnvironmentConfiguration(
            uniform_field_T=tuple(value * nt_to_t for value in FIELD_NT),
            periodic_fields=(
                PeriodicFieldConfiguration(
                    source_id="controlled-x-harmonic",
                    amplitude_world_T=(plan.amplitude_nt * nt_to_t, 0.0, 0.0),
                    frequency_Hz=plan.frequency_hz,
                    phase_rad=plan.phase_rad,
                ),
            ),
        ),
        nodes=(
            SensorNodeConfiguration(
                sensor_id="M1",
                measurement_mode=MeasurementMode.VECTOR,
                position_m=(0.0, 0.0, 0.0),
                errors=NodeErrorConfiguration(
                    white_noise_std_T_per_sample=(plan.white_noise_std_nt * nt_to_t,) * 3,
                    initial_temperature_K=temperature,
                    ambient_temperature_K=temperature,
                    reference_temperature_K=temperature,
                    thermal_bias_T_per_K=(0.0, 0.0, 0.0),
                    bandwidth_Hz=None,
                    saturation_limit_T=800.0e-6,
                ),
            ),
        ),
    )


def generate_episode(specification: LabeledEpisodePlan) -> GeneratedEpisode:
    """Generate observable arrays and features; simulator truth is never archived."""

    plan = specification.plan
    simulator = NetworkSimulator(_network_configuration(plan))
    time_s = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE_HZ
    measured: list[tuple[float, float, float]] = []
    temperatures: list[float] = []
    saturation: list[tuple[bool, bool, bool]] = []
    for offset, sim_time_s in enumerate(time_s):
        frame = simulator.generate(
            session_id=plan.episode_id,
            frame_id=offset + 1,
            sim_time_s=float(sim_time_s),
            epoch_utc=FIXED_EPOCH,
            events=(),
        )
        reading = frame.observation.readings[0]
        if reading.components_T is None or reading.saturation_mask is None:
            raise RuntimeError("controlled 1D.1 episode produced a missing vector reading")
        measured.append(reading.components_T)
        temperatures.append(reading.observed_temperature_K)
        saturation.append(reading.saturation_mask)

    measured_array = np.asarray(measured, dtype=np.dtype("<f8"))
    temperature_array = np.asarray(temperatures, dtype=np.dtype("<f8"))
    saturation_array = np.asarray(saturation, dtype=np.bool_)
    if np.any(saturation_array):
        raise RuntimeError("controlled 1D.1 domain unexpectedly clipped a measurement")

    series = MeasuredVectorSeries(
        acquisition_id=plan.episode_id,
        sensor_id="M1",
        sampling_rate_hz=SAMPLE_RATE_HZ,
        time_s=time_s.tolist(),
        measured_field=[tuple(row) for row in measured_array],
        field_unit="T",
        temperature_k=temperature_array.tolist(),
        saturation_mask=[tuple(row) for row in saturation_array],
    )
    response = extract_windowed_magnetometer_features(
        FeatureExtractionRequest(series=series, channel="x", window=FEATURE_WINDOW)
    )
    features = np.asarray(
        [record.features.values for record in response.windows],
        dtype=np.dtype("<f8"),
    )
    valid_mask = np.asarray(
        [record.quality.valid_for_quantum for record in response.windows],
        dtype=np.bool_,
    )
    windows = tuple(
        WindowDisposition(
            window_id=record.window_id,
            episode_id=plan.episode_id,
            lineage_id=plan.lineage_id,
            start_index=record.start_index,
            end_index=record.end_index,
            valid_for_quantum=record.quality.valid_for_quantum,
            rejection_reasons=record.quality.flags,
        )
        for record in response.windows
    )
    reason_counts = Counter(reason for item in windows for reason in item.rejection_reasons)
    accepted = int(np.count_nonzero(valid_mask))
    coverage = EpisodeCoverage(
        episode_id=plan.episode_id,
        lineage_id=plan.lineage_id,
        proposed_windows=len(windows),
        accepted_windows=accepted,
        rejected_windows=len(windows) - accepted,
        coverage_fraction=accepted / len(windows),
        rejection_reasons=dict(sorted(reason_counts.items())),
    )
    return GeneratedEpisode(
        plan=plan,
        label=specification.label,
        time_s=time_s,
        measured_field_T=measured_array,
        temperature_K=temperature_array,
        saturation_mask=saturation_array,
        features=features,
        valid_mask=valid_mask,
        windows=windows,
        coverage=coverage,
        feature_response=response,
    )


def run_pilot(*, master_seed: int = APPROVED_PILOT_SEED) -> DatasetBuild:
    specifications = build_episode_plans(
        master_seed=master_seed,
        episodes_per_class=12,
        purpose="pilot",
    )
    episodes = tuple(generate_episode(specification) for specification in specifications)
    _validate_pilot(episodes)
    assignments = tuple(
        SplitAssignment(
            episode_id=episode.plan.episode_id,
            lineage_id=episode.plan.lineage_id,
            partition=DatasetPartition.PILOT,
        )
        for episode in episodes
    )
    return DatasetBuild(
        kind="pilot",
        master_seed=master_seed,
        split_seed=None,
        episodes=episodes,
        assignments=assignments,
    )


def _validate_pilot(episodes: tuple[GeneratedEpisode, ...]) -> None:
    for target in (-1, 1):
        selected = [episode for episode in episodes if episode.label.target == target]
        proposed = sum(episode.coverage.proposed_windows for episode in selected)
        accepted = sum(episode.coverage.accepted_windows for episode in selected)
        if any(episode.coverage.accepted_windows == 0 for episode in selected):
            raise RuntimeError(f"pilot target {target} contains an episode with zero valid windows")
        if accepted / proposed < 0.8:
            raise RuntimeError(f"pilot target {target} retained less than 80% of windows")


def build_development_dataset(
    *,
    master_seed: int = APPROVED_DEVELOPMENT_SEED,
    split_seed: int = APPROVED_SPLIT_SEED,
) -> DatasetBuild:
    specifications = build_episode_plans(
        master_seed=master_seed,
        episodes_per_class=60,
        purpose="development",
    )
    labels = tuple(item.label for item in specifications)
    assignments = stratified_lineage_split(labels, split_seed=split_seed)
    episodes = tuple(generate_episode(specification) for specification in specifications)
    return DatasetBuild(
        kind="development",
        master_seed=master_seed,
        split_seed=split_seed,
        episodes=episodes,
        assignments=assignments,
    )


def dataset_identity_metadata(build: DatasetBuild) -> dict[str, Any]:
    """Return stable scientific metadata; runtime and HMAC fields are excluded."""

    return {
        "schema_version": "aqse.dataset-scientific-identity.v1",
        "kind": build.kind,
        "master_seed": build.master_seed,
        "split_seed": build.split_seed,
        "fixed_epoch_utc": FIXED_EPOCH.isoformat().replace("+00:00", "Z"),
        "sampling_rate_hz": SAMPLE_RATE_HZ,
        "duration_s": DURATION_S,
        "sample_count": SAMPLE_COUNT,
        "feature_profile": build.episodes[0].feature_response.profile.model_dump(mode="json"),
        "plans": [plan.model_dump(mode="json") for plan in build.plans],
        "assignments": [item.model_dump(mode="json") for item in build.assignments],
        "label_policy": "aqse.white-noise-regime.v1",
        "label_artifact_sha256": hashlib.sha256(
            canonical_json_bytes([label.model_dump(mode="json") for label in build.labels])
        ).hexdigest(),
        "encoding": {
            "expected": "aqse.tqk8.encoding.phase-direct.v1",
            "scaler_id": None,
            "theta_id": None,
            "reference_bank_id": None,
            "model_id": None,
            "fit_state": "not-fitted",
        },
    }
