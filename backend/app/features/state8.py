from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import log10, sqrt
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from app.features.state8_models import (
    LOCAL_FEATURE_NAMES,
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_FEATURE_NAMES,
    NETWORK_STATE8_PROFILE_ID,
    STATE8_HOP_SAMPLES,
    STATE8_REFERENCE_DURATION_S,
    STATE8_REFERENCE_SAMPLES,
    STATE8_SAMPLING_RATE_HZ,
    STATE8_WINDOW_DURATION_S,
    STATE8_WINDOW_SAMPLES,
    State8ExtractionResponse,
    State8FeatureProfile,
    State8FeatureQuality,
    State8FeatureRecord,
    State8NodeReference,
    State8ObservedReference,
    State8ProfileId,
    State8Values,
)
from app.network.models import MeasurementMode, ObservationFrame, QualityFlag, SensorReading
from app.network.physics import quaternion_world_to_sensor_matrix

TESLA_TO_NANOTESLA = 1.0e9
TIME_ALIGNMENT_TOLERANCE_S = 1.0e-8
SPECTRAL_POWER_FLOOR = 1.0e-12


def _canonical_digest(value: Any) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


_LOCAL_PROFILE = State8FeatureProfile(
    profile_id=LOCAL_STATE8_PROFILE_ID,
    scope="local",
    feature_names=LOCAL_FEATURE_NAMES,
    minimum_network_nodes=1,
)
_NETWORK_PROFILE = State8FeatureProfile(
    profile_id=NETWORK_STATE8_PROFILE_ID,
    scope="network",
    feature_names=NETWORK_FEATURE_NAMES,
    minimum_network_nodes=3,
)


def state8_profile(profile_id: State8ProfileId | str) -> State8FeatureProfile:
    """Return the exact frozen profile; similarly shaped profiles are not coerced."""

    if profile_id == LOCAL_STATE8_PROFILE_ID:
        return _LOCAL_PROFILE
    if profile_id == NETWORK_STATE8_PROFILE_ID:
        return _NETWORK_PROFILE
    raise ValueError(f"unsupported state8 feature profile: {profile_id}")


def state8_profile_fingerprint(profile: State8FeatureProfile) -> str:
    return _canonical_digest(profile.model_dump(mode="json"))


def require_state8_profile(
    profile: State8FeatureProfile,
    expected_profile_id: State8ProfileId | str,
) -> None:
    expected = state8_profile(expected_profile_id)
    if profile != expected:
        raise ValueError(
            "state8 feature profile is incompatible with the requested model family"
        )


@dataclass(frozen=True)
class _FrameIndex:
    by_index: dict[int, ObservationFrame]
    duplicate_indices: frozenset[int]
    misaligned_indices: frozenset[int]


@dataclass(frozen=True)
class _WindowNodeData:
    north_nt: NDArray[np.float64] | None
    temperature_k: NDArray[np.float64] | None
    received_count: int
    usable_count: int
    flags: tuple[str, ...]


def _validate_observation_stream(frames: Sequence[ObservationFrame]) -> None:
    if not frames:
        raise ValueError("at least one ObservationFrame is required")
    session_ids = {frame.session_id for frame in frames}
    if len(session_ids) != 1:
        raise ValueError("state8 extraction cannot mix observation sessions")
    if any(
        right.sim_time_s <= left.sim_time_s
        for left, right in zip(frames, frames[1:])
    ):
        raise ValueError("ObservationFrames must be supplied in causal time order")


def _index_frames(
    frames: Sequence[ObservationFrame],
    *,
    origin_s: float,
) -> _FrameIndex:
    indexed: dict[int, ObservationFrame] = {}
    duplicates: set[int] = set()
    misaligned: set[int] = set()
    for frame in frames:
        raw_index = (frame.sim_time_s - origin_s) * STATE8_SAMPLING_RATE_HZ
        index = int(round(raw_index))
        expected_time_s = origin_s + index / STATE8_SAMPLING_RATE_HZ
        if abs(frame.sim_time_s - expected_time_s) > TIME_ALIGNMENT_TOLERANCE_S:
            misaligned.add(index)
        if index in indexed:
            duplicates.add(index)
        else:
            indexed[index] = frame
    return _FrameIndex(
        by_index=indexed,
        duplicate_indices=frozenset(duplicates),
        misaligned_indices=frozenset(misaligned),
    )


def _reading_map(frame: ObservationFrame) -> tuple[dict[str, SensorReading], set[str]]:
    readings: dict[str, SensorReading] = {}
    duplicates: set[str] = set()
    for reading in frame.readings:
        if reading.sensor_id in readings:
            duplicates.add(reading.sensor_id)
        else:
            readings[reading.sensor_id] = reading
    return readings, duplicates


def _world_north_nt(reading: SensorReading) -> float:
    if reading.components_T is None:
        raise ValueError("vector components are unavailable")
    sensor_field = np.asarray(reading.components_T, dtype=np.float64)
    rotation_world_to_sensor = quaternion_world_to_sensor_matrix(
        reading.orientation_world_to_sensor_wxyz
    )
    world_field = rotation_world_to_sensor.T @ sensor_field
    return float(world_field[0] * TESLA_TO_NANOTESLA)


def _reading_flags(
    reading: SensorReading | None,
    *,
    calibration_version: str | None,
) -> tuple[bool, bool, tuple[str, ...]]:
    """Return received, usable and observable-only failure flags."""

    if reading is None:
        return False, False, ("missing_reading",)
    received = reading.components_T is not None
    flags: set[str] = set()
    if reading.measurement_mode is not MeasurementMode.VECTOR:
        flags.add("unsupported_measurement_mode")
    if reading.components_T is None:
        flags.add("missing_vector_components")
    if not reading.valid:
        flags.add("reading_marked_invalid")
    if reading.saturation_mask is None or any(reading.saturation_mask):
        flags.add("clipped")
    quality_flag_names = {flag.value for flag in reading.quality_flags}
    if QualityFlag.CLIPPED.value in quality_flag_names:
        flags.add("clipped")
    if QualityFlag.STUCK.value in quality_flag_names:
        flags.add("stuck")
    if QualityFlag.CLOCK_ERROR.value in quality_flag_names:
        flags.add("clock_error")
    if QualityFlag.SIGNAL_ABSENT.value in quality_flag_names:
        flags.add("signal_absent")
    if quality_flag_names - {
        QualityFlag.CLIPPED.value,
        QualityFlag.STUCK.value,
        QualityFlag.CLOCK_ERROR.value,
        QualityFlag.SIGNAL_ABSENT.value,
    }:
        flags.add("unsupported_quality_flag")
    if not reading.calibration_version:
        flags.add("calibration_not_declared")
    if (
        calibration_version is not None
        and reading.calibration_version != calibration_version
    ):
        flags.add("calibration_version_changed")
    return received, not flags, tuple(sorted(flags))


def _clock_is_uniform(
    acquisition_times: Sequence[datetime], expected_count: int
) -> bool:
    expected_period_s = 1.0 / STATE8_SAMPLING_RATE_HZ
    return len(acquisition_times) == expected_count and all(
        abs((right - left).total_seconds() - expected_period_s)
        <= TIME_ALIGNMENT_TOLERANCE_S
        for left, right in zip(acquisition_times, acquisition_times[1:])
    )


def build_state8_reference(
    frames: Sequence[ObservationFrame],
) -> State8ObservedReference:
    """Freeze the first observed eight seconds without accepting generator truth."""

    observations = tuple(frames)
    _validate_observation_stream(observations)
    origin_s = observations[0].sim_time_s
    indexed = _index_frames(observations, origin_s=origin_s)
    reference_frames = {
        index: frame
        for index, frame in indexed.by_index.items()
        if 0 <= index < STATE8_REFERENCE_SAMPLES
    }
    node_ids = tuple(
        sorted(
            {
                reading.sensor_id
                for frame in reference_frames.values()
                for reading in frame.readings
            }
        )
    )
    if not node_ids:
        raise ValueError("the observed reference contains no sensor nodes")

    node_references: list[State8NodeReference] = []
    for sensor_id in node_ids:
        flags: set[str] = set()
        north_values: list[float] = []
        temperatures: list[float] = []
        sequence_ids: list[int] = []
        acquisition_times: list[datetime] = []
        calibration_versions: set[str] = set()
        usable_count = 0
        for index in range(STATE8_REFERENCE_SAMPLES):
            frame = reference_frames.get(index)
            if frame is None:
                flags.add("missing_reference_frame")
                continue
            if index in indexed.duplicate_indices or index in indexed.misaligned_indices:
                flags.add("nonuniform_reference_sampling")
            readings, duplicate_sensor_ids = _reading_map(frame)
            if sensor_id in duplicate_sensor_ids:
                flags.add("duplicate_sensor_id")
            reading = readings.get(sensor_id)
            if reading is not None:
                calibration_versions.add(reading.calibration_version)
                sequence_ids.append(reading.sequence_id)
                acquisition_times.append(reading.acquisition_time)
            _, usable, reading_failures = _reading_flags(
                reading,
                calibration_version=None,
            )
            flags.update(reading_failures)
            if not usable or reading is None:
                continue
            north_values.append(_world_north_nt(reading))
            temperatures.append(float(reading.observed_temperature_K))
            usable_count += 1

        if len(calibration_versions) != 1:
            flags.add("reference_calibration_not_stable")
        if len(sequence_ids) != STATE8_REFERENCE_SAMPLES or any(
            right != left + 1 for left, right in zip(sequence_ids, sequence_ids[1:])
        ):
            flags.add("nonuniform_reference_sequence")
        if not _clock_is_uniform(acquisition_times, STATE8_REFERENCE_SAMPLES):
            flags.add("nonuniform_reference_clock")
        valid = usable_count == STATE8_REFERENCE_SAMPLES and not flags
        calibration_version = (
            next(iter(calibration_versions)) if len(calibration_versions) == 1 else None
        )
        node_references.append(
            State8NodeReference(
                sensor_id=sensor_id,
                calibration_version=calibration_version if valid else None,
                usable_sample_count=usable_count,
                north_reference_nt=(float(np.mean(north_values)) if valid else None),
                temperature_reference_k=(
                    float(np.mean(temperatures)) if valid else None
                ),
                valid=valid,
                flags=tuple(sorted(flags)),
            )
        )

    source_frame_ids = tuple(
        reference_frames[index].frame_id for index in sorted(reference_frames)
    )
    payload = {
        "schema_version": "aqse.state8-observed-reference.v1",
        "session_id": observations[0].session_id,
        "reference_start_time_s": origin_s,
        "reference_end_exclusive_s": origin_s + STATE8_REFERENCE_DURATION_S,
        "sampling_rate_hz": STATE8_SAMPLING_RATE_HZ,
        "source_frame_ids": source_frame_ids,
        "node_order": node_ids,
        "nodes": [node.model_dump(mode="json") for node in node_references],
    }
    content_digest = _canonical_digest(payload)
    return State8ObservedReference(
        **payload,
        reference_id=f"aqse-state8-reference-{content_digest[:16]}",
        content_digest=content_digest,
    )


def validate_state8_reference(reference: State8ObservedReference) -> None:
    payload = reference.model_dump(
        mode="json",
        exclude={"reference_id", "content_digest"},
    )
    expected_digest = _canonical_digest(payload)
    if reference.content_digest != expected_digest:
        raise ValueError("state8 observed-reference content digest is invalid")
    if reference.reference_id != f"aqse-state8-reference-{expected_digest[:16]}":
        raise ValueError("state8 observed-reference identity is invalid")
    if reference.reference_end_exclusive_s != (
        reference.reference_start_time_s + STATE8_REFERENCE_DURATION_S
    ):
        raise ValueError("state8 observed-reference duration is invalid")


def _window_node_data(
    *,
    indexed: _FrameIndex,
    sensor_id: str,
    reference: State8NodeReference,
    start_index: int,
) -> _WindowNodeData:
    flags: set[str] = set()
    north_values: list[float] = []
    temperatures: list[float] = []
    sequence_ids: list[int] = []
    acquisition_times: list[datetime] = []
    received_count = 0
    usable_count = 0
    if not reference.valid:
        flags.add("reference_invalid")
    for index in range(start_index, start_index + STATE8_WINDOW_SAMPLES):
        frame = indexed.by_index.get(index)
        if frame is None:
            flags.add("missing_frame")
            continue
        if index in indexed.duplicate_indices or index in indexed.misaligned_indices:
            flags.add("nonuniform_sampling")
        readings, duplicate_sensor_ids = _reading_map(frame)
        if sensor_id in duplicate_sensor_ids:
            flags.add("duplicate_sensor_id")
        reading = readings.get(sensor_id)
        received, usable, reading_failures = _reading_flags(
            reading,
            calibration_version=reference.calibration_version,
        )
        received_count += int(received)
        flags.update(reading_failures)
        if reading is not None:
            sequence_ids.append(reading.sequence_id)
            acquisition_times.append(reading.acquisition_time)
        if not usable or reading is None:
            continue
        north_values.append(_world_north_nt(reading))
        temperatures.append(float(reading.observed_temperature_K))
        usable_count += 1
    if len(sequence_ids) != STATE8_WINDOW_SAMPLES or any(
        right != left + 1 for left, right in zip(sequence_ids, sequence_ids[1:])
    ):
        flags.add("nonuniform_sequence")
    if not _clock_is_uniform(acquisition_times, STATE8_WINDOW_SAMPLES):
        flags.add("nonuniform_clock")
    complete = (
        reference.valid
        and usable_count == STATE8_WINDOW_SAMPLES
        and not flags
    )
    return _WindowNodeData(
        north_nt=(np.asarray(north_values, dtype=np.float64) if complete else None),
        temperature_k=(
            np.asarray(temperatures, dtype=np.float64) if complete else None
        ),
        received_count=received_count,
        usable_count=usable_count,
        flags=tuple(sorted(flags)),
    )


def _linear_slope(values: NDArray[np.float64]) -> float:
    time_s = np.arange(len(values), dtype=np.float64) / STATE8_SAMPLING_RATE_HZ
    centered_time = time_s - float(np.mean(time_s))
    centered_values = values - float(np.mean(values))
    denominator = float(centered_time @ centered_time)
    return float((centered_time @ centered_values) / denominator)


def _band_power_ratio_db(values: NDArray[np.float64]) -> float:
    time_s = np.arange(len(values), dtype=np.float64) / STATE8_SAMPLING_RATE_HZ
    slope = _linear_slope(values)
    intercept = float(np.mean(values) - slope * np.mean(time_s))
    detrended = values - (intercept + slope * time_s)
    hann = np.hanning(len(values))
    windowed = detrended * hann
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / STATE8_SAMPLING_RATE_HZ)
    density = np.abs(np.fft.rfft(windowed)) ** 2 / (
        STATE8_SAMPLING_RATE_HZ * float(np.sum(hann * hann))
    )
    if len(density) > 2:
        density[1:-1] *= 2.0
    frequency_step_hz = STATE8_SAMPLING_RATE_HZ / len(values)
    lower_power = float(
        np.sum(density[(frequencies >= 0.25) & (frequencies < 2.0)])
        * frequency_step_hz
    )
    upper_power = float(
        np.sum(density[(frequencies >= 2.0) & (frequencies <= 20.0)])
        * frequency_step_hz
    )
    return float(
        10.0
        * log10(
            max(lower_power, SPECTRAL_POWER_FLOOR)
            / max(upper_power, SPECTRAL_POWER_FLOOR)
        )
    )


def _pearson(left: NDArray[np.float64], right: NDArray[np.float64]) -> float | None:
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = sqrt(
        float(left_centered @ left_centered) * float(right_centered @ right_centered)
    )
    if denominator <= np.finfo(np.float64).eps:
        return None
    value = float((left_centered @ right_centered) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _common_features(
    data: _WindowNodeData,
    reference: State8NodeReference,
) -> tuple[tuple[float, float, float, float, float] | None, NDArray[np.float64] | None]:
    if (
        data.north_nt is None
        or data.temperature_k is None
        or reference.north_reference_nt is None
        or reference.temperature_reference_k is None
    ):
        return None, None
    anomaly = data.north_nt - reference.north_reference_nt
    median = float(np.median(anomaly))
    common = (
        float(np.mean(anomaly)),
        float(1.4826 * np.median(np.abs(anomaly - median))),
        _linear_slope(anomaly),
        _band_power_ratio_db(anomaly),
        float(np.mean(data.temperature_k) - reference.temperature_reference_k),
    )
    return common, anomaly


def _record(
    *,
    profile: State8FeatureProfile,
    profile_fingerprint: str,
    reference_bundle: State8ObservedReference,
    sensor_id: str,
    start_index: int,
    frame_index: _FrameIndex,
    all_node_data: dict[str, _WindowNodeData],
) -> State8FeatureRecord:
    node_reference = next(
        node for node in reference_bundle.nodes if node.sensor_id == sensor_id
    )
    focal = all_node_data[sensor_id]
    flags = set(focal.flags)
    common, focal_anomaly = _common_features(focal, node_reference)
    received_fraction = focal.received_count / STATE8_WINDOW_SAMPLES
    values: list[float | None] = [None] * 8
    if common is not None:
        values[:5] = common
    values[7] = float(received_fraction)
    peer_ids: tuple[str, ...] = ()

    if profile.profile_id == LOCAL_STATE8_PROFILE_ID:
        if focal_anomaly is not None:
            values[5] = float(sqrt(float(np.mean(np.diff(focal_anomaly) ** 2))))
            correlation = _pearson(focal_anomaly[:-1], focal_anomaly[1:])
            values[6] = correlation
            if correlation is None:
                flags.add("lag1_correlation_undefined")
    else:
        candidate_peer_ids = tuple(
            node for node in reference_bundle.node_order if node != sensor_id
        )
        if len(reference_bundle.node_order) < profile.minimum_network_nodes:
            flags.add("insufficient_network_context")
        elif focal_anomaly is not None:
            peer_anomalies: list[NDArray[np.float64]] = []
            valid_peer_ids: list[str] = []
            for peer_id in candidate_peer_ids:
                peer_reference = next(
                    node
                    for node in reference_bundle.nodes
                    if node.sensor_id == peer_id
                )
                _, peer_anomaly = _common_features(
                    all_node_data[peer_id],
                    peer_reference,
                )
                if peer_anomaly is None:
                    continue
                valid_peer_ids.append(peer_id)
                peer_anomalies.append(peer_anomaly)
            peer_ids = tuple(valid_peer_ids)
            required_peers = profile.minimum_network_nodes - 1
            if len(peer_anomalies) < required_peers:
                flags.add("peer_context_invalid")
                flags.add("insufficient_valid_peer_context")
            else:
                peer_matrix = np.vstack(peer_anomalies)
                peer_median = np.median(peer_matrix, axis=0)
                residual = focal_anomaly - peer_median
                values[5] = float(sqrt(float(np.mean(residual * residual))))
                correlations = [
                    _pearson(focal_anomaly, peer_anomaly)
                    for peer_anomaly in peer_anomalies
                ]
                if any(value is None for value in correlations):
                    flags.add("peer_correlation_undefined")
                else:
                    values[6] = float(
                        np.mean([cast(float, value) for value in correlations])
                    )

    available = cast(
        tuple[bool, bool, bool, bool, bool, bool, bool, bool],
        tuple(value is not None for value in values),
    )
    valid_for_quantum = all(available)
    if not valid_for_quantum and not flags:
        flags.add("feature_unavailable")
    source_frame_ids = tuple(
        frame_index.by_index[index].frame_id
        for index in range(start_index, start_index + STATE8_WINDOW_SAMPLES)
        if index in frame_index.by_index
    )
    start_time_s = (
        reference_bundle.reference_start_time_s
        + start_index / STATE8_SAMPLING_RATE_HZ
    )
    typed_values = cast(State8Values, tuple(values))
    return State8FeatureRecord(
        window_id=(
            f"{reference_bundle.session_id}:{profile.profile_id}:{sensor_id}:"
            f"{start_index}:{start_index + STATE8_WINDOW_SAMPLES}"
        ),
        session_id=reference_bundle.session_id,
        sensor_id=sensor_id,
        profile_id=profile.profile_id,
        profile_fingerprint=profile_fingerprint,
        reference_id=reference_bundle.reference_id,
        start_time_s=start_time_s,
        end_exclusive_time_s=start_time_s + STATE8_WINDOW_DURATION_S,
        source_frame_ids=source_frame_ids,
        peer_sensor_ids=peer_ids,
        values=typed_values,
        quality=State8FeatureQuality(
            valid_for_quantum=valid_for_quantum,
            flags=tuple(sorted(flags)) if not valid_for_quantum else (),
            per_feature_valid=available,
            received_sample_count=focal.received_count,
            usable_sample_count=focal.usable_count,
        ),
    )


def extract_state8_features(
    frames: Sequence[ObservationFrame],
    *,
    reference: State8ObservedReference,
    profile_id: State8ProfileId | str,
    sensor_ids: Sequence[str] | None = None,
) -> State8ExtractionResponse:
    """Extract causal state8 windows using observations and a frozen reference."""

    observations = tuple(frames)
    _validate_observation_stream(observations)
    validate_state8_reference(reference)
    if observations[0].session_id != reference.session_id:
        raise ValueError("observation session differs from the frozen reference")
    profile = state8_profile(profile_id)
    fingerprint = state8_profile_fingerprint(profile)
    if sensor_ids is not None and len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("selected state8 sensor IDs must be unique")
    selected_nodes = (
        reference.node_order if sensor_ids is None else tuple(sorted(sensor_ids))
    )
    if not selected_nodes:
        raise ValueError("at least one state8 sensor must be selected")
    unknown = set(selected_nodes) - set(reference.node_order)
    if unknown:
        raise ValueError(f"unknown state8 sensor IDs: {', '.join(sorted(unknown))}")

    indexed = _index_frames(
        observations,
        origin_s=reference.reference_start_time_s,
    )
    maximum_index = max(indexed.by_index, default=-1)
    latest_start = maximum_index - STATE8_WINDOW_SAMPLES + 1
    return _state8_response_for_starts(
        reference=reference,
        profile=profile,
        profile_fingerprint=fingerprint,
        selected_nodes=selected_nodes,
        indexed=indexed,
        start_indices=range(
            STATE8_REFERENCE_SAMPLES,
            latest_start + 1,
            STATE8_HOP_SAMPLES,
        ),
    )


def extract_latest_state8_window(
    frames: Sequence[ObservationFrame],
    *,
    reference: State8ObservedReference,
    profile_id: State8ProfileId | str,
    sensor_ids: Sequence[str] | None = None,
) -> State8ExtractionResponse:
    """Extract only the newest complete, hop-aligned causal state8 window."""

    observations = tuple(frames)
    _validate_observation_stream(observations)
    validate_state8_reference(reference)
    if observations[0].session_id != reference.session_id:
        raise ValueError("observation session differs from the frozen reference")
    profile = state8_profile(profile_id)
    fingerprint = state8_profile_fingerprint(profile)
    if sensor_ids is not None and len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("selected state8 sensor IDs must be unique")
    selected_nodes = (
        reference.node_order if sensor_ids is None else tuple(sorted(sensor_ids))
    )
    if not selected_nodes:
        raise ValueError("at least one state8 sensor must be selected")
    unknown = set(selected_nodes) - set(reference.node_order)
    if unknown:
        raise ValueError(f"unknown state8 sensor IDs: {', '.join(sorted(unknown))}")

    indexed = _index_frames(
        observations,
        origin_s=reference.reference_start_time_s,
    )
    maximum_index = max(indexed.by_index, default=-1)
    latest_possible_start = maximum_index - STATE8_WINDOW_SAMPLES + 1
    if latest_possible_start < STATE8_REFERENCE_SAMPLES:
        start_indices: tuple[int, ...] = ()
    else:
        hop_count = (
            latest_possible_start - STATE8_REFERENCE_SAMPLES
        ) // STATE8_HOP_SAMPLES
        start_indices = (
            STATE8_REFERENCE_SAMPLES + hop_count * STATE8_HOP_SAMPLES,
        )
    return _state8_response_for_starts(
        reference=reference,
        profile=profile,
        profile_fingerprint=fingerprint,
        selected_nodes=selected_nodes,
        indexed=indexed,
        start_indices=start_indices,
    )


def _state8_response_for_starts(
    *,
    reference: State8ObservedReference,
    profile: State8FeatureProfile,
    profile_fingerprint: str,
    selected_nodes: tuple[str, ...],
    indexed: _FrameIndex,
    start_indices: Sequence[int] | range,
) -> State8ExtractionResponse:
    records: list[State8FeatureRecord] = []
    reference_by_node = {node.sensor_id: node for node in reference.nodes}
    for start_index in start_indices:
        all_node_data = {
            sensor_id: _window_node_data(
                indexed=indexed,
                sensor_id=sensor_id,
                reference=reference_by_node[sensor_id],
                start_index=start_index,
            )
            for sensor_id in reference.node_order
        }
        records.extend(
            _record(
                profile=profile,
                profile_fingerprint=profile_fingerprint,
                reference_bundle=reference,
                sensor_id=sensor_id,
                start_index=start_index,
                frame_index=indexed,
                all_node_data=all_node_data,
            )
            for sensor_id in selected_nodes
        )
    return State8ExtractionResponse(
        profile=profile,
        profile_fingerprint=profile_fingerprint,
        reference=reference,
        node_order=reference.node_order,
        records=tuple(records),
    )
