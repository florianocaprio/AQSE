from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import numpy as np

from app.features.state8 import build_state8_reference, extract_state8_features
from app.features.state8_models import NETWORK_STATE8_PROFILE_ID
from app.network.defaults import default_network_configuration
from app.network.models import (
    DipoleSourceConfiguration,
    EnvironmentConfiguration,
    EventKind,
    NetworkEventConfiguration,
    NetworkSessionConfiguration,
    NodeErrorConfiguration,
    TemperatureDriverConfiguration,
    TemperatureDriverKind,
)
from app.network.simulation import NetworkSimulator
from app.training.canonical import canonical_json_bytes

from .models import (
    FourSensorStudyBuild,
    FourSensorStudyEpisodePlan,
    FourSensorStudyLabel,
    FourSensorStudyObservation,
    StudyPartition,
)
from .protocol import (
    FROZEN_FOUR_SENSOR_STUDY_PROTOCOL,
    GEOMETRY_IDS,
    SENSOR_IDS,
    FourSensorStudyProtocol,
    GeometryId,
    ScenarioName,
)


def _short_identity(prefix: str, value: object) -> str:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return f"{prefix}-{digest[:16]}"


def _domain_entropy(domain: str) -> tuple[int, int, int, int]:
    digest = hashlib.sha256(domain.encode("utf-8")).digest()
    return (
        int.from_bytes(digest[0:4], byteorder="big"),
        int.from_bytes(digest[4:8], byteorder="big"),
        int.from_bytes(digest[8:12], byteorder="big"),
        int.from_bytes(digest[12:16], byteorder="big"),
    )


def _episode_seed(
    *,
    base_seed: int,
    generation_domain: str,
    scenario_index: int,
    replicate: int,
) -> int:
    sequence = np.random.SeedSequence(
        [
            base_seed,
            *_domain_entropy(generation_domain),
            scenario_index,
            replicate,
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _geometry_id(scenario_index: int, replicate: int) -> GeometryId:
    return GEOMETRY_IDS[(scenario_index + replicate) % len(GEOMETRY_IDS)]


def _focal_sensor_id(
    scenario_index: int,
    replicate: int,
    *,
    partition: StudyPartition,
) -> str:
    if partition == "pilot":
        index = (scenario_index + 2 * replicate) % len(SENSOR_IDS)
    else:
        # The added four-replicate block term prevents focal identity from
        # becoming a class proxy while retaining exact global balance.
        index = (
            scenario_index + 2 * replicate + replicate // 4
        ) % len(SENSOR_IDS)
    return SENSOR_IDS[index]


def _spatial_dipole(
    *,
    random: np.random.Generator,
    perturbation_nt: float,
) -> DipoleSourceConfiguration:
    domain = FROZEN_FOUR_SENSOR_STUDY_PROTOCOL.simulation
    direction = random.normal(size=3)
    direction /= np.linalg.norm(direction)
    moment_amplitude = perturbation_nt * 0.01
    return DipoleSourceConfiguration(
        source_id="moving-spatial-source",
        initial_position_m=(
            float(random.uniform(*domain.mobile_source_initial_x_m_range)),
            float(random.uniform(*domain.mobile_source_y_m_range)),
            float(random.uniform(*domain.mobile_source_height_m_range)),
        ),
        velocity_m_per_s=(
            float(random.uniform(*domain.mobile_source_velocity_x_m_per_s_range)),
            0.0,
            0.0,
        ),
        moment_A_m2=tuple(
            float(component * moment_amplitude) for component in direction
        ),
        minimum_distance_m=0.20,
        active_start_time_s=domain.event_start_s,
        active_duration_s=domain.event_duration_s,
    )


def _study_configuration(
    *,
    scenario: ScenarioName,
    geometry_id: GeometryId,
    replicate: int,
    episode_seed: int,
    focal_sensor_id: str,
    generation_domain: str,
) -> tuple[NetworkSessionConfiguration, float, float]:
    protocol = FROZEN_FOUR_SENSOR_STUDY_PROTOCOL
    domain = protocol.simulation
    random = np.random.default_rng(
        np.random.SeedSequence([episode_seed, *_domain_entropy(generation_domain)])
    )
    base = default_network_configuration(4)
    base_by_id = {node.sensor_id: node for node in base.nodes}
    positions = dict(
        zip(SENSOR_IDS, protocol.geometry(geometry_id).positions_m, strict=True)
    )
    perturbation_nt = float(random.uniform(*domain.perturbation_north_nt_range))
    focal_extra_nt = float(random.uniform(*domain.focal_extra_nt_range))
    device_mode = protocol.device_modes[replicate % len(protocol.device_modes)]

    nodes = []
    for sensor_id in SENSOR_IDS:
        node = base_by_id[sensor_id]
        temperature_offset = float(
            random.uniform(*domain.base_temperature_offset_k_range)
        )
        errors = NodeErrorConfiguration(
            white_noise_std_T_per_sample=(
                float(random.uniform(*domain.base_white_noise_nt_range)) * 1.0e-9,
            )
            * 3,
            ou_sigma_T=(
                float(random.uniform(*domain.base_ou_noise_nt_range)) * 1.0e-9,
            )
            * 3,
            ou_tau_s=2.0,
            initial_temperature_K=293.15 + temperature_offset,
            ambient_temperature_K=293.15 + temperature_offset,
            bandwidth_Hz=None,
        )
        if (
            scenario == "DEVICE_COMPATIBLE"
            and sensor_id == focal_sensor_id
            and device_mode == "thermal"
        ):
            errors = errors.model_copy(
                update={
                    "temperature_driver": TemperatureDriverConfiguration(
                        kind=TemperatureDriverKind.RAMP,
                        start_time_s=domain.event_start_s,
                        ramp_rate_K_per_s=float(
                            random.uniform(
                                *domain.focal_temperature_ramp_k_per_s_range
                            )
                        ),
                        ramp_duration_s=domain.event_duration_s,
                    ),
                    "thermal_bias_T_per_K": (
                        float(
                            random.uniform(
                                *domain.focal_thermal_bias_nt_per_k_range
                            )
                        )
                        * 1.0e-9,
                        0.0,
                        0.0,
                    ),
                }
            )
        nodes.append(
            node.model_copy(
                update={
                    "position_m": positions[sensor_id],
                    "errors": errors,
                }
            )
        )

    # Configuration order is nuisance-only. State8 restores sorted sensor IDs.
    node_order = random.permutation(len(nodes)).tolist()
    nodes = [nodes[int(index)] for index in node_order]
    environment = EnvironmentConfiguration(
        uniform_field_T=(20.0e-6, 0.0, 45.0e-6),
        common_ou_sigma_T=(0.10e-9, 0.05e-9, 0.05e-9),
        common_ou_tau_s=3.0,
    )
    events: list[NetworkEventConfiguration] = []

    if scenario == "NORMAL":
        events.append(
            NetworkEventConfiguration(
                event_id="normal-sham-event",
                kind=EventKind.WORLD_FIELD_OFFSET,
                start_time_s=domain.event_start_s,
                duration_s=domain.event_duration_s,
            )
        )
    elif scenario == "ENVIRONMENT_COMPATIBLE":
        environment = environment.model_copy(
            update={
                "dipoles": (
                    _spatial_dipole(
                        random=random,
                        perturbation_nt=perturbation_nt,
                    ),
                )
            }
        )
    elif scenario == "DEVICE_COMPATIBLE":
        if device_mode == "bias":
            events.append(
                NetworkEventConfiguration(
                    event_id="focal-device-bias",
                    kind=EventKind.NODE_BIAS,
                    start_time_s=domain.event_start_s,
                    duration_s=domain.event_duration_s,
                    target_sensor_ids=(focal_sensor_id,),
                    field_offset_T=(perturbation_nt * 1.0e-9, 0.0, 0.0),
                )
            )
        elif device_mode == "drift":
            events.append(
                NetworkEventConfiguration(
                    event_id="focal-device-drift",
                    kind=EventKind.NODE_DRIFT,
                    start_time_s=domain.event_start_s,
                    duration_s=domain.event_duration_s,
                    target_sensor_ids=(focal_sensor_id,),
                    drift_rate_T_per_s=(
                        perturbation_nt / domain.event_duration_s * 1.0e-9,
                        0.0,
                        0.0,
                    ),
                )
            )
        elif device_mode == "noise":
            events.append(
                NetworkEventConfiguration(
                    event_id="focal-device-noise",
                    kind=EventKind.NODE_NOISE_BURST,
                    start_time_s=domain.event_start_s,
                    duration_s=domain.event_duration_s,
                    target_sensor_ids=(focal_sensor_id,),
                    noise_multiplier=float(
                        random.uniform(*domain.focal_noise_multiplier_range)
                    ),
                )
            )
    elif scenario == "MIXED_OR_AMBIGUOUS":
        environment = environment.model_copy(
            update={
                "dipoles": (
                    _spatial_dipole(
                        random=random,
                        perturbation_nt=perturbation_nt,
                    ),
                )
            }
        )
        events.append(
            NetworkEventConfiguration(
                event_id="mixed-focal-device-bias",
                kind=EventKind.NODE_BIAS,
                start_time_s=domain.event_start_s,
                duration_s=domain.event_duration_s,
                target_sensor_ids=(focal_sensor_id,),
                field_offset_T=(focal_extra_nt * 1.0e-9, 0.0, 0.0),
            )
        )

    configuration = NetworkSessionConfiguration(
        session_name="AQSE four-sensor frozen study",
        random_seed=episode_seed,
        sampling_rate_Hz=domain.sampling_rate_hz,
        ui_refresh_rate_Hz=5.0,
        time_scale=1.0,
        buffer_duration_s=domain.episode_duration_s,
        environment=environment,
        nodes=tuple(nodes),
        events=tuple(events),
    )
    return configuration, perturbation_nt, focal_extra_nt


def _build_plan(
    *,
    protocol: FourSensorStudyProtocol,
    partition: StudyPartition,
    scenario_index: int,
    scenario: ScenarioName,
    replicate: int,
    protocol_freeze_digest: str | None,
) -> FourSensorStudyEpisodePlan:
    if partition == "pilot":
        base_seed = protocol.seeds.pilot_generation
        generation_domain = protocol.pilot_generation_domain_separator
    else:
        base_seed = protocol.seeds.canonical_generation
        generation_domain = protocol.canonical_generation_domain_separator
    geometry_id = _geometry_id(scenario_index, replicate)
    focal_sensor_id = _focal_sensor_id(
        scenario_index,
        replicate,
        partition=partition,
    )
    episode_seed = _episode_seed(
        base_seed=base_seed,
        generation_domain=generation_domain,
        scenario_index=scenario_index,
        replicate=replicate,
    )
    configuration, perturbation_nt, focal_extra_nt = _study_configuration(
        scenario=scenario,
        geometry_id=geometry_id,
        replicate=replicate,
        episode_seed=episode_seed,
        focal_sensor_id=focal_sensor_id,
        generation_domain=generation_domain,
    )
    lineage_payload = {
        "study_id": protocol.study_id,
        "generation_domain": generation_domain,
        "scenario_index": scenario_index,
        "geometry_id": geometry_id,
        "replicate": replicate,
        "generation_seed": episode_seed,
    }
    episode_payload = {
        **lineage_payload,
        "partition": partition,
        "protocol_digest": protocol.digest,
        "protocol_freeze_digest": protocol_freeze_digest,
    }
    return FourSensorStudyEpisodePlan(
        episode_id=_short_identity("four-sensor-episode", episode_payload),
        generative_lineage_id=_short_identity(
            "four-sensor-lineage",
            lineage_payload,
        ),
        generation_domain=generation_domain,
        partition=partition,
        scenario=scenario,
        geometry_id=geometry_id,
        replicate_index=replicate,
        episode_seed=episode_seed,
        focal_sensor_id=focal_sensor_id,
        perturbation_north_nt=perturbation_nt,
        focal_extra_nt=focal_extra_nt,
        protocol_freeze_digest=protocol_freeze_digest,
        network_configuration=configuration.model_dump(mode="json"),
    )


def build_pilot_plans(
    protocol: FourSensorStudyProtocol = FROZEN_FOUR_SENSOR_STUDY_PROTOCOL,
) -> tuple[FourSensorStudyEpisodePlan, ...]:
    if protocol != FROZEN_FOUR_SENSOR_STUDY_PROTOCOL:
        raise ValueError("only the declared four-sensor study protocol is supported")
    return tuple(
        _build_plan(
            protocol=protocol,
            partition="pilot",
            scenario_index=scenario_index,
            scenario=scenario,
            replicate=replicate,
            protocol_freeze_digest=None,
        )
        for scenario_index, scenario in enumerate(protocol.scenarios)
        for replicate in range(protocol.pilot.episodes_per_scenario)
    )


def build_canonical_plans(
    *,
    protocol_freeze_digest: str,
    protocol: FourSensorStudyProtocol = FROZEN_FOUR_SENSOR_STUDY_PROTOCOL,
) -> tuple[FourSensorStudyEpisodePlan, ...]:
    """Build the 100 final TEST plans after the protocol freeze is published."""

    if protocol != FROZEN_FOUR_SENSOR_STUDY_PROTOCOL:
        raise ValueError("only the frozen four-sensor study protocol is supported")
    if re.fullmatch(r"[a-f0-9]{64}", protocol_freeze_digest) is None:
        raise ValueError("a published 64-character protocol freeze digest is required")
    return tuple(
        _build_plan(
            protocol=protocol,
            partition="test",
            scenario_index=scenario_index,
            scenario=scenario,
            replicate=replicate,
            protocol_freeze_digest=protocol_freeze_digest,
        )
        for scenario_index, scenario in enumerate(protocol.scenarios)
        for replicate in range(protocol.canonical.episodes_per_scenario)
    )


def generate_episode(
    plan: FourSensorStudyEpisodePlan,
) -> tuple[FourSensorStudyObservation, FourSensorStudyLabel]:
    configuration = NetworkSessionConfiguration.model_validate(
        plan.network_configuration
    )
    simulator = NetworkSimulator(configuration)
    domain = FROZEN_FOUR_SENSOR_STUDY_PROTOCOL.simulation
    sample_count = int(domain.episode_duration_s * domain.sampling_rate_hz)
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frames = []
    for index in range(sample_count):
        buffered = simulator.generate(
            session_id=plan.episode_id,
            frame_id=index + 1,
            sim_time_s=index / domain.sampling_rate_hz,
            epoch_utc=epoch,
            events=configuration.events,
        )
        # Generator truth is discarded at this boundary. Only observations
        # reach State8 and the predictor channel.
        frames.append(buffered.observation)

    reference_samples = int(domain.reference_end_s * domain.sampling_rate_hz)
    reference = build_state8_reference(frames[:reference_samples])
    extraction = extract_state8_features(
        frames,
        reference=reference,
        profile_id=NETWORK_STATE8_PROFILE_ID,
        sensor_ids=(plan.focal_sensor_id,),
    )
    primary = tuple(
        record
        for record in extraction.records
        if np.isclose(
            record.start_time_s,
            domain.supervised_window_start_s,
            rtol=0.0,
            atol=1.0e-12,
        )
        and np.isclose(
            record.end_exclusive_time_s,
            domain.supervised_window_end_s,
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    if len(primary) != 1:
        raise RuntimeError("State8 extraction did not yield one supervised window")
    observation = FourSensorStudyObservation(
        episode_id=plan.episode_id,
        generative_lineage_id=plan.generative_lineage_id,
        partition=plan.partition,
        network_feature=primary[0],
        network_replay_features=extraction.records,
    )
    label = FourSensorStudyLabel(
        episode_id=plan.episode_id,
        generative_lineage_id=plan.generative_lineage_id,
        partition=plan.partition,
        scenario=plan.scenario,
        geometry_id=plan.geometry_id,
        focal_sensor_id=plan.focal_sensor_id,
    )
    return observation, label


def build_pilot(
    protocol: FourSensorStudyProtocol = FROZEN_FOUR_SENSOR_STUDY_PROTOCOL,
) -> FourSensorStudyBuild:
    plans = build_pilot_plans(protocol)
    generated = tuple(generate_episode(plan) for plan in plans)
    return FourSensorStudyBuild(
        kind="pilot",
        protocol_digest=protocol.digest,
        episode_plans=plans,
        observations=tuple(item[0] for item in generated),
        labels=tuple(item[1] for item in generated),
    )


def build_canonical(
    *,
    protocol_freeze_digest: str,
    protocol: FourSensorStudyProtocol = FROZEN_FOUR_SENSOR_STUDY_PROTOCOL,
) -> FourSensorStudyBuild:
    """Generate the final 100 TEST episodes; this function performs no fitting."""

    plans = build_canonical_plans(
        protocol=protocol,
        protocol_freeze_digest=protocol_freeze_digest,
    )
    generated = tuple(generate_episode(plan) for plan in plans)
    return FourSensorStudyBuild(
        kind="final_test",
        protocol_digest=protocol.digest,
        protocol_freeze_digest=protocol_freeze_digest,
        episode_plans=plans,
        observations=tuple(item[0] for item in generated),
        labels=tuple(item[1] for item in generated),
    )
