from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np

from app.demo.protocol import (
    FROZEN_NETWORK_DEMO_PROTOCOL,
    SCENARIOS,
    NetworkDemoProtocol,
    ScenarioName,
)
from app.demo.study_models import (
    NetworkStudyBuild,
    NetworkStudyEpisodePlan,
    NetworkStudyLabel,
    NetworkStudyObservation,
    StudyPartition,
)
from app.features.state8 import build_state8_reference, extract_latest_state8_window
from app.features.state8_models import (
    LOCAL_STATE8_PROFILE_ID,
    NETWORK_STATE8_PROFILE_ID,
)
from app.network.defaults import default_network_configuration
from app.network.models import (
    DipoleSourceConfiguration,
    EnvironmentConfiguration,
    EventKind,
    NetworkEventConfiguration,
    NetworkSessionConfiguration,
    NodeErrorConfiguration,
    SensorNodeConfiguration,
    TemperatureDriverConfiguration,
    TemperatureDriverKind,
)
from app.network.simulation import NetworkSimulator
from app.training.canonical import canonical_json_bytes


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


def _episode_seed(scenario_index: int, node_count: int, replicate: int) -> int:
    protocol = FROZEN_NETWORK_DEMO_PROTOCOL
    sequence = np.random.SeedSequence(
        [
            protocol.seeds.generation,
            *_domain_entropy(protocol.generation_domain_separator),
            scenario_index,
            node_count,
            replicate,
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _split_assignments() -> dict[tuple[str, int, int], StudyPartition]:
    protocol = FROZEN_NETWORK_DEMO_PROTOCOL
    random = np.random.default_rng(
        np.random.SeedSequence(
            [
                protocol.seeds.split,
                *_domain_entropy(protocol.generation_domain_separator),
            ]
        )
    )
    assignments: dict[tuple[str, int, int], StudyPartition] = {}
    for scenario in SCENARIOS:
        for node_count in FROZEN_NETWORK_DEMO_PROTOCOL.node_counts:
            order = random.permutation(5).tolist()
            by_replicate: dict[int, StudyPartition] = {
                **{int(item): "train" for item in order[:3]},
                int(order[3]): "validation",
                int(order[4]): "test",
            }
            for replicate, partition in by_replicate.items():
                assignments[(scenario, node_count, replicate)] = partition
    return assignments


def _study_configuration(
    *,
    scenario: ScenarioName,
    node_count: int,
    replicate: int,
    episode_seed: int,
    generation_domain: str,
) -> tuple[NetworkSessionConfiguration, float, float, str]:
    protocol = FROZEN_NETWORK_DEMO_PROTOCOL
    domain = protocol.simulation
    random = np.random.default_rng(
        np.random.SeedSequence(
            [episode_seed, *_domain_entropy(generation_domain)]
        )
    )
    base = default_network_configuration(node_count)
    focal_sensor_id = f"S{replicate % node_count + 1}"
    perturbation_nt = float(random.uniform(*domain.perturbation_north_nt_range))
    focal_extra_nt = float(random.uniform(*domain.focal_extra_nt_range))
    device_mode = protocol.device_modes_by_replicate[replicate]
    nodes: list[SensorNodeConfiguration] = []
    for node in base.nodes:
        white_nt = float(random.uniform(*domain.base_white_noise_nt_range))
        ou_nt = float(random.uniform(*domain.base_ou_noise_nt_range))
        temperature_offset = float(
            random.uniform(*domain.base_temperature_offset_k_range)
        )
        errors = NodeErrorConfiguration(
            white_noise_std_T_per_sample=(white_nt * 1.0e-9,) * 3,
            ou_sigma_T=(ou_nt * 1.0e-9,) * 3,
            ou_tau_s=2.0,
            initial_temperature_K=293.15 + temperature_offset,
            ambient_temperature_K=293.15 + temperature_offset,
            bandwidth_Hz=None,
        )
        if (
            scenario == "DEVICE_COMPATIBLE"
            and node.sensor_id == focal_sensor_id
            and device_mode == "thermal"
        ):
            errors = errors.model_copy(
                update={
                    "temperature_driver": TemperatureDriverConfiguration(
                        kind=TemperatureDriverKind.RAMP,
                        ramp_rate_K_per_s=float(
                            random.uniform(
                                *domain.focal_temperature_ramp_k_per_s_range
                            )
                        ),
                        ramp_duration_s=domain.episode_duration_s,
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
        nodes.append(node.model_copy(update={"errors": errors}))

    # Ordering is nuisance metadata only and is randomized independently for
    # every episode. Feature extraction later restores deterministic sensor ID
    # ordering and predeclares the focal node before observing measurements.
    node_order = random.permutation(len(nodes)).tolist()
    nodes = [nodes[int(index)] for index in node_order]
    common_event = {
        "start_time_s": domain.event_start_s,
        "duration_s": domain.event_duration_s,
        "field_offset_T": (perturbation_nt * 1.0e-9, 0.0, 0.0),
    }
    events: list[NetworkEventConfiguration] = []
    environment = EnvironmentConfiguration(
        uniform_field_T=(20.0e-6, 0.0, 45.0e-6),
        common_ou_sigma_T=(0.10e-9, 0.05e-9, 0.05e-9),
        common_ou_tau_s=3.0,
    )
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
        direction = random.normal(size=3)
        direction /= np.linalg.norm(direction)
        moment_amplitude = perturbation_nt * 0.01
        environment = environment.model_copy(
            update={
                "dipoles": (
                    DipoleSourceConfiguration(
                        source_id="moving-spatial-source",
                        initial_position_m=(
                            float(
                                random.uniform(
                                    *domain.mobile_source_initial_x_m_range
                                )
                            ),
                            float(random.uniform(*domain.mobile_source_y_m_range)),
                            float(
                                random.uniform(*domain.mobile_source_height_m_range)
                            ),
                        ),
                        velocity_m_per_s=(
                            float(
                                random.uniform(
                                    *domain.mobile_source_velocity_x_m_per_s_range
                                )
                            ),
                            0.0,
                            0.0,
                        ),
                        moment_A_m2=(
                            float(direction[0] * moment_amplitude),
                            float(direction[1] * moment_amplitude),
                            float(direction[2] * moment_amplitude),
                        ),
                        minimum_distance_m=0.20,
                    ),
                )
            }
        )
        events.append(
            NetworkEventConfiguration(
                event_id="moving-source-analysis-window",
                kind=EventKind.WORLD_FIELD_OFFSET,
                start_time_s=domain.event_start_s,
                duration_s=domain.event_duration_s,
            )
        )
    elif scenario == "DEVICE_COMPATIBLE":
        if device_mode == "bias":
            events.append(
                NetworkEventConfiguration(
                    event_id="focal-device-bias",
                    kind=EventKind.NODE_BIAS,
                    target_sensor_ids=(focal_sensor_id,),
                    **common_event,
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
        else:
            events.append(
                NetworkEventConfiguration(
                    event_id="focal-device-thermal-window",
                    kind=EventKind.WORLD_FIELD_OFFSET,
                    start_time_s=domain.event_start_s,
                    duration_s=domain.event_duration_s,
                )
            )
    elif scenario == "MIXED_OR_AMBIGUOUS":
        ambiguous_mode = protocol.ambiguous_modes_by_replicate[replicate]
        if ambiguous_mode == "common_field":
            events.append(
                NetworkEventConfiguration(
                    event_id="ambiguous-common-field",
                    kind=EventKind.WORLD_FIELD_OFFSET,
                    **common_event,
                )
            )
        elif node_count == 1:
            events.append(
                NetworkEventConfiguration(
                    event_id="ambiguous-single-instrument-offset",
                    kind=EventKind.NODE_BIAS,
                    target_sensor_ids=(focal_sensor_id,),
                    **common_event,
                )
            )
        else:
            events.append(
                NetworkEventConfiguration(
                    event_id="ambiguous-shared-instrument-offset",
                    kind=EventKind.SHARED_INSTRUMENT_OFFSET,
                    target_sensor_ids=tuple(node.sensor_id for node in nodes),
                    **common_event,
                )
            )
        # Two examples form an explicitly ambiguous common/shared pair. The
        # remaining declared replicates add a focal component but retain the
        # same MIXED_OR_AMBIGUOUS label.
        if replicate in protocol.mixed_focal_component_replicates:
            events.append(
                NetworkEventConfiguration(
                    event_id="mixed-focal-component",
                    kind=EventKind.NODE_BIAS,
                    start_time_s=domain.event_start_s,
                    duration_s=domain.event_duration_s,
                    target_sensor_ids=(focal_sensor_id,),
                    field_offset_T=(focal_extra_nt * 1.0e-9, 0.0, 0.0),
                )
            )

    configuration = NetworkSessionConfiguration(
        session_name="AQSE bounded network study",
        random_seed=episode_seed,
        sampling_rate_Hz=domain.sampling_rate_hz,
        ui_refresh_rate_Hz=5.0,
        time_scale=1.0,
        buffer_duration_s=domain.episode_duration_s,
        environment=environment,
        nodes=tuple(nodes),
        events=tuple(events),
    )
    return configuration, perturbation_nt, focal_extra_nt, focal_sensor_id


def build_network_study_plans(
    protocol: NetworkDemoProtocol = FROZEN_NETWORK_DEMO_PROTOCOL,
) -> tuple[NetworkStudyEpisodePlan, ...]:
    if protocol != FROZEN_NETWORK_DEMO_PROTOCOL:
        raise ValueError("only the frozen aqse-network-demo-v1 protocol is supported")
    assignments = _split_assignments()
    plans: list[NetworkStudyEpisodePlan] = []
    for scenario_index, scenario in enumerate(protocol.scenarios):
        for node_count in protocol.node_counts:
            for replicate in range(protocol.replicates_per_cell):
                seed = _episode_seed(scenario_index, node_count, replicate)
                configuration, amplitude, focal_extra, focal = _study_configuration(
                    scenario=scenario,
                    node_count=node_count,
                    replicate=replicate,
                    episode_seed=seed,
                    generation_domain=protocol.generation_domain_separator,
                )
                lineage_payload = {
                    "study_id": protocol.study_id,
                    "generation_domain": protocol.generation_domain_separator,
                    "scenario_index": scenario_index,
                    "node_count": node_count,
                    "replicate": replicate,
                    "generation_seed": seed,
                }
                lineage_id = _short_identity("network-lineage", lineage_payload)
                episode_payload = {
                    **lineage_payload,
                    "partition": assignments[(scenario, node_count, replicate)],
                    "protocol_digest": protocol.digest,
                }
                plans.append(
                    NetworkStudyEpisodePlan(
                        episode_id=_short_identity("network-episode", episode_payload),
                        generative_lineage_id=lineage_id,
                        generation_domain=protocol.generation_domain_separator,
                        partition=assignments[(scenario, node_count, replicate)],
                        scenario=scenario,
                        node_count=node_count,
                        replicate_index=replicate,
                        episode_seed=seed,
                        focal_sensor_id=focal,
                        perturbation_north_nt=amplitude,
                        focal_extra_nt=focal_extra,
                        network_configuration=configuration.model_dump(mode="json"),
                    )
                )
    return tuple(plans)


def generate_network_study_episode(
    plan: NetworkStudyEpisodePlan,
) -> tuple[NetworkStudyObservation, NetworkStudyLabel]:
    configuration = NetworkSessionConfiguration.model_validate(
        plan.network_configuration
    )
    simulator = NetworkSimulator(configuration)
    domain = FROZEN_NETWORK_DEMO_PROTOCOL.simulation
    reference_frames = []
    supervised_frames = []
    sample_count = int(domain.episode_duration_s * domain.sampling_rate_hz)
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(sample_count):
        sim_time_s = index / domain.sampling_rate_hz
        buffered = simulator.generate(
            session_id=plan.episode_id,
            frame_id=index + 1,
            sim_time_s=sim_time_s,
            epoch_utc=epoch,
            events=configuration.events,
        )
        # Only observation frames cross the feature boundary. Truth is
        # intentionally discarded at the simulator boundary.
        observation = buffered.observation
        if domain.reference_start_s <= sim_time_s < domain.reference_end_s:
            reference_frames.append(observation)
        if (
            domain.supervised_window_start_s
            <= sim_time_s
            < domain.supervised_window_end_s
        ):
            supervised_frames.append(observation)

    reference = build_state8_reference(reference_frames)
    local = extract_latest_state8_window(
        supervised_frames,
        reference=reference,
        profile_id=LOCAL_STATE8_PROFILE_ID,
        sensor_ids=(plan.focal_sensor_id,),
    )
    matching_local = tuple(
        record
        for record in local.records
        if record.sensor_id == plan.focal_sensor_id
        and np.isclose(
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
    if len(matching_local) != 1:
        raise RuntimeError("local study extraction must yield the frozen focal window")
    local_record = matching_local[0]
    network_record = None
    if plan.node_count >= 3:
        network = extract_latest_state8_window(
            supervised_frames,
            reference=reference,
            profile_id=NETWORK_STATE8_PROFILE_ID,
            sensor_ids=(plan.focal_sensor_id,),
        )
        matching_network = tuple(
            record
            for record in network.records
            if record.sensor_id == plan.focal_sensor_id
            and np.isclose(
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
        if len(matching_network) != 1:
            raise RuntimeError(
                "network study extraction must yield the frozen focal window"
            )
        network_record = matching_network[0]
    if not local_record.quality.valid_for_quantum:
        raise RuntimeError("canonical study produced an invalid local feature")
    if network_record is not None and not network_record.quality.valid_for_quantum:
        raise RuntimeError("canonical study produced an invalid network feature")
    observation = NetworkStudyObservation(
        episode_id=plan.episode_id,
        generative_lineage_id=plan.generative_lineage_id,
        partition=plan.partition,
        node_count=plan.node_count,
        focal_sensor_id=plan.focal_sensor_id,
        local_feature=local_record,
        network_feature=network_record,
    )
    label = NetworkStudyLabel(
        episode_id=plan.episode_id,
        generative_lineage_id=plan.generative_lineage_id,
        partition=plan.partition,
        local_label=("NORMAL" if plan.scenario == "NORMAL" else "CHANGE_DETECTED"),
        network_label=plan.scenario,
    )
    return observation, label


def build_network_study(
    protocol: NetworkDemoProtocol = FROZEN_NETWORK_DEMO_PROTOCOL,
) -> NetworkStudyBuild:
    plans = build_network_study_plans(protocol)
    observations: list[NetworkStudyObservation] = []
    labels: list[NetworkStudyLabel] = []
    for plan in plans:
        observation, label = generate_network_study_episode(plan)
        observations.append(observation)
        labels.append(label)
    return NetworkStudyBuild(
        protocol_digest=protocol.digest,
        episode_plans=plans,
        observations=tuple(observations),
        labels=tuple(labels),
    )
