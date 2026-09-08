from __future__ import annotations

from app.network.models import (
    DipoleSourceConfiguration,
    EnvironmentConfiguration,
    EventKind,
    FiniteVectorSimulationConfiguration,
    GaussianAnomalyConfiguration,
    MeasurementMode,
    MotionKind,
    NetworkEventConfiguration,
    NetworkPresetCatalog,
    NetworkPresetDescriptor,
    NetworkSessionConfiguration,
    NodeErrorConfiguration,
    NodeMotionConfiguration,
    PeriodicFieldConfiguration,
    SensorNodeConfiguration,
    VectorScenario,
    VectorScenarioCatalog,
    VectorScenarioDescriptor,
)

_NODE_POSITIONS_M = (
    (-0.5, -0.5, 0.0),
    (0.5, -0.5, 0.0),
    (0.5, 0.5, 0.0),
    (-0.5, 0.5, 0.0),
    (-0.5, -0.5, 0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
)


def _default_errors() -> NodeErrorConfiguration:
    return NodeErrorConfiguration(
        white_noise_std_T_per_sample=(0.5e-9,) * 3,
        random_walk_q_T2_per_s=(1.0e-22,) * 3,
        ou_sigma_T=(0.5e-9,) * 3,
        ou_tau_s=2.0,
        # Disabled by default: bandwidth is an explicit modelling choice.
        bandwidth_Hz=None,
    )


def _default_node(
    index: int,
    position_m: tuple[float, float, float],
) -> SensorNodeConfiguration:
    return SensorNodeConfiguration(
        sensor_id=f"S{index + 1}",
        role="sensor",
        profile="synthetic_magnetometer",
        measurement_mode=MeasurementMode.VECTOR,
        position_m=position_m,
        errors=_default_errors(),
    )


def default_network_configuration(node_count: int = 4) -> NetworkSessionConfiguration:
    if not 1 <= node_count <= 8:
        raise ValueError("node_count must be between 1 and 8")
    positions = ((0.0, 0.0, 0.0),) if node_count == 1 else _NODE_POSITIONS_M
    nodes = tuple(
        _default_node(index, positions[index]) for index in range(node_count)
    )
    return NetworkSessionConfiguration(
        nodes=nodes,
        environment=EnvironmentConfiguration(
            dipoles=(
                DipoleSourceConfiguration(
                    source_id="reference-dipole",
                    initial_position_m=(0.0, 0.0, 1.5),
                    moment_A_m2=(0.0, 0.0, 0.5),
                    minimum_distance_m=0.1,
                ),
            )
        ),
    )


def network_preset_configuration(preset_id: str) -> NetworkSessionConfiguration:
    node_counts = {
        "single_sensor": 1,
        "gradiometry": 2,
        "localization_2d": 4,
        "tracking_3d": 8,
        "area_monitoring": 4,
        "quantum_preview_signal": 4,
        "eight_node_event_demo": 8,
        "single_sensor_ambiguity_demo": 1,
    }
    try:
        configuration = default_network_configuration(node_counts[preset_id])
    except KeyError as exc:
        raise ValueError(f"unknown network preset: {preset_id}") from exc
    if preset_id == "quantum_preview_signal":
        direction_scale = 8.0e-9 / (20.0**2 + 45.0**2) ** 0.5
        periodic = PeriodicFieldConfiguration(
            source_id="preview-harmonic-5Hz",
            amplitude_world_T=(20.0 * direction_scale, 0.0, 45.0 * direction_scale),
            frequency_Hz=5.0,
            phase_rad=0.0,
        )
        configuration = NetworkSessionConfiguration.model_validate(
            {
                **configuration.model_dump(),
                "environment": {
                    **configuration.environment.model_dump(),
                    "periodic_fields": [periodic.model_dump()],
                },
            }
        )
    elif preset_id == "eight_node_event_demo":
        mobile_source = DipoleSourceConfiguration(
            source_id="mobile-demo-dipole",
            # At t=8 s the source crosses the midpoint above S5/S6.
            initial_position_m=(-4.0, -0.5, 1.0),
            velocity_m_per_s=(0.5, 0.0, 0.0),
            moment_A_m2=(0.0, 0.0, 0.5),
            minimum_distance_m=0.1,
        )
        events = (
            NetworkEventConfiguration(
                event_id="demo-common-background",
                kind=EventKind.WORLD_FIELD_OFFSET,
                start_time_s=2.0,
                duration_s=2.0,
                field_offset_T=(15.0e-9, -5.0e-9, 10.0e-9),
            ),
            NetworkEventConfiguration(
                event_id="demo-s3-drift",
                kind=EventKind.NODE_DRIFT,
                start_time_s=4.0,
                duration_s=3.0,
                target_sensor_ids=("S3",),
                drift_rate_T_per_s=(2.0e-9, 0.0, -1.0e-9),
            ),
            NetworkEventConfiguration(
                event_id="demo-s1-s2-shared-offset",
                kind=EventKind.SHARED_INSTRUMENT_OFFSET,
                start_time_s=9.0,
                duration_s=2.0,
                target_sensor_ids=("S1", "S2"),
                field_offset_T=(8.0e-9, 8.0e-9, 0.0),
            ),
            NetworkEventConfiguration(
                event_id="demo-s4-dropout",
                kind=EventKind.DROPOUT,
                start_time_s=11.0,
                duration_s=1.0,
                target_sensor_ids=("S4",),
            ),
            NetworkEventConfiguration(
                event_id="demo-simultaneous-physical",
                kind=EventKind.WORLD_FIELD_OFFSET,
                start_time_s=13.0,
                duration_s=2.0,
                field_offset_T=(-20.0e-9, 5.0e-9, 0.0),
            ),
            NetworkEventConfiguration(
                event_id="demo-simultaneous-s4-fault",
                kind=EventKind.DROPOUT,
                start_time_s=13.0,
                duration_s=2.0,
                target_sensor_ids=("S4",),
            ),
        )
        configuration = NetworkSessionConfiguration.model_validate(
            {
                **configuration.model_dump(),
                "session_name": "AQSE eight-node event demo",
                "environment": {
                    **configuration.environment.model_dump(),
                    "dipoles": [mobile_source.model_dump()],
                },
                "events": [event.model_dump() for event in events],
            }
        )
    elif preset_id == "single_sensor_ambiguity_demo":
        events = (
            NetworkEventConfiguration(
                event_id="single-world-change",
                kind=EventKind.WORLD_FIELD_OFFSET,
                start_time_s=1.0,
                duration_s=2.0,
                field_offset_T=(10.0e-9, 0.0, 0.0),
            ),
            NetworkEventConfiguration(
                event_id="single-device-bias",
                kind=EventKind.NODE_BIAS,
                start_time_s=4.0,
                duration_s=2.0,
                target_sensor_ids=("S1",),
                field_offset_T=(10.0e-9, 0.0, 0.0),
            ),
        )
        configuration = NetworkSessionConfiguration.model_validate(
            {
                **configuration.model_dump(),
                "session_name": "AQSE single-sensor ambiguity demo",
                "events": [event.model_dump() for event in events],
            }
        )
    return configuration


def network_presets() -> NetworkPresetCatalog:
    return NetworkPresetCatalog(
        presets=(
            NetworkPresetDescriptor(
                preset_id="single_sensor",
                display_name="Single-sensor control",
                recommended_node_count=1,
                description="One sensor; local diagnostics only, no network inference.",
            ),
            NetworkPresetDescriptor(
                preset_id="gradiometry",
                display_name="Basic gradiometry",
                recommended_node_count=2,
                description="A pair with a declared baseline; observability is not guaranteed.",
            ),
            NetworkPresetDescriptor(
                preset_id="localization_2d",
                display_name="2D experimental geometry",
                recommended_node_count=4,
                description="A planar square geometry; not a universal sensor minimum.",
            ),
            NetworkPresetDescriptor(
                preset_id="tracking_3d",
                display_name="3D experimental geometry",
                recommended_node_count=8,
                description="A non-coplanar cube geometry; not a universal sensor minimum.",
            ),
            NetworkPresetDescriptor(
                preset_id="area_monitoring",
                display_name="Area monitoring",
                recommended_node_count=4,
                description="A configurable local array with no universal coverage claim.",
            ),
            NetworkPresetDescriptor(
                preset_id="quantum_preview_signal",
                display_name="Quantum preview harmonic signal",
                recommended_node_count=4,
                description=(
                    "Explicit 5 Hz world-frame source for validating the fixed-theta "
                    "preview path; it is not AQSE scientific logic."
                ),
            ),
            NetworkPresetDescriptor(
                preset_id="eight_node_event_demo",
                display_name="Eight-node causal event demo",
                recommended_node_count=8,
                description=(
                    "Reproducible mobile-source and multilabel fault timeline; injected "
                    "causes are truth labels, not automatic diagnoses."
                ),
            ),
            NetworkPresetDescriptor(
                preset_id="single_sensor_ambiguity_demo",
                display_name="Single-sensor ambiguity control",
                recommended_node_count=1,
                description=(
                    "Separate equal-magnitude world-field and device-bias phases show why "
                    "one sensor alone cannot identify a cause from similar readings."
                ),
            ),
        )
    )


def vector_scenarios() -> VectorScenarioCatalog:
    return VectorScenarioCatalog(
        scenarios=(
            VectorScenarioDescriptor(
                scenario_id=VectorScenario.STATIC_REFERENCE,
                display_name="Static reference",
                description="Fixed pose in a uniform local NED magnetic field.",
            ),
            VectorScenarioDescriptor(
                scenario_id=VectorScenario.CALIBRATION_TUMBLE,
                display_name="Calibration tumble",
                description="Deterministic low-discrepancy Haar pose sweep over SO(3).",
            ),
            VectorScenarioDescriptor(
                scenario_id=VectorScenario.HIGH_DYNAMIC,
                display_name="High dynamic rotation",
                description="Continuous multi-axis rotation with explicit angular rates.",
            ),
            VectorScenarioDescriptor(
                scenario_id=VectorScenario.LINEAR_TRANSLATION,
                display_name="Linear translation",
                description="Continuous translation through a symmetric traceless gradient.",
            ),
            VectorScenarioDescriptor(
                scenario_id=VectorScenario.LOCAL_ANOMALY_CROSSING,
                display_name="Local anomaly crossing",
                description="Continuous pass through a smooth world-frame Gaussian anomaly.",
            ),
            VectorScenarioDescriptor(
                scenario_id=VectorScenario.COMBINED_STRESS,
                display_name="Combined stress",
                description="Motion, spatial field and composable instrument events together.",
            ),
        )
    )


def default_vector_configuration(
    scenario: VectorScenario = VectorScenario.STATIC_REFERENCE,
) -> FiniteVectorSimulationConfiguration:
    sensor_id = "vector-magnetometer-001"
    environment = EnvironmentConfiguration()
    events: tuple[NetworkEventConfiguration, ...] = ()
    position_m = (0.0, 0.0, 0.0)
    motion = NodeMotionConfiguration()
    errors = _default_errors().model_copy(
        update={
            "gain_matrix": (
                (1.01, 0.0, 0.0),
                (0.0, 0.995, 0.0),
                (0.0, 0.0, 1.005),
            ),
            "soft_iron_matrix": (
                (1.02, 0.01, 0.0),
                (0.01, 0.99, 0.005),
                (0.0, 0.005, 1.01),
            ),
            "cross_axis_matrix": (
                (1.0, 0.004, -0.002),
                (-0.003, 1.0, 0.003),
                (0.002, -0.004, 1.0),
            ),
        }
    )

    if scenario is VectorScenario.CALIBRATION_TUMBLE:
        motion = NodeMotionConfiguration(kind=MotionKind.CALIBRATION_TUMBLE)
    elif scenario is VectorScenario.HIGH_DYNAMIC:
        motion = NodeMotionConfiguration(
            kind=MotionKind.HIGH_DYNAMIC,
            angular_rate_rad_per_s=(5.0, 7.0, 9.0),
            modulation_frequency_Hz=(0.37, 0.53, 0.71),
        )
    elif scenario is VectorScenario.LINEAR_TRANSLATION:
        position_m = (-1.0, -0.25, 0.0)
        motion = NodeMotionConfiguration(
            kind=MotionKind.LINEAR_TRANSLATION,
            velocity_m_per_s=(0.6, 0.2, 0.0),
        )
        environment = EnvironmentConfiguration(
            gradient_T_per_m=(
                (100.0e-9, 20.0e-9, 0.0),
                (20.0e-9, -40.0e-9, 0.0),
                (0.0, 0.0, -60.0e-9),
            )
        )
    elif scenario is VectorScenario.LOCAL_ANOMALY_CROSSING:
        position_m = (-1.0, 0.0, 0.0)
        motion = NodeMotionConfiguration(
            kind=MotionKind.LOCAL_ANOMALY_CROSSING,
            velocity_m_per_s=(1.0, 0.0, 0.0),
        )
        environment = EnvironmentConfiguration(
            gaussian_anomalies=(
                GaussianAnomalyConfiguration(
                    anomaly_id="finite-local-anomaly",
                    peak_amplitude_T=200.0e-9,
                    direction_world=(0.0, 0.0, 1.0),
                    center_position_m=(0.0, 0.0, 0.0),
                    spatial_scale_m=0.25,
                ),
            )
        )
    elif scenario is VectorScenario.COMBINED_STRESS:
        position_m = (-0.8, -0.2, 0.0)
        motion = NodeMotionConfiguration(
            kind=MotionKind.COMBINED_STRESS,
            velocity_m_per_s=(0.6, 0.15, 0.0),
            angular_rate_rad_per_s=(1.7, 2.3, 3.1),
        )
        environment = EnvironmentConfiguration(
            gradient_T_per_m=(
                (80.0e-9, 10.0e-9, 0.0),
                (10.0e-9, -30.0e-9, 0.0),
                (0.0, 0.0, -50.0e-9),
            ),
            dipoles=(
                DipoleSourceConfiguration(
                    source_id="combined-source",
                    initial_position_m=(0.0, 0.1, 0.6),
                    moment_A_m2=(0.1, 0.0, 0.2),
                    minimum_distance_m=0.1,
                ),
            ),
        )
        events = (
            NetworkEventConfiguration(
                event_id="combined-bias",
                kind=EventKind.NODE_BIAS,
                start_time_s=0.5,
                duration_s=1.0,
                target_sensor_ids=(sensor_id,),
                field_offset_T=(20.0e-9, -10.0e-9, 5.0e-9),
            ),
            NetworkEventConfiguration(
                event_id="combined-noise",
                kind=EventKind.NODE_NOISE_BURST,
                start_time_s=1.0,
                duration_s=0.5,
                target_sensor_ids=(sensor_id,),
                noise_multiplier=4.0,
            ),
        )

    return FiniteVectorSimulationConfiguration(
        scenario=scenario,
        environment=environment,
        node=SensorNodeConfiguration(
            sensor_id=sensor_id,
            measurement_mode=MeasurementMode.VECTOR,
            position_m=position_m,
            motion=motion,
            errors=errors,
        ),
        events=events,
    )
