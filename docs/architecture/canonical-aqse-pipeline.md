# Canonical AQSE Pipeline

## Document status

This document defines the target architecture for AQSE Milestone 1C and the
continuous 1–8-node research direction. It is a design contract, not evidence
that every described component has already been implemented or validated.

The protected baseline at the start of Milestone 1C consists of the scalar
Milestone 1B magnetometer, its legacy eight-feature extractor, and Floriano's
TQK8 implementation. Milestone 1C now implements finite-vector sensing, the
continuous 1–8-node simulator/session API, and a bounded fixed-theta quantum
preview. That implementation status is not, by itself, a claim of scientific
validation. The candidate continuous-network feature profile, AFSE, and the
final neural model remain pending.

## Canonical processing path

    Sensor
      ↓
    Signal Processing
      ↓
    Feature Extraction
      ↓
    8-dimensional Feature Vector
      ↓
    VQC
      ↓
    TQK
      ↓
    Local Functional Embedding / AFSE
      ↓
    Classical Neural Network
      ↓
    Task Output

This complete path is the target architecture. A temporary milestone may stop
at a real kernel preview, but it must not relabel a Gram matrix as an embedding
or hide the mandatory AFSE boundary.

The number of sensors, measured channels, features, and qubits are independent
quantities:

- the network contains from one to eight sensor nodes;
- each node declares its real measurement mode;
- every accepted analysis window produces exactly eight features under one
  versioned feature profile;
- TQK8 receives eight encoded angles on eight logical qubits;
- the same encoder can process windows from different nodes when their feature
  profiles and model versions are compatible.

Eight sensors do not imply one qubit per sensor, eight separate circuits, or
twenty-four concatenated vector components.

## Training and inference are different computations

### Controlled training loop

    labelled training windows
             ↓
       VQC(theta)
             ↓
           TQK
             ↓
      alignment loss
             ↓
            QNG
             ↓
       updated theta
             └──────────────→ VQC(theta')

QNG is the optimizer that updates the VQC parameters. It is not a layer through
which every inference sample passes. Labels enter the training objective; QNG
does not create labels and does not determine whether a disturbance is physical
or instrumental.

### Frozen inference path

    measured window
          ↓
    frozen feature profile
          ↓
    frozen AngleScaler
          ↓
    VQC(theta*)
          ↓
    TQK(theta*)
          ↓
    versioned AFSE representation
          ↓
    frozen classical model
          ↓
    prediction and uncertainty status

Any deployed inference result must identify the exact feature profile, scaler,
theta snapshot, quantum backend mode, AFSE version, and downstream model that
produced it. Updating theta invalidates cached kernels and representations tied
to the previous encoder version.

## Causal information channels

AQSE separates three contracts.

### Observation channel

Contains only information that a real acquisition path could expose:

- measured scalar or vector samples;
- acquisition and arrival times;
- measured or nominal pose, with provenance;
- observed temperature and operational telemetry;
- calibration identifier;
- quality, availability, clipping, and communication flags.

### Truth channel

Contains simulator-only information:

- exact environmental field and source states;
- injected physical events;
- hidden device parameters and fault causes;
- exact source trajectories;
- seed and generator state where required for replay.

Truth may create labels and evaluation targets. It must never enter feature
extraction, scaling, inference, or prediction through a hidden field.

### Prediction channel

Contains model outputs and their lineage:

- window and node or network identifier;
- model and backend version;
- scores or probabilities with calibration status;
- uncertainty or not-identifiable status;
- data age and execution duration;
- feature, scaler, theta, and representation versions.

The predictor must produce the same result for identical observations and
model state even if truth is hidden, changed, or unavailable.

## Clock and execution separation

The system has independent cadences for:

1. simulated physical time;
2. sensor sampling;
3. observation transport, when a nonzero transport model is introduced;
4. UI refresh;
5. feature-window analysis;
6. quantum preview or inference;
7. controlled training jobs.

Browser rendering must not advance the physical model or draw random samples.
Changing UI refresh rate must not alter the generated signal. A slow quantum
worker must not block the authoritative simulator. Before continuous analysis
or inference is connected, its skipped-window, queue-depth, data-age, and
execution-duration metrics must be implemented and exposed; they are not
provided by the current simulator/session API.

Each continuous session has one authoritative backend job. Sequence numbers,
simulated time, acquisition time, arrival time, configuration version, and
effective frame define ordering. The implemented local model records an arrival
timestamp with zero transport delay. Pause, resume, stop, reconnect, and replay
semantics must be verified at the session boundary; synchronous replay is
bounded to 5,000 frames.

## Module responsibilities

| Boundary | Responsibility | Must not do |
| --- | --- | --- |
| Environment | Produce world-frame magnetic truth from fields, gradients, sources, trajectories, and physical events | Add device readout failures |
| Sensor model | Transform world truth into device-frame observations with response, errors, saturation, availability, and telemetry | Change world truth to simulate an electronic fault |
| Synchronization | Align only samples already acquired and expose gaps or clock uncertainty | Use future samples in an online window |
| Feature extraction | Produce a versioned eight-value vector plus measured quality | Read simulator truth or silently combine incompatible channels |
| Quantum adapter | Reuse the supplied AngleScaler, VQC, and TQK implementation | Reimplement Floriano's circuit or run QNG during preview |
| Training service | Evaluate the declared loss and update theta through the supplied training implementation | Treat training loss as held-out performance |
| AFSE | Eventually convert trained kernel geometry into a fixed-size local representation | Invent an embedding before its mathematics is approved |
| Classical model | Map the approved representation to a task output | Present a score as calibrated certainty without evidence |
| Experiment archive | Preserve configuration, lineage, observations, truth separately, and actual execution records | Mix incompatible model or feature versions |

## Version and provenance tuple

Every derived artifact must be attributable to at least:

    session_id
    observation_schema_version
    feature_profile_id and extractor_version
    reference_dataset_id
    scaler_id
    theta_version
    quantum_backend_mode
    kernel_or_embedding_version
    downstream_model_version

An artifact with a missing or incompatible tuple is stale and must not be
silently combined with current results.

## Current Milestone 1C capability status

| Capability | Current status |
| --- | --- |
| Scalar finite magnetometer simulation | Implemented and validated in Milestone 1B |
| Legacy scalar eight-feature profile | Implemented and validated in Milestone 1B |
| TQK8 exact-state engines | Implemented and validated in Milestone 1A |
| Vector finite magnetometer | Implemented in Milestone 1C; validation evidence remains separate |
| Real bounded sensor-to-kernel preview | Implemented in Milestone 1C with fixed theta and M ≤ 128 |
| Continuous 1–8-node network | Simulator, in-memory session API, bounded buffer, and SSE implemented in Milestone 1C |
| Continuous-network 4 s / 1 s-hop feature profile | Candidate contract; not implemented or scientifically frozen |
| QNG sensor-training integration | Scientific implementation available but not connected |
| AFSE mathematics | Pending Floriano's explicit approval |
| Final neural model | Not implemented |
| Physical QPU execution | Not configured and not implied |

## Non-negotiable invariants

- Preserve the original TQK8 scientific implementation and tests.
- Preserve the scalar Milestone 1B API and feature semantics.
- Use SI units internally for new network physics; convert explicitly at API
  and UI boundaries.
- Keep simulated truth separate from observed data and predictions.
- Never encode a missing sample as a valid zero measurement.
- Never report an unavailable estimator as a fabricated number.
- Never present simulation, ideal statevector execution, or a kernel matrix as
  physical quantum-sensor state.
- Never claim improved hardware sensitivity or quantum advantage without a
  controlled, reproducible comparison.

## Related specifications

- [Scientific workbench](scientific-workbench.md)
- [Scientific scope and limitations](scientific-scope-and-limitations.md)
- [Vector magnetometer network](../sensors/vector-magnetometer-network.md)
- [Feature profiles and quality](../features/feature-profiles-and-quality.md)
- [Quantum preview contract](../quantum/quantum-preview-contract.md)
- [AFSE boundary](../quantum/afse-boundary.md)
- [Milestone 1C validation plan](../validation/milestone-1c-validation-plan.md)
