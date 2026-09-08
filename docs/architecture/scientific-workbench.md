# AQSE Scientific Workbench

## Purpose

The frontend is an operational scientific workbook over shared experiment
state. It must not become a slide deck, a decorative dashboard, or a collection
of disconnected demonstrations. Every displayed result must come from a real
backend computation, a real simulator output, or a visibly marked future
capability.

All application labels, buttons, errors, help text, charts, and diagrams are in
English. This document defines behavior and information architecture; it does
not claim that every worksheet is already implemented.

## Current implementation boundary

Milestone 1C implements the workbook shell, overview, finite-vector controls,
continuous 1–8-node session controls and charts, legacy-compatible feature
extraction, and the bounded fixed-theta quantum preview. The QNG worksheet is a
non-executing capability boundary, AFSE exposes contracts only, and the neural
worksheet is explicitly not implemented. The continuous-network relational
feature profile, training orchestration, AFSE mathematics, neural inference,
durable experiment archival, and analysis-worker metrics remain pending.

## Worksheet map

### 1. Overview

Shows the current executed experiment rather than a static architecture image:

- backend and worker readiness;
- simulation lifecycle and simulated time;
- node count, active nodes, and selected sensor;
- latest observation and feature-window age;
- feature-profile and quality summary;
- quantum-preview status and executed theta version;
- explicit statuses for AFSE and the neural model;
- injected truth and predicted diagnosis in separate areas.

### 2. Sensors

Configures the environment and from one to eight nodes. It contains:

- XY geometry with editable altitude or Down coordinate;
- keyboard-accessible numeric pose editing;
- selected-node configuration;
- world and body frame diagrams;
- time-domain traces and vector components;
- saturation, availability, stale-data, and telemetry states;
- explicit controls for same-seed replay and a new realization.

Dragging a node edits a configuration. It does not move the node during an
active acquisition unless a trajectory change is explicitly scheduled and
recorded.

### 3. Features

Selects channel, feature profile, causal window duration, and overlap or hop.
It displays:

- raw and calibrated readout used by the extractor;
- one eight-value vector per accepted window;
- units, extractor version, and provenance;
- per-feature validity and overall preview eligibility;
- rejected windows and reasons;
- the distinction between the implemented legacy-compatible finite-vector
  profile and the candidate, unimplemented continuous-network profile.

### 4. Quantum Engine

Displays the actual TQK8 contract and provides an explicit bounded preview:

- eight qubit wires;
- both feature uploads;
- sixteen draft theta controls;
- the reference dataset and fitted scaler snapshot;
- an explicit Run Quantum Preview action;
- a real kernel heatmap and measured kernel diagnostics;
- draft versus executed theta and stale-result indication.

No circuit is executed merely because a control changed.

### 5. QNG Training

The target design represents training as a separate, cancellable job:

- selected labelled training dataset;
- active objective, loss terms, and weights;
- metric mode, damping, learning rate, step limit, and budget;
- loss and validation history produced by real jobs;
- candidate versus active theta versions;
- promotion or rollback state.

The worksheet must never draw QNG as a post-kernel inference layer.
The current worksheet cannot start or report a job; it reports that the supplied
scientific QNG implementation is available but not connected to sensor training.

### 6. Local Embedding / AFSE

Shows the mandatory architectural boundary and its actual status:

    Architecture defined — mathematical implementation pending

Until Floriano approves the mathematics, the worksheet contains contracts and
lineage only. It contains no dummy vectors, invented scatter plots, or placeholder
numbers presented as results.

### 7. Neural Model

Shows the target downstream role, accepted representation contract, model
version, and status. It must not rename the current demonstration SVC as the
final AQSE neural model. No prediction is displayed unless a real model was
actually executed.

### 8. Experiments

The target experiment archive owns reproducibility and comparison:

- session configuration and command log;
- seed and named substreams;
- observation and truth exports as separate artifacts;
- feature, scaler, theta, kernel, embedding, and model versions;
- train, validation, and test episode splits;
- replay in blind or truth-visible mode;
- comparison of two models on exactly the same observations;
- actual resource, latency, circuit, and metric records.

The current worksheet exposes only browser-held workbench execution records. It
does not constitute durable archival, a complete command log, or a measured
performance record.

## Shared application state

One typed state tree is shared by all worksheets. Each section must distinguish
four categories.

### Draft configuration

Values currently edited by the user but not yet applied. Examples include node
pose, environment parameters, window policy, theta, and job budget.

### Executed configuration

The immutable snapshot accepted for a simulation, feature extraction, preview,
or training run. It includes an identifier, version, and effective simulated
time.

### Computed results

Outputs tied to an executed snapshot: observations, truth, feature windows,
quality, kernel, training history, or predictions.

### Stale state

A result becomes stale when a draft dependency differs from its executed
snapshot. Stale output may remain visible for comparison but must be marked and
must not be presented as the result of the current draft.

Recommended top-level state domains are:

    connection
    session
    environmentDraft / environmentExecuted
    nodeDrafts / nodeExecuted
    observationBuffer
    truthBuffer
    selectedNode and selectedChannel
    featureDraft / featureBatch
    thetaDraft / thetaExecuted
    scalerSnapshot
    quantumPreview
    trainingJob
    afseStatus
    modelStatus
    experimentLineage

Truth state must not be imported by feature or inference selectors. Blind mode
hides truth without changing observations or predictions.

## Operational status model

The workbench must distinguish failures that require different actions.

| State | Meaning | Expected UI behavior |
| --- | --- | --- |
| Backend unavailable | HTTP or transport health failed | Show connection error and retry; do not label a sensor faulty |
| Simulation stopped | Authoritative session is intentionally stopped | Preserve last run with stopped state and data age |
| Simulation paused | Simulated time is not advancing | Preserve persistent process state for resume |
| Application error | A service rejected or failed a computation | Show correlated error identifier without exposing a stack trace |
| Sensor dropout | Simulator produced no observation for a node | Emit null measurement and composable quality flag |
| Frozen sensor | Repeated stale device value | Mark frozen and stale; do not treat it as a new valid sample |
| Saturated sensor | Pre-clipped value exceeded range | Show clipped components and percentage |
| Quantum worker delayed | Preview or inference is behind observations | Once an analysis worker exists, show data age, skipped-window count, and actual last result time |

An HTTP 200 response can legitimately carry a simulated device fault. A device
fault must not become an HTTP 500. Conversely, a backend outage must not be
represented as a physical sensor event.

## Session lifecycle

The authoritative lifecycle is:

    CREATED → RUNNING ⇄ PAUSED → STOPPED
                   ↓
                 ERROR

Reset creates a new initial state; it is not an alias for resume. Current replay
reconstructs the generated frame count from the session configuration, seed, and
scheduled event records and is synchronous only up to 5,000 frames. Longer
sessions must be reset and replayed incrementally. A future durable experiment
archive must preserve the complete configuration and command log. Generate New
Realization creates and displays a new seed before execution and records it in
workbench state.

Two browser tabs must attach to the same authoritative session rather than
creating two physical worlds accidentally. Commands are idempotent and carry a
configuration version. A control update takes effect only at a declared sample
or simulated-time boundary.

## Time and buffering policy

The continuous-network operational starting point is configurable and is not a
hardware specification. Current and candidate settings are:

- implemented default: 100 samples/s per node;
- implemented default: approximately 5 UI refreshes/s;
- candidate only: 4 s analysis windows with 1 s hop;
- implemented default: a bounded recent visualization buffer of 120 s;
- not implemented: chunked archival for longer sessions.

The finite-vector Milestone 1C preview instead uses its own legacy-compatible
window policy: 1 s windows with 50% overlap. These profiles are not silently
interchanged.

Visual decimation affects only drawing. The current session exposes simulation
lag, buffer capacity, overwritten-frame count, and retrieval gaps. A separate
analysis worker, archival stream, queue depth, skipped-window count, prediction
data age, and analysis compute-duration metrics are not implemented yet.

The `quantum_preview_signal` preset adds a declared 5 Hz world-frame validation
source with 8 nT vector-amplitude magnitude so the legacy harmonic profile can
exercise the fixed-theta preview path. It is not AQSE scientific logic. The
field-provider catalog reports constant and synthetic providers configured and
WMM unavailable/not configured; the workbench must not imply geographic WMM
accuracy.

## Scientific graphics

Diagrams must explain frames, transforms, and data provenance. The vector
sensor schematic uses consistent colors for X, Y, and Z across body axes,
traces, calibration clouds, and projections. It distinguishes:

- Earth or reference field in the world frame;
- measured vector in the sensor frame;
- orientation transform;
- simulated truth, observed measurement, and model estimate.

Cause badges and prediction badges are separate components backed by separate
state. A map labelled coverage displays the declared reference source, SNR or
detection criterion, orientation assumption, and uncertainty; it must not draw
universal detection circles.

## Interaction and accessibility

- Every drag operation has a numeric and keyboard-accessible alternative.
- Every slider shows its numerical value and unit.
- Presets expose the coefficients they change.
- A generic randomization control is not permitted.
- Errors are actionable and identify which configuration is rejected.
- Missing estimates display Not available or Not identifiable.
- Charts expose units, time base, data age, and whether data are decimated.
- Decorative motion is avoided; simulation motion represents actual state.

## Status language

Use only statuses supported by executable state:

- Implemented
- Available but not connected
- Architecture defined — mathematical implementation pending
- Not implemented
- Running, Paused, Stopped, Delayed, Failed, or Stale

Terms such as optimized, calibrated, learned, quantum-enhanced, real-time, or
validated require evidence recorded for the exact artifact being displayed.
