# Local Functional Embedding / AFSE Boundary

## Status

**Architecture defined — mathematical implementation pending.**

AFSE is a mandatory component of the target AQSE architecture, but its exact
mathematics has not been approved by Floriano. This document defines only the
software and scientific boundary required to avoid conflating a batch kernel
with a fixed-size local representation.

No AFSE vectors, performance values, plots, or trained models are claimed here.
Milestone 1C contains typed `KernelGeometryReference`, `LocalEmbeddingBatch`, and
`LocalEmbeddingEngine` boundary contracts only; no transform implementation is
connected. The supplied QNG implementation remains a separate training
capability that is available but not connected, and the final neural model remains not
implemented. Neither boundary may be bypassed by relabelling preview output.

## Position in AQSE

    trained and versioned TQK geometry
                    ↓
       Local Functional Embedding / AFSE
                    ↓
       fixed-size local representation z(x)
                    ↓
             classical neural model

During inference, AFSE uses a frozen quantum encoder and a frozen fitted AFSE
artifact. During training, any AFSE-specific fitting is a declared operation on
training data only. It is not an implicit transformation performed independently
on every query batch.

## Why a Gram row is insufficient

For a preview batch of M samples, the Gram row

    [K(x,x1), ..., K(x,xM)]

has dimension M and changes meaning when the batch membership or order changes.
The complete M×M kernel is relational geometry, not automatically a stable local
embedding for one sample.

An AFSE contract must instead define how a query sample obtains a fixed-size
representation relative to a versioned, frozen reference object. The output
dimension, basis, regularization, out-of-reference behavior, and update policy
must be approved before implementation.

## Typed boundary

An eventual AFSE input contract should contain:

    query_feature_profile_id
    query_window_ids
    encoder_version
    scaler_id
    theta_version
    quantum_backend_semantics
    reference_geometry_id
    kernel evaluations required by the approved method

An eventual fitted artifact should contain:

    afse_method_id
    afse_version
    approved hyperparameters
    fitted training/reference dataset identifier
    reference_geometry identifier
    output dimension
    numerical regularization metadata
    software version and seed where applicable

An eventual output should contain:

    query_window_id
    z with the approved fixed dimension
    validity and out-of-reference status
    encoder, scaler, theta, reference, and AFSE versions
    execution metadata

The boundary accepts no simulator truth or hidden event cause.

## Explicitly unapproved mathematics

Until Floriano gives a separate instruction, AQSE must not select or implement:

- Nyström approximation;
- kernel PCA or ordinary PCA;
- landmark similarity embedding;
- spectral embedding;
- random projection;
- learned projection;
- autoencoder;
- neural kernel embedding;
- any other dimensionality-reduction formula.

A placeholder implementation returning zeros, random vectors, raw Gram rows, or
identity-transformed features is also prohibited. The workbench exposes the
boundary and status without fabricating output.

## Decisions required before implementation

Floriano must explicitly approve at least:

1. the mathematical map from TQK geometry to z(x);
2. the fixed output dimension;
3. reference or landmark selection and its training-only policy;
4. normalization and regularization;
5. fitting, update, and invalidation behavior when theta changes;
6. out-of-reference detection or confidence semantics;
7. compatibility with feature profiles and missing-data masks;
8. validation criteria and classical baselines.

If the selected method uses landmarks, the landmark bank must come only from
training data. If it uses spectral truncation, eigenvalue thresholds and
regularization must be documented. These statements are constraints, not an
endorsement of either method.

## Versioning and invalidation

Changing any of the following creates an incompatible representation space:

- feature profile or calibration;
- AngleScaler snapshot;
- theta or VQC implementation;
- TQK semantics;
- AFSE method, reference object, output dimension, or regularization.

Old and new z vectors must not be mixed under one downstream model. A model
checkpoint records the full representation lineage and is promoted only after
validation on frozen episodes.

## Workbench behavior while pending

The AFSE worksheet may show:

- the canonical input and output contracts;
- upstream encoder and reference-geometry versions;
- decisions still awaiting approval;
- absence of a fitted artifact;
- the status Architecture defined — mathematical implementation pending.

It must not show:

- a generated embedding plot;
- synthetic latent coordinates;
- a completed status;
- a neural prediction that bypasses the missing representation;
- the quantum-preview heatmap relabelled as AFSE.
