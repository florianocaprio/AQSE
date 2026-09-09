# AQSE 1D.1 archive contract v2

## Scope

Version 2 is the corrective archive contract for new Milestone 1D.1 fixtures.
It does not authorize dataset regeneration, migration, encoding, fitting,
training, test evaluation, AFSE, or neural processing. Existing study archives
remain byte-for-byte historical v1 artifacts.

## Identity layers

AQSE keeps three meanings separate:

1. `observation_content_digest` identifies normalized observable arrays and
   their declared schema, dtype, shape, and units. It excludes episode IDs,
   partitions, labels, seeds, hidden causes, export IDs, and write times.
2. `observation_binding_digest` binds that content to an episode, lineage, and
   preserved raw-source interval. Replay, re-export, and paired relationships
   therefore remain auditable without changing content identity.
3. `scientific_digest` binds numeric content identities, window ledgers, the
   full feature-profile fingerprint, label-channel hashes,
   generation-channel hashes, and software provenance. Operational path,
   duration, peak memory, and write timestamp stay in `execution.json` and do
   not affect numeric identity.

Checksums demonstrate integrity and identity of serialized content. They do
not establish physical authenticity, and sealing is not a cryptographic
defense against a machine administrator.

## Canonical representation and channel separation

`features.npy` is the sole archived numeric representation of the eight
feature values. Partition `metadata.json` contains profile, observable
quality, coverage, indices, times, raw-source references, and window identity,
but does not duplicate feature values or generation plans. Labels live in
`labels/<partition>.json`. Seeds, configured noise, intervention plans, and
assignments live in the separate `generation/<partition>.json` channel.

`load_observations()` returns a typed, immutable-boundary DTO containing only
observable arrays, features, observable quality, profile, and approved
provenance references. `load_labels()` and `load_generation_channel()` are
explicitly separate operations. None is connected to production encoding or
training in 1D.1.

Before issuing a fresh process-local HMAC,
`resign_partition_windows()` reruns the versioned harmonic extractor using
only archived raw observations. It compares features with fixed
`rtol=1e-10` and `atol=1e-12`, and requires exact schema, identity, quality,
and valid-mask consistency. Plans, labels, and configured noise are not inputs
to that recalculation.

## Opaque integrity and semantic access

`verify_archive_opaque()` validates the bounded manifest, containment,
symlink policy, exact file allowlist, byte counts, and streaming SHA-256. It
does not call NumPy or decode partition metadata, labels, quality, or plans.

Semantic loading is partition-scoped. Train and validation do not decode test
payloads. A test request must pass authorization and append an identity-bound,
serialized ledger event before any semantic test read. The writer has a
narrow, separate privilege to validate the complete staging candidate before
atomic publication; this initial validation does not record a test-open event.

NPY headers are inspected before allocation. V2 accepts only the six declared
arrays, their primitive float64/bool dtypes, compatible shapes, and C order.
Element counts, logical/physical byte mismatch, JSON size, non-finite values,
truncation, object/string/structured dtypes, extra files, path traversal, and
symlinks are rejected.

## Software provenance

New v2 archives record algorithm/schema IDs, unit conversion, calibration and
pose identifiers, effective source-file SHA-256 values, Python/NumPy/Pydantic
versions, repository base SHA when discoverable, and dirty status when
discoverable. When Git metadata is unavailable, the revision is explicitly
`unrecorded`; it is never inferred later from a release commit.

## Historical compatibility and non-executed migration proposal

V1 archives remain supported for bounded opaque integrity verification only.
They are not represented as conforming to v2 because they lack the v2 channel
separation and provenance bindings. Semantic use would require a separately
approved rebuild or migration study that:

- works from authorized source samples without opening the reserved test;
- preserves sample values, plans, seeds, splits, and quality policy;
- creates a new archive identity rather than rewriting historical digests;
- records whether provenance is demonstrable or `unrecorded`;
- compares old and candidate payload hashes before publication.

No migration or rebuild is performed by this corrective task.
