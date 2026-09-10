"""AQSE local embedding contracts and approved Nyström implementation."""

from app.embeddings.contracts import (
    KernelGeometryReference,
    LocalEmbeddingBatch,
    LocalEmbeddingEngine,
)
from app.embeddings.nystrom import (
    AFSE_METHOD_ID,
    AFSEQueryContext,
    NystromAFSE,
    NystromAFSEArtifact,
    fit_nystrom_afse,
    query_context_for,
    validate_afse_artifact,
)

__all__ = [
    "KernelGeometryReference",
    "LocalEmbeddingBatch",
    "LocalEmbeddingEngine",
    "AFSE_METHOD_ID",
    "AFSEQueryContext",
    "NystromAFSE",
    "NystromAFSEArtifact",
    "fit_nystrom_afse",
    "query_context_for",
    "validate_afse_artifact",
]
