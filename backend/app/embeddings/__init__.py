"""Application boundary for the future AQSE local embedding stage."""

from app.embeddings.contracts import (
    KernelGeometryReference,
    LocalEmbeddingBatch,
    LocalEmbeddingEngine,
)

__all__ = [
    "KernelGeometryReference",
    "LocalEmbeddingBatch",
    "LocalEmbeddingEngine",
]
