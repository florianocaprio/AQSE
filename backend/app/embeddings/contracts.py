from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KernelGeometryReference(BaseModel):
    """Versioned reference to trained TQK geometry, not an embedding."""

    model_config = ConfigDict(extra="forbid")

    kernel_id: str
    sample_ids: tuple[str, ...]
    feature_profile_id: str
    theta_version: str
    scaler_version: str
    trained: bool


class LocalEmbeddingBatch(BaseModel):
    """Versioned fixed-size AFSE output produced by an approved implementation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    method_version: str
    sample_ids: tuple[str, ...]
    vectors: list[list[float]] = Field(min_length=1)
    reference_size: int = Field(gt=0)
    reconstruction_residuals: tuple[float, ...]
    heuristic_ood: tuple[bool, ...]

    @model_validator(mode="after")
    def validate_rows(self) -> LocalEmbeddingBatch:
        row_count = len(self.sample_ids)
        if len(self.vectors) != row_count:
            raise ValueError("embedding vectors must match sample identifiers")
        if len(self.reconstruction_residuals) != row_count:
            raise ValueError("embedding residuals must match sample identifiers")
        if len(self.heuristic_ood) != row_count:
            raise ValueError("embedding OOD flags must match sample identifiers")
        if any(len(row) != self.reference_size for row in self.vectors):
            raise ValueError("embedding vectors must use the frozen reference dimension")
        return self


class LocalEmbeddingEngine(Protocol):
    """Stable application boundary for an explicitly approved AFSE implementation."""

    def transform(self, geometry: KernelGeometryReference) -> LocalEmbeddingBatch:
        """Produce approved local embeddings from trained kernel geometry."""
        ...
