from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


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
    """Future fixed-size AFSE output; no implementation exists in Milestone 1C."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    method_version: str
    sample_ids: tuple[str, ...]
    vectors: list[list[float]] = Field(min_length=1)


class LocalEmbeddingEngine(Protocol):
    """Boundary owned by the future, explicitly approved AFSE implementation."""

    def transform(self, geometry: KernelGeometryReference) -> LocalEmbeddingBatch:
        """Produce approved local embeddings from trained kernel geometry."""
        ...
