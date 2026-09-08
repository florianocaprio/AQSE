from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PipelineResult:
    """Result envelope returned by a future AQSE pipeline run."""

    data: Any

