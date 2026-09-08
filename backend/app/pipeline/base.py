from abc import ABC, abstractmethod

from app.models import PipelineResult


class AQSEPipeline(ABC):
    """Contract for coordinating AQSE processing stages."""

    @abstractmethod
    def run(self) -> PipelineResult:
        """Run one future processing cycle."""

