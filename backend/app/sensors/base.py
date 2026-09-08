from abc import ABC, abstractmethod
from typing import Any


class SensorSource(ABC):
    """Contract for a future quantum sensor data source."""

    @abstractmethod
    def read(self) -> Any:
        """Return the next sensor observation."""

