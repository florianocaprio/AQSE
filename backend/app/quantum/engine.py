from abc import ABC, abstractmethod
from typing import Any


class QuantumEngine(ABC):
    """Adapter boundary for Qiskit code supplied by the project author."""

    @abstractmethod
    def execute(self, sensor_data: Any) -> Any:
        """Process sensor data using a future quantum implementation."""

