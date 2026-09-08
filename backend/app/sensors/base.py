from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from app.sensors.models import SensorAcquisition, SensorConfiguration

ConfigurationT = TypeVar("ConfigurationT", bound=SensorConfiguration)


class SensorSource(ABC, Generic[ConfigurationT]):
    """Generic contract for a configurable sensor acquisition source."""

    @property
    @abstractmethod
    def sensor_type(self) -> str:
        """Return the stable sensor type identifier exposed by the API."""

    @abstractmethod
    def acquire(self, configuration: ConfigurationT) -> SensorAcquisition:
        """Produce one acquisition using the supplied configuration."""
