from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class QuantumEngine(ABC):
    """Application boundary for an exact AQSE quantum state engine."""

    @abstractmethod
    def state(
        self,
        features: Sequence[float],
        parameters: Sequence[float],
    ) -> NDArray[np.complex128]:
        """Return one exact statevector without contacting a physical QPU."""
