from __future__ import annotations

from threading import BoundedSemaphore

# One process-local admission slot for every explicit heavy exact-state operation.
heavy_quantum_slot = BoundedSemaphore(value=1)
