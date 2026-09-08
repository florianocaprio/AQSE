from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

router = APIRouter()


class CapabilityStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "implemented",
        "available_not_connected",
        "architecture_defined",
        "not_implemented",
    ]
    detail: str


class WorkbenchCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector_sensor: CapabilityStatus
    continuous_network: CapabilityStatus
    quantum_preview: CapabilityStatus
    qng_training: CapabilityStatus
    local_embedding_afse: CapabilityStatus
    neural_model: CapabilityStatus
    physical_qpu: CapabilityStatus


@router.get("/workbench/capabilities", response_model=WorkbenchCapabilities)
def workbench_capabilities() -> WorkbenchCapabilities:
    return WorkbenchCapabilities(
        vector_sensor=CapabilityStatus(
            status="implemented",
            detail="Synthetic three-axis magnetometer simulation is available",
        ),
        continuous_network=CapabilityStatus(
            status="implemented",
            detail="Continuous local network supports one to eight simulated nodes",
        ),
        quantum_preview=CapabilityStatus(
            status="implemented",
            detail="Bounded fixed-theta AngleScaler, VQC and TQK preview is available",
        ),
        qng_training=CapabilityStatus(
            status="available_not_connected",
            detail="Scientific QNG implementation available; sensor training integration pending",
        ),
        local_embedding_afse=CapabilityStatus(
            status="architecture_defined",
            detail="Architecture defined; mathematical implementation pending",
        ),
        neural_model=CapabilityStatus(
            status="not_implemented",
            detail="Scientific design pending",
        ),
        physical_qpu=CapabilityStatus(
            status="not_implemented",
            detail="No physical QPU is configured or contacted",
        ),
    )
