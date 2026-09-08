from fastapi import APIRouter

from app.preprocessing import compute_frequency_spectrum, extract_magnetometer_features
from app.sensors.magnetometer import (
    DEFAULT_MAGNETOMETER_CONFIGURATION,
    QUANTUM_MAGNETOMETER_TYPE,
    QuantumMagnetometerSimulator,
)
from app.sensors.models import (
    MagnetometerConfiguration,
    MagnetometerSimulationResponse,
    SensorCatalogResponse,
    SensorDescriptor,
)

router = APIRouter()
magnetometer = QuantumMagnetometerSimulator()


@router.get("/sensors", response_model=SensorCatalogResponse)
def list_simulated_sensors() -> SensorCatalogResponse:
    return SensorCatalogResponse(
        sensors=[
            SensorDescriptor(
                sensor_type=QUANTUM_MAGNETOMETER_TYPE,
                display_name="Quantum Magnetometer",
                simulation=True,
                physical_unit="nT",
            )
        ]
    )


@router.get(
    "/sensors/magnetometer/defaults",
    response_model=MagnetometerConfiguration,
)
def magnetometer_defaults() -> MagnetometerConfiguration:
    return DEFAULT_MAGNETOMETER_CONFIGURATION


@router.post(
    "/sensors/magnetometer/simulate",
    response_model=MagnetometerSimulationResponse,
)
def simulate_magnetometer(
    configuration: MagnetometerConfiguration,
) -> MagnetometerSimulationResponse:
    acquisition = magnetometer.acquire(configuration)
    spectrum = compute_frequency_spectrum(acquisition)
    features = extract_magnetometer_features(acquisition, spectrum)
    return MagnetometerSimulationResponse(
        acquisition=acquisition,
        spectrum=spectrum,
        features=features,
    )
