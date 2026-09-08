from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.sensors.base import SensorSource
from app.sensors.models import MagnetometerConfiguration, SensorAcquisition

QUANTUM_MAGNETOMETER_TYPE = "quantum_magnetometer"
DEFAULT_MAGNETOMETER_CONFIGURATION = MagnetometerConfiguration()


class QuantumMagnetometerSimulator(SensorSource[MagnetometerConfiguration]):
    """Configurable synthetic magnetic-field source for local research demos.

    This simulator is intentionally not a model of a specific commercial
    instrument and makes no claim of experimental accuracy.
    """

    @property
    def sensor_type(self) -> str:
        return QUANTUM_MAGNETOMETER_TYPE

    def acquire(
        self,
        configuration: MagnetometerConfiguration,
    ) -> SensorAcquisition:
        sample_count = configuration.number_of_samples
        time = np.arange(sample_count, dtype=np.float64) / configuration.sampling_rate

        sinusoid = configuration.amplitude * np.sin(
            2.0 * np.pi * configuration.frequency * time + configuration.phase
        )
        drift = configuration.drift_rate * time
        rng = np.random.default_rng(configuration.random_seed)
        noise = rng.normal(0.0, configuration.noise_std, size=sample_count)

        anomaly_width = max(
            3.0 / configuration.sampling_rate,
            configuration.duration * 0.02,
        )
        anomaly = np.zeros(sample_count, dtype=np.float64)
        if configuration.anomaly_enabled and configuration.anomaly_time is not None:
            offset = (time - configuration.anomaly_time) / anomaly_width
            anomaly = configuration.anomaly_amplitude * np.exp(-0.5 * offset**2)

        signal = configuration.background_field + sinusoid + drift + noise + anomaly

        return SensorAcquisition(
            sensor_id=configuration.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=datetime.now(timezone.utc),
            sampling_rate=configuration.sampling_rate,
            time=time.tolist(),
            signal=signal.tolist(),
            time_unit="s",
            physical_unit="nT",
            configuration=configuration.model_dump(mode="json"),
            metadata={
                "temperature": configuration.temperature,
                "temperature_unit": "K",
                "anomaly_shape": "gaussian_transient",
                "anomaly_width": anomaly_width,
                "anomaly_width_unit": "s",
                "scientific_accuracy_claimed": False,
            },
        )
