"""Versioned feature extraction contracts for AQSE observations."""

from app.features.models import (
    FeatureChannel,
    FeatureExtractionRequest,
    FeatureExtractionResponse,
    FeatureProfile,
    FeatureQuality,
    FeatureWindowConfiguration,
    MeasuredVectorSeries,
    WindowFeatureRecord,
)
from app.features.windowed import extract_windowed_magnetometer_features

__all__ = [
    "FeatureChannel",
    "FeatureExtractionRequest",
    "FeatureExtractionResponse",
    "FeatureProfile",
    "FeatureQuality",
    "FeatureWindowConfiguration",
    "MeasuredVectorSeries",
    "WindowFeatureRecord",
    "extract_windowed_magnetometer_features",
]
