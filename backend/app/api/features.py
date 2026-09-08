from fastapi import APIRouter, HTTPException

from app.features import (
    FeatureExtractionRequest,
    FeatureExtractionResponse,
    extract_windowed_magnetometer_features,
)

router = APIRouter()


@router.post(
    "/features/vector-magnetometer/extract",
    response_model=FeatureExtractionResponse,
)
def extract_vector_magnetometer_features(
    request: FeatureExtractionRequest,
) -> FeatureExtractionResponse:
    try:
        return extract_windowed_magnetometer_features(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
