from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.quantum_health import router as quantum_health_router
from app.api.sensors import router as sensors_router

app = FastAPI(title="AQSE Backend")
app.include_router(health_router, prefix="/api")
app.include_router(quantum_health_router, prefix="/api")
app.include_router(sensors_router, prefix="/api")
