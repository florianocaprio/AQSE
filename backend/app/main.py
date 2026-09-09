from fastapi import FastAPI

from app.api.demo import router as demo_router
from app.api.features import router as features_router
from app.api.health import router as health_router
from app.api.network import router as network_router
from app.api.quantum_health import router as quantum_health_router
from app.api.quantum_preview import router as quantum_preview_router
from app.api.sensors import router as sensors_router
from app.api.training import router as training_router
from app.api.workbench import router as workbench_router

app = FastAPI(title="AQSE Backend")
app.include_router(health_router, prefix="/api")
app.include_router(quantum_health_router, prefix="/api")
app.include_router(sensors_router, prefix="/api")
app.include_router(network_router, prefix="/api")
app.include_router(features_router, prefix="/api")
app.include_router(demo_router, prefix="/api")
app.include_router(quantum_preview_router, prefix="/api")
app.include_router(workbench_router, prefix="/api")
app.include_router(training_router, prefix="/api")
