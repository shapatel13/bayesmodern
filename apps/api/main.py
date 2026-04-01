from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers.analysis import router as analysis_router
from apps.api.routers.health import router as health_router
from security.secrets import validate_live_llm_config
from utils.config import get_settings
from utils.logging import configure_logging

settings = get_settings()
configure_logging(settings)
secret_status = validate_live_llm_config(settings)

app = FastAPI(
    title="PRIORI-X API",
    version="0.1.0",
    summary="Offline-first Bayesian clinical reasoning research API",
    description=(
        "PRIORI-X is a clinician-research workbench. It is not a clinical autopilot and must "
        "not be used for autonomous medical decision-making."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")


@app.get("/", tags=["meta"])
def read_root() -> dict[str, object]:
    return {
        "name": settings.app_name,
        "mode": "research",
        "allow_live_llm": settings.allow_live_llm,
        "live_llm_provider_ready": secret_status.provider_ready,
        "default_model_provider": settings.default_model_provider,
        "openai_reasoning_model": settings.openai_reasoning_model,
        "disclaimer": "Research only. Not for autonomous clinical use.",
    }
