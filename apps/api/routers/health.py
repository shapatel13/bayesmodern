from __future__ import annotations

from fastapi import APIRouter

from apps.api.schemas.health import HealthResponse
from security.secrets import validate_live_llm_config
from utils.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    secret_status = validate_live_llm_config(settings)
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
        safety_mode=settings.safety_mode,
        allow_live_llm=settings.allow_live_llm,
        provider_ready=secret_status.provider_ready,
        missing_required_secrets=secret_status.missing_required_secrets,
    )

