from __future__ import annotations

from pydantic import BaseModel

from utils.config import Settings


class SecretStatus(BaseModel):
    allow_live_llm: bool
    provider_ready: bool
    missing_required_secrets: list[str]


def mask_secret(value: str | None) -> str:
    if not value:
        return "<unset>"
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}***{value[-3:]}"


def validate_live_llm_config(settings: Settings) -> SecretStatus:
    missing: list[str] = []
    if settings.allow_live_llm and not settings.live_llm_provider_ready:
        missing.append("OPENAI_API_KEY or GOOGLE_API_KEY")
    return SecretStatus(
        allow_live_llm=settings.allow_live_llm,
        provider_ready=settings.live_llm_provider_ready,
        missing_required_secrets=missing,
    )

