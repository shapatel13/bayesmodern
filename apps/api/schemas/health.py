from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    safety_mode: str
    allow_live_llm: bool
    provider_ready: bool
    default_model_provider: str
    openai_parser_model: str
    openai_reasoning_model: str
    openai_verifier_model: str
    missing_required_secrets: list[str]
