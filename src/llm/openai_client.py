from __future__ import annotations

from openai import OpenAI

from utils.config import Settings


def build_openai_client(settings: Settings) -> OpenAI:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for live OpenAI evaluation flows.")
    return OpenAI(api_key=settings.openai_api_key)

