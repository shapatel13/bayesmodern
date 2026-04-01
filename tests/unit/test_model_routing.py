from llm.model_router import route_models
from security.secrets import validate_live_llm_config
from utils.config import Settings


def test_openai_route_uses_pinned_gpt54nano_snapshot() -> None:
    settings = Settings(
        _env_file=None,
        allow_live_llm=True,
        default_model_provider="openai",
        openai_api_key="test-key",
    )
    routing = route_models(settings)
    assert routing.parser_model == "gpt-5.4-nano-2026-03-17"
    assert routing.reasoning_model == "gpt-5.4-nano-2026-03-17"
    assert routing.verifier_model == "gpt-5.4-nano-2026-03-17"


def test_secret_validation_requires_matching_provider_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        allow_live_llm=True,
        default_model_provider="openai",
        openai_api_key=None,
    )
    status = validate_live_llm_config(settings)
    assert status.provider_ready is False
    assert status.missing_required_secrets == ["OPENAI_API_KEY"]
