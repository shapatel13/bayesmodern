from __future__ import annotations

from llm.structured_output import ModelRoutingDecision
from utils.config import Settings


def route_models(settings: Settings, complexity_hint: str = "standard") -> ModelRoutingDecision:
    if not settings.allow_live_llm:
        return ModelRoutingDecision(
            parser_model="offline-keyword-parser",
            reasoning_model="deterministic-bayes-engine",
            verifier_model="offline-contradiction-checker",
            mode="offline",
        )
    return ModelRoutingDecision(
        parser_model=f"{settings.default_model_provider}-fast-parser",
        reasoning_model=f"{settings.default_model_provider}-{complexity_hint}-reasoner",
        verifier_model=f"{settings.default_model_provider}-verifier",
        mode="offline-eval-with-external-model",
    )

