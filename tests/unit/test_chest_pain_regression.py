from __future__ import annotations

from agent.orchestrator import PRIORIXOrchestrator
from utils.config import Settings


def test_crushing_chest_pain_hypotension_ranks_acs_first_offline() -> None:
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_text_case(
        "review-acs-1",
        "Crushing chest pain, hypotension, and diaphoresis.",
    )

    assert report.differential.ranked[0].slug == "acs"
    assert report.triage.urgency == "emergent"


def test_crushing_chest_pain_recommends_acs_workup_first() -> None:
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_text_case(
        "review-acs-2",
        "Crushing substernal chest pain radiating to the jaw with diaphoresis.",
    )
    top_tests = [recommendation.slug for recommendation in report.next_best_tests[:3]]

    assert "ecg" in top_tests
    assert "hs_troponin" in top_tests
