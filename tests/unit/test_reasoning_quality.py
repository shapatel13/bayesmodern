from __future__ import annotations

from agent.orchestrator import PRIORIXOrchestrator
from utils.config import Settings


def test_low_signal_case_flags_clinician_review() -> None:
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_text_case(
        "quality-1",
        "Patient feels unwell and weak.",
    )

    assert report.decision_quality.needs_clinician_review is True
    assert report.decision_quality.low_signal_case is True
    assert report.decision_quality.reasons


def test_completed_tests_are_not_ranked_as_fresh_recommendations() -> None:
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_text_case(
        "quality-2",
        "Crushing chest pain with diaphoresis. ECG showed ischemic changes and troponin 0.12 already resulted.",
    )
    recommendation_map = {recommendation.slug: recommendation for recommendation in report.next_best_tests}

    if "ecg" in recommendation_map:
        assert recommendation_map["ecg"].disposition == "already_answered"
    if "hs_troponin" in recommendation_map:
        assert recommendation_map["hs_troponin"].disposition == "already_answered"
