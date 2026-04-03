from __future__ import annotations

from agent.orchestrator import PRIORIXOrchestrator
from utils.config import Settings


def test_anticoagulated_melena_case_ranks_upper_gi_bleed_first() -> None:
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_text_case(
        "review-bleed-1",
        "Patient on warfarin with melena and symptomatic anemia after recent dose escalation.",
    )

    top_entry = report.differential.ranked[0]
    assert top_entry.slug == "upper_gi_bleed"
    assert top_entry.interval_low <= top_entry.posterior <= top_entry.interval_high
    assert report.triage.urgency in {"urgent", "emergent"}


def test_anticoagulated_melena_case_prefers_bleeding_workup_over_pe_rule_out() -> None:
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_text_case(
        "review-bleed-2",
        "Patient on warfarin with melena and symptomatic anemia after recent dose escalation.",
    )
    top_tests = [recommendation.slug for recommendation in report.next_best_tests[:4]]

    assert top_tests[0] == "repeat_hemoglobin"
    assert "pt_inr" in top_tests[:3]
    assert "type_screen" in top_tests
    assert "d_dimer" not in top_tests
