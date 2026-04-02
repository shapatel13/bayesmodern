from __future__ import annotations

from agent.orchestrator import PRIORIXOrchestrator
from utils.config import Settings


def test_mechanism_layer_detects_mixed_congestion_and_impaired_contractility() -> None:
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_text_case(
        "mechanism-1",
        (
            "Hypotension with orthopnea, crackles, leg edema, elevated JVP, diaphoresis, "
            "substernal chest pressure, and low ejection fraction."
        ),
    )

    top_mechanisms = [estimate.slug for estimate in report.mechanism_states.ranked[:4]]
    top_tests = [recommendation.slug for recommendation in report.next_best_tests[:4]]

    assert "venous_congestion" in top_mechanisms
    assert "impaired_contractility" in top_mechanisms
    assert report.mechanism_states.mixed_physiology is True
    assert "bedside_echo" in top_tests


def test_mechanism_layer_detects_bleeding_low_preload_and_medication_effect() -> None:
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_text_case(
        "mechanism-2",
        "Patient on warfarin with melena, symptomatic anemia, hypotension, and oliguria after recent dose escalation.",
    )

    top_mechanisms = [estimate.slug for estimate in report.mechanism_states.ranked[:4]]
    top_tests = [recommendation.slug for recommendation in report.next_best_tests[:4]]

    assert "hemorrhagic_tendency_active_blood_loss" in top_mechanisms
    assert "low_effective_arterial_volume" in top_mechanisms
    assert "medication_toxicity_effect" in top_mechanisms
    assert "repeat_hemoglobin" in top_tests
