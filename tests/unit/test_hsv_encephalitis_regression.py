from __future__ import annotations

from agent.orchestrator import PRIORIXOrchestrator
from utils.config import Settings


HSV_GOLD_CASE = (
    "8-year-old woman presents with altered mental status. Brain MRI shows unilateral mesial temporal lobe "
    "FLAIR hyperintensity/enhancement. She has acute kidney injury with creatinine rising during empiric IV "
    "acyclovir. LP #1, obtained more than 72 hours after symptom onset, shows no pleocytosis and negative "
    "HSV-1/2 PCR. EEG shows diffuse slowing without periodic lateralized epileptiform discharges or other "
    "classic temporal epileptiform features. Despite continued concern because of the MRI pattern, LP #2 is "
    "repeated 3 to 7 days later and again shows no pleocytosis and negative HSV PCR. No alternative data "
    "strongly support HSV, and no progressive EEG or CSF inflammatory signal emerges."
)


def test_hsv_gold_case_does_not_collapse_into_cardiopulmonary_default_reasoning() -> None:
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_text_case("hsv-gold-regression-1", HSV_GOLD_CASE)
    ranked_slugs = [entry.slug for entry in report.differential.ranked[:4]]
    hsv_entry = next(entry for entry in report.differential.ranked if entry.slug == "hsv_encephalitis")
    non_hsv_entry = next(entry for entry in report.differential.ranked if entry.slug == "non_hsv_temporal_encephalitis")

    assert "non_hsv_temporal_encephalitis" in ranked_slugs[:2]
    assert hsv_entry.posterior <= 0.08
    assert non_hsv_entry.posterior > hsv_entry.posterior
    assert ranked_slugs[0] not in {"pneumonia", "heart_failure", "pe", "acs"}


def test_hsv_gold_case_surfaces_repeat_pcr_dominance_and_renal_threshold_language() -> None:
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_text_case("hsv-gold-regression-2", HSV_GOLD_CASE)
    hsv_entry = next(entry for entry in report.differential.ranked if entry.slug == "hsv_encephalitis")
    evidence_against_keys = {item.finding_key for item in hsv_entry.evidence_against}

    assert {"hsv_pcr_negative_timed", "repeat_hsv_pcr_negative_dependent"} <= evidence_against_keys
    assert report.threshold_decision.action in {"observe", "test"}
    threshold_text = f"{report.threshold_decision.clinician_language} {report.threshold_decision.plain_language}".lower()
    assert "acyclovir" in threshold_text
    assert "negative csf hsv pcr" in threshold_text or "negative spinal-fluid hsv pcr" in threshold_text
    assert "moderate positive support" in report.differential.model_note.lower()
    assert any("moderate positive support" in note.lower() for note in report.reasoning_runtime.special_reasoning_notes)
    assert any("partially dependent" in note.lower() for note in report.reasoning_runtime.test_dependency_notes)


def test_hsv_gold_case_prefers_neuroinfectious_reassessment_over_cardiopulmonary_testing() -> None:
    orchestrator = PRIORIXOrchestrator(
        settings=Settings(_env_file=None, allow_live_llm=False, default_model_provider="offline")
    )

    report = orchestrator.analyze_text_case("hsv-gold-regression-3", HSV_GOLD_CASE)
    top_tests = {recommendation.slug: recommendation for recommendation in report.next_best_tests[:4]}

    assert "csf_autoimmune_panel" in top_tests
    assert report.next_best_tests[0].slug == "csf_autoimmune_panel"
    assert "d_dimer" not in top_tests
    if "bnp" in top_tests:
        assert top_tests["bnp"].disposition == "unnecessary"
