from __future__ import annotations

from core.models import ClinicalDecisionContext
from llm.openai_case_review import AIReviewerFeedback, review_case_with_openai
from llm.structured_output import ModelRoutingDecision, ResearchReport
from utils.config import Settings


def _sample_report() -> ResearchReport:
    return ResearchReport(
        context=ClinicalDecisionContext(
            case_id="review-1",
            specialty="general_internal_medicine",
            findings=[],
            completed_tests=["pt_inr"],
            medications=["apixaban"],
            adverse_events=[],
            symptoms_free_text="Melena and dyspnea.",
            age_years=72,
            pregnant=False,
            renal_impairment=True,
            hemodynamic_instability=True,
            critical_values_present=True,
            safety_mode="conservative",
        ),
        differential={
            "ranked": [
                {
                    "slug": "upper_gi_bleed",
                    "name": "Upper Gastrointestinal Bleeding / Anticoagulant-Associated Hemorrhage",
                    "prior": 0.25,
                    "posterior": 0.62,
                    "interval_low": 0.54,
                    "interval_high": 0.7,
                    "evidence_for": [],
                    "evidence_against": [],
                    "symptom_coverage": 0.6,
                    "explaining_away": [],
                    "calibration_state": "moderately_uncertain",
                    "provenance_badges": ["source:hard-coded"],
                },
                {
                    "slug": "heart_failure",
                    "name": "Acute Decompensated Heart Failure",
                    "prior": 0.2,
                    "posterior": 0.28,
                    "interval_low": 0.22,
                    "interval_high": 0.34,
                    "evidence_for": [],
                    "evidence_against": [],
                    "symptom_coverage": 0.5,
                    "explaining_away": [],
                    "calibration_state": "fragile",
                    "provenance_badges": ["source:hard-coded"],
                },
            ],
            "posterior_mass_top3": 0.9,
            "model_note": "demo",
        },
        mechanism_states={
            "ranked": [
                {
                    "slug": "impaired_contractility",
                    "name": "Impaired Contractility / Cardiogenic Physiology",
                    "category": "cardiovascular",
                    "prior": 0.14,
                    "posterior": 0.73,
                    "interval_low": 0.64,
                    "interval_high": 0.8,
                    "evidence_for": [],
                    "evidence_against": [],
                    "confidence_state": "moderately_uncertain",
                    "provenance_badges": ["source:hard-coded"],
                },
                {
                    "slug": "hemorrhagic_tendency_active_blood_loss",
                    "name": "Hemorrhagic Tendency / Active Blood Loss",
                    "category": "hematology",
                    "prior": 0.12,
                    "posterior": 0.68,
                    "interval_low": 0.6,
                    "interval_high": 0.75,
                    "evidence_for": [],
                    "evidence_against": [],
                    "confidence_state": "moderately_uncertain",
                    "provenance_badges": ["source:hard-coded"],
                },
            ],
            "active_states": [
                "Impaired Contractility / Cardiogenic Physiology",
                "Hemorrhagic Tendency / Active Blood Loss",
            ],
            "mixed_physiology": True,
            "summary": "Mixed physiology signal: Impaired Contractility / Cardiogenic Physiology plus Hemorrhagic Tendency / Active Blood Loss.",
            "model_note": "demo",
        },
        next_best_tests=[
            {
                "slug": "pt_inr",
                "name": "PT/INR",
                "score": 0.41,
                "expected_information_gain": 0.1,
                "expected_posterior_movement": 0.17,
                "mechanistic_information_gain": 0.08,
                "stewardship_score": 0.78,
                "disposition": "worth_it_now",
                "discriminates_between": ["upper_gi_bleed"],
                "target_states": ["hemorrhagic_tendency_active_blood_loss", "medication_toxicity_effect"],
                "rationale": "demo",
                "lr_plus": 2.1,
                "lr_minus": 0.82,
                "direct_cost": 18.0,
                "downstream_cost": 60.0,
                "risk_penalty": 0.0,
                "provenance_badges": ["source:hard-coded"],
            }
        ],
        triage={
            "urgency": "urgent",
            "reasons": ["demo"],
            "admit_threshold_crossed": True,
            "icu_threshold_crossed": False,
        },
        threshold_decision={
            "action": "test",
            "clinician_language": "demo",
            "plain_language": "demo",
        },
        reasoning_runtime={
            "mode": "curated_only",
            "open_world_considered": False,
            "open_world_triggered": False,
            "gate_reason": "Curated path stayed active.",
            "notes": [],
        },
        decision_quality={
            "needs_clinician_review": True,
            "reasons": ["Broad differential remained after scoring."],
            "structured_signal_count": 8,
            "top_differential_gap": 0.1,
            "low_signal_case": False,
            "broad_differential": True,
            "mixed_mechanism_uncertainty": True,
        },
        contradictions=[],
        provenance_warnings=[],
        model_route=ModelRoutingDecision(
            parser_model="gpt-5.4-nano-2026-03-17",
            reasoning_model="gpt-5.4-nano-2026-03-17",
            verifier_model="gpt-5.4-nano-2026-03-17",
            mode="offline",
        ),
    )


def test_openai_case_review_normalizes_review_output(monkeypatch) -> None:
    class _Parsed:
        def __init__(self, payload: AIReviewerFeedback) -> None:
            self.output_parsed = payload

    class _Responses:
        def parse(self, **kwargs):
            assert kwargs["model"] == "gpt-5.4"
            return _Parsed(
                AIReviewerFeedback(
                    summary="Mechanism mix is more important than a single disease label.",
                    gold_diagnosis="Acute Decompensated Heart Failure",
                    reviewed_mechanism_states=[
                        "Impaired Contractility / Cardiogenic Physiology",
                        "Venous Congestion",
                        "Fluid Intolerance",
                    ],
                    reviewed_contributing_processes=["active blood loss", "NSAID effect"],
                    acceptable_tests=["Repeat hemoglobin trend", "Bedside echocardiography"],
                    gold_triage="emergent",
                    review_tags=["bad_mechanism_inference", "bad_next_test", "not_a_real_tag"],
                    preferred_next_action="Repeat hemoglobin and bedside volume-status clarification.",
                    mechanism_feedback_summary="Cardiogenic and hemorrhagic mechanisms are both active.",
                    review_notes="The current next action is too narrow.",
                    review_status_recommendation="approved",
                    reviewer_confidence=0.91,
                )
            )

    class _Client:
        responses = _Responses()

    monkeypatch.setattr("llm.openai_case_review.build_openai_client", lambda settings: _Client())
    feedback = review_case_with_openai(
        "72-year-old with melena, dyspnea, edema, and shock physiology.",
        _sample_report(),
        Settings(
            _env_file=None,
            allow_live_llm=True,
            openai_api_key="test-key",
            openai_case_review_model="gpt-5.4",
        ),
    )

    assert feedback.gold_diagnosis == "heart_failure"
    assert feedback.reviewed_mechanism_states == [
        "impaired_contractility",
        "venous_congestion",
        "fluid_intolerance",
    ]
    assert feedback.acceptable_tests == ["repeat_hemoglobin", "bedside_echo"]
    assert feedback.review_tags == ["bad_mechanism_inference", "bad_next_test"]
