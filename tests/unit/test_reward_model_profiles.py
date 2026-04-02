from __future__ import annotations

from agent.reward_model import CompositeRewardModel
from llm.structured_output import GenerationAuditResult, ModelRoutingDecision, ResearchReport
from priorix_tasks.common import BenchmarkTask


def _report(*, urgency: str = "urgent", medications: list[str] | None = None, adverse_events: list[str] | None = None) -> ResearchReport:
    return ResearchReport(
        context={
            "case_id": "case-1",
            "specialty": "general_internal_medicine",
            "findings": [],
            "completed_tests": [],
            "comorbidities": [],
            "medications": medications or [],
            "adverse_events": adverse_events or [],
            "symptoms_free_text": "demo",
            "age_years": None,
            "pregnant": False,
            "renal_impairment": False,
            "hemodynamic_instability": urgency == "emergent",
            "critical_values_present": urgency == "emergent",
            "safety_mode": "conservative",
        },
        differential={
            "ranked": [
                {
                    "slug": "pe",
                    "name": "Pulmonary Embolism",
                    "prior": 0.2,
                    "posterior": 0.72,
                    "interval_low": 0.6,
                    "interval_high": 0.8,
                    "evidence_for": [],
                    "evidence_against": [],
                    "symptom_coverage": 0.5,
                    "explaining_away": [],
                    "calibration_state": "moderately_uncertain",
                    "provenance_badges": ["source:hard-coded"],
                }
            ],
            "posterior_mass_top3": 0.72,
            "model_note": "demo",
        },
        next_best_tests=[],
        triage={
            "urgency": urgency,
            "reasons": ["demo"],
            "admit_threshold_crossed": urgency in {"urgent", "emergent"},
            "icu_threshold_crossed": urgency == "emergent",
        },
        threshold_decision={
            "action": "observe" if urgency == "emergent" else "test",
            "clinician_language": "demo",
            "plain_language": "demo",
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


def test_diagnosis_reward_profile_marks_correct_case() -> None:
    reward = CompositeRewardModel().score(
        BenchmarkTask(
            task_id="diag-1",
            source_dataset="synthetic",
            split="test",
            task_type="diagnosis_open",
            prompt="demo",
            gold_diagnosis="pe",
            acceptable_tests=[],
            gold_triage="urgent",
        ),
        _report(urgency="urgent"),
    )

    assert reward.reward_profile == "diagnostic"
    assert reward.total_reward > 0.8
    assert reward.hard_veto is False


def test_triage_reward_profile_vetoes_missed_emergent_case() -> None:
    reward = CompositeRewardModel().score(
        BenchmarkTask(
            task_id="triage-1",
            source_dataset="mietic",
            split="test",
            task_type="triage",
            prompt="demo",
            gold_triage="emergent",
        ),
        _report(urgency="urgent"),
    )

    assert reward.reward_profile == "triage"
    assert reward.hard_veto is True
    assert "unsafe_recommendation" in reward.failure_categories


def test_medication_safety_reward_tracks_extraction_quality() -> None:
    reward = CompositeRewardModel().score(
        BenchmarkTask(
            task_id="med-1",
            source_dataset="n2c2_2018_track2",
            split="test",
            task_type="medication_safety",
            prompt="demo",
            metadata={"gold_medications": ["lisinopril"], "gold_adverse_events": ["angioedema"]},
        ),
        _report(medications=["lisinopril"], adverse_events=["angioedema"]),
    )

    assert reward.reward_profile == "medication_safety"
    assert reward.medication_extraction_quality == 1.0
    assert reward.adverse_event_quality == 1.0
    assert reward.hard_veto is False


def test_medication_safety_reward_vetoes_missed_adverse_event() -> None:
    reward = CompositeRewardModel().score(
        BenchmarkTask(
            task_id="med-2",
            source_dataset="n2c2_2018_track2",
            split="test",
            task_type="medication_safety",
            prompt="demo",
            metadata={"gold_medications": ["warfarin"], "gold_adverse_events": ["gi bleed"]},
        ),
        _report(medications=["warfarin"], adverse_events=[]),
    )

    assert reward.hard_veto is True
    assert "medication_safety_failure" in reward.failure_categories


def test_generation_audit_reward_vetoes_undercalled_high_risk_output() -> None:
    report = _report(urgency="routine")
    report.generation_audit = GenerationAuditResult(
        predicted_risk_grade=1,
        recommended_action="accept",
        issue_types=[],
        rationale="No issues detected.",
        confidence=0.6,
        reference_available=True,
        physician_reference_available=True,
    )
    reward = CompositeRewardModel().score(
        BenchmarkTask(
            task_id="audit-1",
            source_dataset="medval_bench",
            split="test",
            task_type="generation_audit",
            prompt="audit prompt",
            gold_risk_grade=4,
            metadata={"physician_risk_grade": 4},
        ),
        report,
    )

    assert reward.reward_profile == "generation_audit"
    assert reward.hard_veto is True
    assert "generation_audit_failure" in reward.failure_categories
