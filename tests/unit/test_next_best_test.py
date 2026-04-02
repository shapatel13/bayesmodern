from core.models import CandidateTest, ClinicalDecisionContext, LikelihoodRatioRange, MechanismStateEstimate, MechanismStateResult
from core.next_best_test import NextBestTestEngine


def test_next_best_test_prefers_discriminative_low_risk_option() -> None:
    engine = NextBestTestEngine()
    context = ClinicalDecisionContext(case_id="case-1")
    candidates = [
        CandidateTest(
            slug="d_dimer",
            name="D-dimer",
            target_diagnoses=["pe"],
            diagnosis_lrs={"pe": LikelihoodRatioRange(positive_lr=2.2, negative_lr=0.2)},
            direct_cost=40,
            downstream_cost=0,
            actionability=0.7,
        ),
        CandidateTest(
            slug="cta",
            name="CT pulmonary angiography",
            target_diagnoses=["pe"],
            diagnosis_lrs={"pe": LikelihoodRatioRange(positive_lr=8.0, negative_lr=0.1)},
            direct_cost=850,
            downstream_cost=300,
            radiation=0.6,
            actionability=0.9,
        ),
    ]
    ranked = engine.rank({"pe": 0.25}, candidates, context)
    assert ranked[0].slug == "d_dimer"


def test_next_best_test_can_prefer_mechanistic_clarifier_when_state_uncertainty_is_high() -> None:
    engine = NextBestTestEngine()
    context = ClinicalDecisionContext(case_id="case-2", hemodynamic_instability=True)
    mechanism_result = MechanismStateResult(
        ranked=[
            MechanismStateEstimate(
                slug="fluid_responsiveness",
                name="Fluid Responsiveness",
                category="hemodynamics",
                prior=0.15,
                posterior=0.52,
                interval_low=0.34,
                interval_high=0.68,
                evidence_for=[],
                evidence_against=[],
                confidence_state="fragile",
                provenance_badges=["source:hard-coded"],
            ),
            MechanismStateEstimate(
                slug="fluid_intolerance",
                name="Fluid Intolerance",
                category="hemodynamics",
                prior=0.16,
                posterior=0.49,
                interval_low=0.31,
                interval_high=0.66,
                evidence_for=[],
                evidence_against=[],
                confidence_state="fragile",
                provenance_badges=["source:hard-coded"],
            ),
        ],
        active_states=[],
        mixed_physiology=False,
        summary="Mechanism layer remains broad.",
        model_note="test fixture",
    )
    candidates = [
        CandidateTest(
            slug="cxr",
            name="Chest radiograph",
            target_diagnoses=["heart_failure", "pneumonia"],
            diagnosis_lrs={
                "heart_failure": LikelihoodRatioRange(positive_lr=1.45, negative_lr=0.82),
                "pneumonia": LikelihoodRatioRange(positive_lr=1.6, negative_lr=0.78),
            },
            direct_cost=180,
            downstream_cost=40,
            actionability=0.55,
        ),
        CandidateTest(
            slug="plr_lvot_vti",
            name="PLR with LVOT VTI assessment",
            target_diagnoses=[],
            diagnosis_lrs={},
            target_states=["fluid_responsiveness", "fluid_intolerance"],
            state_lrs={
                "fluid_responsiveness": LikelihoodRatioRange(positive_lr=3.0, negative_lr=0.58),
                "fluid_intolerance": LikelihoodRatioRange(positive_lr=0.58, negative_lr=1.45),
            },
            direct_cost=85,
            downstream_cost=20,
            actionability=0.95,
            urgency_modifier=1.25,
            bedside=True,
            mechanistic_note="Clarifies whether preload augmentation is likely to help or harm.",
        ),
    ]

    ranked = engine.rank(
        {"heart_failure": 0.27, "pneumonia": 0.25},
        candidates,
        context,
        mechanism_result=mechanism_result,
    )

    assert ranked[0].slug == "plr_lvot_vti"
    assert ranked[0].mechanistic_information_gain > 0.1
