from core.models import CandidateTest, ClinicalDecisionContext, LikelihoodRatioRange
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

