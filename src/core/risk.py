from __future__ import annotations

from core.models import CandidateTest, ClinicalDecisionContext, TestRiskPenalty


def compute_test_risk_penalty(test: CandidateTest, context: ClinicalDecisionContext) -> TestRiskPenalty:
    penalty = test.invasiveness + test.radiation + test.logistic_burden
    reasons: list[str] = []

    if test.invasiveness > 0.4:
        reasons.append("Invasive procedure burden")
    if test.radiation > 0.3:
        reasons.append("Radiation exposure")
    if context.renal_impairment and test.nephrotoxicity > 0.1:
        penalty += 0.4
        reasons.append("Renal risk amplified by kidney impairment")
    if test.bleed_risk > 0.2:
        penalty += 0.25
        reasons.append("Bleeding risk")
    if test.already_done or test.slug in context.completed_tests:
        penalty += 1.0
        reasons.append("Already completed")

    return TestRiskPenalty(total_penalty=penalty, reasons=reasons)

