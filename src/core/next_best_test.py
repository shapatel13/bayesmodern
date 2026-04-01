from __future__ import annotations

from core.bayes import bayes_update
from core.models import CandidateTest, ClinicalDecisionContext, TestRecommendation
from core.risk import compute_test_risk_penalty


def _expected_posterior_movement(current_probability: float, lr_plus: float, lr_minus: float) -> float:
    positive_shift = abs(bayes_update(current_probability, lr_plus) - current_probability)
    negative_shift = abs(bayes_update(current_probability, lr_minus) - current_probability)
    return (positive_shift + negative_shift) / 2.0


class NextBestTestEngine:
    def rank(
        self,
        current_differential: dict[str, float],
        candidates: list[CandidateTest],
        context: ClinicalDecisionContext,
    ) -> list[TestRecommendation]:
        recommendations: list[TestRecommendation] = []
        for test in candidates:
            if test.already_done or test.slug in context.completed_tests:
                recommendations.append(
                    TestRecommendation(
                        slug=test.slug,
                        name=test.name,
                        score=0.0,
                        expected_information_gain=0.0,
                        expected_posterior_movement=0.0,
                        stewardship_score=0.0,
                        disposition="already_answered",
                        discriminates_between=test.target_diagnoses,
                    rationale="This test is already documented as completed.",
                    lr_plus=1.0,
                    lr_minus=1.0,
                    provenance_badges=[f"source:{test.source_type.replace('_', '-')}", *test.provenance_refs],
                )
            )
                continue

            movements: list[float] = []
            target_lrs_plus: list[float] = []
            target_lrs_minus: list[float] = []
            for diagnosis in test.target_diagnoses:
                probability = current_differential.get(diagnosis, 0.0)
                lr = test.diagnosis_lrs.get(diagnosis)
                if probability <= 0 or lr is None:
                    continue
                movements.append(_expected_posterior_movement(probability, lr.positive_lr, lr.negative_lr))
                target_lrs_plus.append(lr.positive_lr)
                target_lrs_minus.append(lr.negative_lr)

            info_gain = sum(movements) / max(len(movements), 1)
            average_lr_plus = sum(target_lrs_plus) / max(len(target_lrs_plus), 1)
            average_lr_minus = sum(target_lrs_minus) / max(len(target_lrs_minus), 1)
            risk_penalty = compute_test_risk_penalty(test, context)
            cost_penalty = (test.direct_cost + test.downstream_cost) / 1_000.0
            stewardship_score = max(
                0.0,
                info_gain * test.actionability / (1.0 + (2.0 * cost_penalty) + (1.5 * risk_penalty.total_penalty)),
            )
            score = stewardship_score * test.urgency_modifier

            if score >= 0.1:
                disposition = "worth_it_now"
            elif score >= 0.04:
                disposition = "defer"
            else:
                disposition = "unnecessary"

            recommendations.append(
                TestRecommendation(
                    slug=test.slug,
                    name=test.name,
                    score=score,
                    expected_information_gain=info_gain,
                    expected_posterior_movement=max(movements) if movements else 0.0,
                    stewardship_score=stewardship_score,
                    disposition=disposition,
                    discriminates_between=test.target_diagnoses,
                    rationale=(
                        f"Expected information gain {info_gain:.3f}; stewardship score {stewardship_score:.3f}. "
                        f"Risk penalties: {', '.join(risk_penalty.reasons) if risk_penalty.reasons else 'low.'}"
                    ),
                    lr_plus=average_lr_plus,
                    lr_minus=average_lr_minus,
                    provenance_badges=[f"source:{test.source_type.replace('_', '-')}", *test.provenance_refs],
                )
            )

        recommendations.sort(key=lambda item: item.score, reverse=True)
        return recommendations
