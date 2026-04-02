from __future__ import annotations

from core.bayes import bayes_update, shannon_entropy
from core.models import CandidateTest, ClinicalDecisionContext, MechanismStateResult, TestRecommendation
from core.policy import ReasoningPolicy, get_reasoning_policy
from core.thresholds import DecisionThresholds
from core.risk import compute_test_risk_penalty


def _contextual_test_multiplier(test: CandidateTest, context: ClinicalDecisionContext) -> float:
    present_keys = {finding.key for finding in context.findings if finding.present}
    acs_supportive_context = bool(
        present_keys & {"pressure_chest_pain", "pain_radiation", "diaphoresis", "troponin_positive"}
    ) or context.hemodynamic_instability
    bleeding_supportive_context = bool(
        present_keys & {"anticoagulated", "active_gi_bleeding", "melena", "symptomatic_anemia"}
    ) or bool(
        {event.strip().lower() for event in context.adverse_events} & {"gi bleed", "melena", "symptomatic anemia"}
    ) or bool(
        {medication.strip().lower() for medication in context.medications}
        & {"warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran", "enoxaparin"}
    )
    if "acs" in test.target_diagnoses:
        if bleeding_supportive_context and not acs_supportive_context:
            return 0.35
        return 1.35 if acs_supportive_context else 0.7
    if "upper_gi_bleed" in test.target_diagnoses:
        return 1.45 if bleeding_supportive_context else 0.8
    if bleeding_supportive_context and set(test.target_diagnoses) & {"pe", "pneumonia", "heart_failure", "acs"}:
        return 0.45
    return 1.0


def _expected_posterior_movement(current_probability: float, lr_plus: float, lr_minus: float) -> float:
    positive_shift = abs(bayes_update(current_probability, lr_plus) - current_probability)
    negative_shift = abs(bayes_update(current_probability, lr_minus) - current_probability)
    return (positive_shift + negative_shift) / 2.0


def _entropy_gain(current_probability: float, lr_plus: float, lr_minus: float) -> float:
    current_entropy = shannon_entropy({"target": current_probability, "other": max(1.0 - current_probability, 1e-6)})
    positive_entropy = shannon_entropy(
        {"target": bayes_update(current_probability, lr_plus), "other": max(1.0 - bayes_update(current_probability, lr_plus), 1e-6)}
    )
    negative_entropy = shannon_entropy(
        {"target": bayes_update(current_probability, lr_minus), "other": max(1.0 - bayes_update(current_probability, lr_minus), 1e-6)}
    )
    return max(0.0, current_entropy - ((positive_entropy + negative_entropy) / 2.0))


def _threshold_crossing_gain(
    current_probability: float,
    lr_plus: float,
    lr_minus: float,
    thresholds: DecisionThresholds | None,
) -> float:
    if thresholds is None:
        return 0.0
    positive_probability = bayes_update(current_probability, lr_plus)
    negative_probability = bayes_update(current_probability, lr_minus)
    thresholds_to_check = (
        thresholds.test_threshold,
        thresholds.admit_threshold,
        thresholds.treatment_threshold,
        thresholds.icu_threshold,
    )
    gain = 0.0
    for threshold in thresholds_to_check:
        current_side = current_probability >= threshold
        if (positive_probability >= threshold) != current_side:
            gain += 0.5
        if (negative_probability >= threshold) != current_side:
            gain += 0.5
    return gain


def _discrimination_gain(current_differential: dict[str, float], test: CandidateTest) -> float:
    if len(test.target_diagnoses) < 2:
        return 0.0
    ranked_targets = sorted(
        ((diagnosis, current_differential.get(diagnosis, 0.0)) for diagnosis in test.target_diagnoses),
        key=lambda item: item[1],
        reverse=True,
    )
    top_targets = ranked_targets[:2]
    if len(top_targets) < 2:
        return 0.0
    first_lr = test.diagnosis_lrs.get(top_targets[0][0])
    second_lr = test.diagnosis_lrs.get(top_targets[1][0])
    if first_lr is None or second_lr is None:
        return 0.0
    return abs(first_lr.positive_lr - second_lr.positive_lr) + abs(first_lr.negative_lr - second_lr.negative_lr)


class NextBestTestEngine:
    def rank(
        self,
        current_differential: dict[str, float],
        candidates: list[CandidateTest],
        context: ClinicalDecisionContext,
        mechanism_result: MechanismStateResult | None = None,
        thresholds: DecisionThresholds | None = None,
        policy: ReasoningPolicy | str | None = None,
    ) -> list[TestRecommendation]:
        resolved_policy = get_reasoning_policy(policy) if isinstance(policy, str) or policy is None else policy
        recommendations: list[TestRecommendation] = []
        top_diagnosis_slug = max(current_differential, key=current_differential.get) if current_differential else None
        top_diagnosis_posterior = current_differential.get(top_diagnosis_slug, 0.0) if top_diagnosis_slug else 0.0
        mechanism_map = {
            estimate.slug: estimate.posterior
            for estimate in (mechanism_result.ranked if mechanism_result is not None else [])
        }
        for test in candidates:
            note_prefix = f"{test.evidence_note} " if test.evidence_note else ""
            mechanism_prefix = f"{test.mechanistic_note} " if test.mechanistic_note else ""
            if test.already_done or test.slug in context.completed_tests:
                recommendations.append(
                    TestRecommendation(
                        slug=test.slug,
                        name=test.name,
                        score=0.0,
                        expected_information_gain=0.0,
                        expected_posterior_movement=0.0,
                        mechanistic_information_gain=0.0,
                        stewardship_score=0.0,
                        disposition="already_answered",
                        discriminates_between=test.target_diagnoses,
                        target_states=test.target_states,
                        rationale=f"{note_prefix}This test is already documented as completed.".strip(),
                        lr_plus=1.0,
                        lr_minus=1.0,
                        direct_cost=test.direct_cost,
                        downstream_cost=test.downstream_cost,
                        risk_penalty=1.0,
                        provenance_badges=[f"source:{test.source_type.replace('_', '-')}", *test.provenance_refs],
                    )
                )
                continue

            movements: list[float] = []
            entropy_gains: list[float] = []
            threshold_gains: list[float] = []
            target_lrs_plus: list[float] = []
            target_lrs_minus: list[float] = []
            mechanistic_movements: list[float] = []
            mechanistic_entropy_gains: list[float] = []
            mechanistic_uncertainty_signals: list[float] = []
            for diagnosis in test.target_diagnoses:
                probability = current_differential.get(diagnosis, 0.0)
                lr = test.diagnosis_lrs.get(diagnosis)
                if probability <= 0 or lr is None:
                    continue
                movements.append(_expected_posterior_movement(probability, lr.positive_lr, lr.negative_lr))
                entropy_gains.append(_entropy_gain(probability, lr.positive_lr, lr.negative_lr))
                threshold_gains.append(_threshold_crossing_gain(probability, lr.positive_lr, lr.negative_lr, thresholds))
                target_lrs_plus.append(lr.positive_lr)
                target_lrs_minus.append(lr.negative_lr)
            for state_slug in test.target_states:
                probability = mechanism_map.get(state_slug, 0.0)
                lr = test.state_lrs.get(state_slug)
                if probability <= 0 or lr is None:
                    continue
                mechanistic_movements.append(_expected_posterior_movement(probability, lr.positive_lr, lr.negative_lr))
                mechanistic_entropy_gains.append(_entropy_gain(probability, lr.positive_lr, lr.negative_lr))
                mechanistic_uncertainty_signals.append(1.0 - min(abs(probability - 0.5) / 0.5, 1.0))

            info_gain = sum(movements) / max(len(movements), 1)
            entropy_gain = sum(entropy_gains) / max(len(entropy_gains), 1)
            threshold_gain = sum(threshold_gains) / max(len(threshold_gains), 1)
            mechanistic_info_gain = (
                (sum(mechanistic_movements) / max(len(mechanistic_movements), 1))
                + (sum(mechanistic_entropy_gains) / max(len(mechanistic_entropy_gains), 1))
            ) / 2.0
            discrimination_gain = _discrimination_gain(current_differential, test)
            average_lr_plus = sum(target_lrs_plus) / max(len(target_lrs_plus), 1)
            average_lr_minus = sum(target_lrs_minus) / max(len(target_lrs_minus), 1)
            risk_penalty = compute_test_risk_penalty(test, context)
            cost_penalty = (test.direct_cost + test.downstream_cost) / 1_000.0
            evidence_value = (
                (info_gain * resolved_policy.next_test.movement_weight)
                + (entropy_gain * resolved_policy.next_test.entropy_weight)
                + (discrimination_gain * 0.05 * resolved_policy.next_test.discrimination_weight)
                + (threshold_gain * 0.1 * resolved_policy.next_test.threshold_weight)
                + (mechanistic_info_gain * 0.85)
            )
            urgency_bonus = 1.0 + (
                resolved_policy.next_test.urgency_bonus_weight
                if context.hemodynamic_instability or context.critical_values_present
                else 0.0
            )
            diagnosis_focus_multiplier = 1.0
            if (
                top_diagnosis_slug
                and top_diagnosis_slug in test.target_diagnoses
                and top_diagnosis_posterior >= 0.3
                and test.source_type == "llm_inferred"
            ):
                diagnosis_focus_multiplier = 1.4
            target_diagnosis_posterior = max(
                (current_differential.get(diagnosis, 0.0) for diagnosis in test.target_diagnoses),
                default=0.0,
            )
            diagnosis_alignment_multiplier = 0.7 + min(target_diagnosis_posterior, 0.6)
            mechanism_alignment_multiplier = 1.0
            if mechanistic_uncertainty_signals:
                mechanism_alignment_multiplier += 0.45 * (
                    sum(mechanistic_uncertainty_signals) / len(mechanistic_uncertainty_signals)
                )
            stewardship_score = max(
                0.0,
                (
                    evidence_value
                    * max(test.actionability, 0.1) ** resolved_policy.next_test.actionability_weight
                    * urgency_bonus
                    * diagnosis_focus_multiplier
                    * diagnosis_alignment_multiplier
                    * mechanism_alignment_multiplier
                    * _contextual_test_multiplier(test, context)
                )
                / (
                    1.0
                    + (resolved_policy.next_test.cost_weight * cost_penalty)
                    + (resolved_policy.next_test.risk_weight * risk_penalty.total_penalty)
                ),
            )
            score = stewardship_score * test.urgency_modifier

            if score >= resolved_policy.next_test.worth_it_threshold:
                disposition = "worth_it_now"
            elif score >= resolved_policy.next_test.defer_threshold:
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
                    mechanistic_information_gain=mechanistic_info_gain,
                    stewardship_score=stewardship_score,
                    disposition=disposition,
                    discriminates_between=test.target_diagnoses,
                    target_states=test.target_states,
                    rationale=(
                        f"{note_prefix}{mechanism_prefix}Expected movement {info_gain:.3f}; entropy gain {entropy_gain:.3f}; "
                        f"mechanistic gain {mechanistic_info_gain:.3f}; threshold gain {threshold_gain:.3f}; discrimination gain {discrimination_gain:.3f}. "
                        f"stewardship score {stewardship_score:.3f}. "
                        f"Risk penalties: {', '.join(risk_penalty.reasons) if risk_penalty.reasons else 'low.'}"
                    ).strip(),
                    lr_plus=average_lr_plus,
                    lr_minus=average_lr_minus,
                    direct_cost=test.direct_cost,
                    downstream_cost=test.downstream_cost,
                    risk_penalty=risk_penalty.total_penalty,
                    provenance_badges=[f"source:{test.source_type.replace('_', '-')}", *test.provenance_refs],
                )
            )

        recommendations.sort(key=lambda item: item.score, reverse=True)
        return recommendations
