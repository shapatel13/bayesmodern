from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from core.bayes import bayes_update, normalize_distribution, shannon_entropy
from core.calibration import classify_calibration
from core.likelihood_ratios import update_probability_from_evidence
from core.models import (
    ClinicalDecisionContext,
    DiagnosisHypothesis,
    DifferentialEntry,
    DifferentialResult,
    ProbabilityInterval,
)
from core.policy import ReasoningPolicy, get_reasoning_policy
from core.uncertainty import normalize_sampled_distributions, sample_lr, sample_probability


def _symptom_coverage(context: ClinicalDecisionContext, hypothesis: DiagnosisHypothesis) -> float:
    present_keys = {finding.key for finding in context.findings if finding.present}
    if not present_keys:
        return 0.0
    supported = {evidence.finding_key for evidence in hypothesis.supporting_findings}
    return len(present_keys & supported) / len(present_keys)


def _apply_explaining_away(
    raw_posteriors: dict[str, float],
    coverages: Mapping[str, float],
    policy: ReasoningPolicy,
) -> dict[str, float]:
    ranked = sorted(raw_posteriors.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2:
        return raw_posteriors
    top_slug, top_posterior = ranked[0]
    adjusted = raw_posteriors.copy()
    for slug, posterior in ranked[1:]:
        overlap = min(coverages.get(top_slug, 0.0), coverages.get(slug, 0.0))
        adjustment = max(0.7, 1.0 - (top_posterior * overlap * policy.differential.overlap_penalty_strength))
        adjusted[slug] = posterior * adjustment
    adjusted[top_slug] = max(adjusted[top_slug], top_posterior)
    return normalize_distribution(adjusted)


def _contradiction_load(contributions: list) -> float:
    return sum(max(0.0, 1.0 - item.applied_lr) for item in contributions if item.direction == "against")


def _contextual_prior_multiplier(
    context: ClinicalDecisionContext,
    hypothesis: DiagnosisHypothesis,
    policy: ReasoningPolicy,
) -> float:
    multiplier = 1.0
    present_keys = {finding.key for finding in context.findings if finding.present}
    adverse_events = {event.strip().lower() for event in context.adverse_events}
    medications = {medication.strip().lower() for medication in context.medications}
    if hypothesis.dangerous and (context.hemodynamic_instability or context.critical_values_present):
        multiplier += policy.differential.dangerous_context_boost * max(hypothesis.urgency_weight, 0.5)
    if context.renal_impairment and hypothesis.slug == "heart_failure":
        multiplier += 0.05
    if hypothesis.slug == "upper_gi_bleed":
        if present_keys & {"active_gi_bleeding", "melena", "symptomatic_anemia"}:
            multiplier += 0.55
        if adverse_events & {"gi bleed", "melena", "symptomatic anemia"}:
            multiplier += 0.3
        if medications & {"warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran", "enoxaparin"}:
            multiplier += 0.15
    return multiplier


def _coverage_multiplier(coverage: float, policy: ReasoningPolicy) -> float:
    return 1.0 + (coverage * policy.differential.coverage_bonus_strength)


def _contradiction_multiplier(contradiction_load: float, policy: ReasoningPolicy) -> float:
    return 1.0 / (1.0 + (policy.differential.contradiction_penalty_strength * contradiction_load))


def _sample_hypothesis_posterior(
    context: ClinicalDecisionContext,
    hypothesis: DiagnosisHypothesis,
    rng: np.random.Generator,
    adjustment_factor: float = 1.0,
) -> float:
    findings_by_key = {finding.key: finding for finding in context.findings}
    sampled_prior = sample_probability(hypothesis.prior, None, None, rng)
    posterior = sampled_prior
    for evidence in hypothesis.supporting_findings + hypothesis.contradicting_findings:
        finding = findings_by_key.get(evidence.finding_key)
        if finding is None or finding.present is None:
            continue
        center = evidence.lr.positive_lr if finding.present else evidence.lr.negative_lr
        low = evidence.lr.positive_lr_low if finding.present else evidence.lr.negative_lr_low
        high = evidence.lr.positive_lr_high if finding.present else evidence.lr.negative_lr_high
        posterior = bayes_update(posterior, sample_lr(center, low, high, rng))
    return posterior * adjustment_factor


class DifferentialEngine:
    def rank(
        self,
        context: ClinicalDecisionContext,
        hypotheses: list[DiagnosisHypothesis],
        samples: int = 500,
        seed: int = 17,
        policy: ReasoningPolicy | str | None = None,
    ) -> DifferentialResult:
        resolved_policy = get_reasoning_policy(policy) if isinstance(policy, str) or policy is None else policy
        findings_by_key = {finding.key: finding for finding in context.findings}
        raw_posteriors: dict[str, float] = {}
        evidence_for: dict[str, list] = {}
        evidence_against: dict[str, list] = {}
        coverages: dict[str, float] = {}
        adjustment_factors: dict[str, float] = {}

        for hypothesis in hypotheses:
            posterior, support = update_probability_from_evidence(
                prior=hypothesis.prior,
                findings_by_key=findings_by_key,
                evidence_items=hypothesis.supporting_findings + hypothesis.contradicting_findings,
            )
            coverage = _symptom_coverage(context, hypothesis)
            adjusted_posterior = posterior
            adjusted_posterior *= _contextual_prior_multiplier(context, hypothesis, resolved_policy)
            adjusted_posterior *= _coverage_multiplier(coverage, resolved_policy)
            adjusted_posterior *= _contradiction_multiplier(_contradiction_load(support), resolved_policy)
            raw_posteriors[hypothesis.slug] = adjusted_posterior
            adjustment_factors[hypothesis.slug] = adjusted_posterior / max(posterior, 1e-9)
            evidence_for[hypothesis.slug] = [item for item in support if item.direction == "for"]
            evidence_against[hypothesis.slug] = [item for item in support if item.direction == "against"]
            coverages[hypothesis.slug] = coverage

        normalized = _apply_explaining_away(normalize_distribution(raw_posteriors), coverages, resolved_policy)
        rng = np.random.default_rng(seed)
        sampled = [
            {
                hypothesis.slug: _sample_hypothesis_posterior(
                    context,
                    hypothesis,
                    rng,
                    adjustment_factor=adjustment_factors.get(hypothesis.slug, 1.0),
                )
                for hypothesis in hypotheses
            }
            for _ in range(max(samples, resolved_policy.differential.sample_count))
        ]
        intervals = normalize_sampled_distributions(sampled)

        ranked_entries: list[DifferentialEntry] = []
        for hypothesis in hypotheses:
            interval = intervals.get(
                hypothesis.slug,
                ProbabilityInterval(
                    mean=normalized[hypothesis.slug],
                    low=normalized[hypothesis.slug],
                    high=normalized[hypothesis.slug],
                ),
            )
            ranked_entries.append(
                DifferentialEntry(
                    slug=hypothesis.slug,
                    name=hypothesis.name,
                    prior=hypothesis.prior,
                    posterior=normalized[hypothesis.slug],
                    interval_low=interval.low,
                    interval_high=interval.high,
                    evidence_for=evidence_for[hypothesis.slug],
                    evidence_against=evidence_against[hypothesis.slug],
                    symptom_coverage=coverages[hypothesis.slug],
                    explaining_away=self._explanations(hypothesis.slug, normalized, coverages),
                    calibration_state=classify_calibration(normalized[hypothesis.slug], interval.high - interval.low),
                    provenance_badges=self._provenance_badges(evidence_for[hypothesis.slug], evidence_against[hypothesis.slug]),
                )
            )

        ranked_entries.sort(key=lambda item: item.posterior, reverse=True)
        return DifferentialResult(
            ranked=ranked_entries,
            posterior_mass_top3=sum(entry.posterior for entry in ranked_entries[:3]),
            posterior_entropy=shannon_entropy({entry.slug: entry.posterior for entry in ranked_entries}),
            model_note=(
                f"Hybrid deterministic Bayesian differential with Monte Carlo uncertainty, "
                f"explaining-away, and reasoning policy `{resolved_policy.version}`."
            ),
        )

    @staticmethod
    def _explanations(slug: str, normalized: Mapping[str, float], coverages: Mapping[str, float]) -> list[str]:
        top_slug = max(normalized, key=normalized.get)
        if slug == top_slug:
            return []
        overlap = min(coverages.get(slug, 0.0), coverages.get(top_slug, 0.0))
        if overlap > 0.25:
            return [f"{top_slug} covers overlapping findings and suppresses this hypothesis slightly."]
        return []

    @staticmethod
    def _provenance_badges(evidence_for: list, evidence_against: list) -> list[str]:
        refs: set[str] = set()
        source_types: set[str] = set()
        for item in evidence_for + evidence_against:
            refs.update(item.provenance_refs)
            source_types.add(item.source_type.replace("_", "-"))
        badges = [f"source:{source_type}" for source_type in sorted(source_types) if source_type]
        badges.extend(sorted(refs))
        return badges
