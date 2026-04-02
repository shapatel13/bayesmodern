from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from core.bayes import bayes_update
from core.calibration import classify_calibration
from core.models import (
    ClinicalDecisionContext,
    ClinicalFinding,
    FindingContribution,
    LatentStateDefinition,
    MechanismStateEstimate,
    MechanismStateResult,
    ProbabilityInterval,
)
from core.uncertainty import sample_lr, sample_probability, summarize_samples
from evidence.latent_state_catalog import LATENT_STATE_CATALOG


def _soften_lr(likelihood_ratio: float, confidence: float) -> float:
    bounded_confidence = min(max(confidence, 0.0), 1.0)
    if bounded_confidence == 0.0:
        return 1.0
    return float(math.exp(math.log(max(likelihood_ratio, 1e-6)) * bounded_confidence))


def _mechanism_badges(contributions: list[FindingContribution]) -> list[str]:
    refs: set[str] = set()
    source_types: set[str] = set()
    for contribution in contributions:
        refs.update(contribution.provenance_refs)
        source_types.add(contribution.source_type.replace("_", "-"))
    badges = [f"source:{source_type}" for source_type in sorted(source_types) if source_type]
    badges.extend(sorted(refs))
    return badges


def _derived_signal(
    key: str,
    label: str,
    support_count: int,
    *,
    threshold: int,
    confidence: float,
) -> ClinicalFinding | None:
    if support_count < threshold:
        return None
    return ClinicalFinding(
        key=key,
        label=label,
        present=True,
        confidence=confidence,
        source_type="hard_coded",
    )


def _findings_map(context: ClinicalDecisionContext) -> dict[str, ClinicalFinding]:
    findings_by_key = {finding.key: finding for finding in context.findings}
    if context.hemodynamic_instability:
        findings_by_key.setdefault(
            "hemodynamic_instability",
            ClinicalFinding(
                key="hemodynamic_instability",
                label="Hemodynamic instability",
                present=True,
                confidence=1.0,
                source_type="hard_coded",
            ),
        )
    if context.critical_values_present:
        findings_by_key.setdefault(
            "critical_values_present",
            ClinicalFinding(
                key="critical_values_present",
                label="Critical values present",
                present=True,
                confidence=1.0,
                source_type="hard_coded",
            ),
        )
    if context.renal_impairment:
        findings_by_key.setdefault(
            "renal_impairment",
            ClinicalFinding(
                key="renal_impairment",
                label="Renal impairment",
                present=True,
                confidence=1.0,
                source_type="hard_coded",
            ),
        )

    low_preload_count = sum(
        1
        for key in ("active_gi_bleeding", "melena", "symptomatic_anemia", "hemodynamic_instability", "tachycardia", "oliguria")
        if findings_by_key.get(key, ClinicalFinding(key=key, label=key)).present
    )
    venous_congestion_count = sum(
        1
        for key in ("orthopnea", "crackles", "leg_edema", "elevated_jvp")
        if findings_by_key.get(key, ClinicalFinding(key=key, label=key)).present
    )
    fluid_intolerance_count = sum(
        1
        for key in ("orthopnea", "crackles", "leg_edema", "elevated_jvp", "hypoxemia")
        if findings_by_key.get(key, ClinicalFinding(key=key, label=key)).present
    )

    derived_findings = [
        _derived_signal(
            "low_preload_signal",
            "Low preload signal",
            low_preload_count,
            threshold=2,
            confidence=min(1.0, 0.45 + (0.15 * low_preload_count)),
        ),
        _derived_signal(
            "venous_congestion_signal",
            "Venous congestion signal",
            venous_congestion_count,
            threshold=2,
            confidence=min(1.0, 0.45 + (0.18 * venous_congestion_count)),
        ),
        _derived_signal(
            "fluid_intolerance_signal",
            "Fluid intolerance signal",
            fluid_intolerance_count,
            threshold=2,
            confidence=min(1.0, 0.45 + (0.15 * fluid_intolerance_count)),
        ),
    ]
    for derived in derived_findings:
        if derived is not None:
            findings_by_key[derived.key] = derived
    return findings_by_key


def _apply_state_evidence(
    prior: float,
    definition: LatentStateDefinition,
    findings_by_key: Mapping[str, ClinicalFinding],
) -> tuple[float, list[FindingContribution]]:
    posterior = prior
    contributions: list[FindingContribution] = []
    for evidence in definition.supporting_findings + definition.contradicting_findings:
        finding = findings_by_key.get(evidence.finding_key)
        if finding is None or finding.present is None:
            continue
        base_lr = evidence.lr.positive_lr if finding.present else evidence.lr.negative_lr
        applied_lr = _soften_lr(base_lr, finding.confidence)
        posterior = bayes_update(posterior, applied_lr)
        direction = "for" if applied_lr > 1.0 else "against" if applied_lr < 1.0 else "neutral"
        contributions.append(
            FindingContribution(
                finding_key=evidence.finding_key,
                label=evidence.label,
                direction=direction,
                applied_lr=applied_lr,
                rationale=evidence.rationale,
                provenance_refs=evidence.provenance_refs,
                source_type=evidence.source_type,
            )
        )
    return posterior, contributions


def _sample_state_posterior(
    prior: float,
    definition: LatentStateDefinition,
    findings_by_key: Mapping[str, ClinicalFinding],
    rng: np.random.Generator,
) -> float:
    posterior = sample_probability(prior, None, None, rng)
    for evidence in definition.supporting_findings + definition.contradicting_findings:
        finding = findings_by_key.get(evidence.finding_key)
        if finding is None or finding.present is None:
            continue
        center = evidence.lr.positive_lr if finding.present else evidence.lr.negative_lr
        low = evidence.lr.positive_lr_low if finding.present else evidence.lr.negative_lr_low
        high = evidence.lr.positive_lr_high if finding.present else evidence.lr.negative_lr_high
        posterior = bayes_update(posterior, _soften_lr(sample_lr(center, low, high, rng), finding.confidence))
    return posterior


def _summary(ranked: list[MechanismStateEstimate]) -> tuple[list[str], bool, str]:
    if not ranked:
        return [], False, "Mechanism layer unavailable."
    active = [estimate.name for estimate in ranked if estimate.posterior >= 0.55][:3]
    mixed = len(active) >= 2
    if mixed:
        return active, True, f"Mixed physiology signal: {' plus '.join(active[:2])}."
    if active:
        return active, False, f"Dominant mechanism signal: {active[0]}."
    top_two = [estimate.name for estimate in ranked[:2]]
    return top_two, False, f"Mechanism layer remains broad; strongest signals are {' and '.join(top_two)}."


class LatentStateEngine:
    def infer(
        self,
        context: ClinicalDecisionContext,
        *,
        definitions: Mapping[str, LatentStateDefinition] | None = None,
        samples: int = 400,
        seed: int = 17,
    ) -> MechanismStateResult:
        state_definitions = definitions or LATENT_STATE_CATALOG
        findings_by_key = _findings_map(context)
        rng = np.random.default_rng(seed)
        estimates: list[MechanismStateEstimate] = []

        for definition in state_definitions.values():
            posterior, contributions = _apply_state_evidence(definition.prior, definition, findings_by_key)
            sampled_values = [
                _sample_state_posterior(definition.prior, definition, findings_by_key, rng)
                for _ in range(max(samples, 100))
            ]
            interval: ProbabilityInterval = summarize_samples(sampled_values)
            evidence_for = [item for item in contributions if item.direction == "for"]
            evidence_against = [item for item in contributions if item.direction == "against"]
            estimates.append(
                MechanismStateEstimate(
                    slug=definition.slug,
                    name=definition.name,
                    category=definition.category,
                    prior=definition.prior,
                    posterior=posterior,
                    interval_low=interval.low,
                    interval_high=interval.high,
                    evidence_for=evidence_for,
                    evidence_against=evidence_against,
                    confidence_state=classify_calibration(posterior, interval.high - interval.low),
                    provenance_badges=_mechanism_badges(contributions),
                )
            )

        estimates.sort(key=lambda item: item.posterior, reverse=True)
        active_states, mixed_physiology, summary = _summary(estimates)
        return MechanismStateResult(
            ranked=estimates,
            active_states=active_states,
            mixed_physiology=mixed_physiology,
            summary=summary,
            model_note="Independent deterministic latent-state updates with soft-evidence weighting and Monte Carlo uncertainty.",
        )

