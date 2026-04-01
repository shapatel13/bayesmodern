from __future__ import annotations

from collections.abc import Mapping

from core.bayes import bayes_update
from core.models import ClinicalFinding, FindingContribution, HypothesisEvidence


def lr_for_finding(finding: ClinicalFinding, evidence: HypothesisEvidence) -> float:
    if finding.present is None:
        return 1.0
    return evidence.lr.positive_lr if finding.present else evidence.lr.negative_lr


def contribution_direction(applied_lr: float) -> str:
    if applied_lr > 1.0:
        return "for"
    if applied_lr < 1.0:
        return "against"
    return "neutral"


def update_probability_from_evidence(
    prior: float,
    findings_by_key: Mapping[str, ClinicalFinding],
    evidence_items: list[HypothesisEvidence],
) -> tuple[float, list[FindingContribution]]:
    posterior = prior
    contributions: list[FindingContribution] = []
    for evidence in evidence_items:
        finding = findings_by_key.get(evidence.finding_key)
        if finding is None:
            continue
        applied_lr = lr_for_finding(finding, evidence)
        posterior = bayes_update(posterior, applied_lr)
        contributions.append(
            FindingContribution(
                finding_key=evidence.finding_key,
                label=evidence.label,
                direction=contribution_direction(applied_lr),
                applied_lr=applied_lr,
                rationale=evidence.rationale,
                provenance_refs=evidence.provenance_refs,
                source_type=evidence.source_type,
            )
        )
    return posterior, contributions
