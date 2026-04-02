from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from agent.prompt_registry import render_task_prompt, resolve_prompt_template
from core.models import CandidateTest, ClinicalDecisionContext, ClinicalFinding, DiagnosisHypothesis, HypothesisEvidence, LikelihoodRatioRange
from llm.openai_client import build_openai_client
from utils.config import Settings
from utils.logging import get_logger


Strength = Literal["weak", "moderate", "strong"]
EvidenceDirection = Literal["for", "against"]
OPEN_WORLD_HYPOTHESIS_REF = "llm:open_world_hypothesis_generation"
OPEN_WORLD_TEST_REF = "llm:open_world_test_generation"
logger = get_logger(__name__)


class OpenWorldEvidenceClue(BaseModel):
    label: str
    finding_key: str | None = None
    direction: EvidenceDirection = "for"
    strength: Strength = "moderate"
    note: str | None = None


class OpenWorldHypothesisProposal(BaseModel):
    slug: str
    name: str
    category: str = "general"
    dangerous: bool = False
    urgency_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    prior_hint: float = Field(default=0.08, gt=0.0, lt=1.0)
    rationale: str
    evidence_clues: list[OpenWorldEvidenceClue] = Field(default_factory=list)


class OpenWorldTestProposal(BaseModel):
    slug: str
    name: str
    target_diagnoses: list[str] = Field(default_factory=list)
    rationale: str
    rule_in_strength: Strength = "moderate"
    rule_out_strength: Strength = "weak"
    actionability: float = Field(default=0.7, ge=0.0, le=1.0)
    urgency_modifier: float = Field(default=1.0, ge=0.5, le=2.0)
    direct_cost_hint_usd: float | None = Field(default=None, ge=0.0)
    downstream_cost_hint_usd: float | None = Field(default=None, ge=0.0)
    invasiveness: float = Field(default=0.0, ge=0.0, le=1.0)
    radiation: float = Field(default=0.0, ge=0.0, le=1.0)
    nephrotoxicity: float = Field(default=0.0, ge=0.0, le=1.0)
    bleed_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    logistic_burden: float = Field(default=0.0, ge=0.0, le=1.0)
    bedside: bool = False


class OpenWorldReasoningPlan(BaseModel):
    specialty: str = "general_internal_medicine"
    hypotheses: list[OpenWorldHypothesisProposal] = Field(default_factory=list)
    suggested_tests: list[OpenWorldTestProposal] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unspecified"


def _normalize_text(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _default_prompt_template(prompt_template: str | None) -> str:
    if prompt_template:
        return prompt_template
    return resolve_prompt_template("active")


def _context_summary(context: ClinicalDecisionContext) -> str:
    finding_lines = [
        f"- {finding.label} [{finding.key}]"
        for finding in context.findings
        if finding.present
    ]
    medications = ", ".join(context.medications) or "none"
    adverse_events = ", ".join(context.adverse_events) or "none"
    completed_tests = ", ".join(context.completed_tests) or "none"
    comorbidities = ", ".join(context.comorbidities) or "none"
    return "\n".join(
        [
            f"Free-text note: {context.symptoms_free_text or 'none'}",
            f"Current specialty hint: {context.specialty}",
            "Structured positive findings:",
            *(finding_lines or ["- none"]),
            f"Medications: {medications}",
            f"Adverse events / harms: {adverse_events}",
            f"Completed tests: {completed_tests}",
            f"Comorbidities: {comorbidities}",
            f"Hemodynamic instability: {context.hemodynamic_instability}",
            f"Critical values present: {context.critical_values_present}",
            f"Renal impairment: {context.renal_impairment}",
            f"Pregnant: {context.pregnant}",
        ]
    )


def _clue_supported_by_context(clue: OpenWorldEvidenceClue, context: ClinicalDecisionContext) -> bool:
    note_text = _normalize_text(context.symptoms_free_text)
    structured_values = (
        [finding.label for finding in context.findings]
        + [finding.key.replace("_", " ") for finding in context.findings]
        + context.medications
        + context.adverse_events
        + context.comorbidities
        + context.completed_tests
    )
    searchable = " ".join(
        part
        for part in [note_text, *(_normalize_text(item) for item in structured_values if item)]
        if part
    )
    searchable_tokens = set(searchable.split())

    for candidate in [clue.label, clue.finding_key or ""]:
        normalized = _normalize_text(str(candidate).replace("_", " "))
        if not normalized:
            continue
        if normalized in searchable:
            return True
        tokens = [token for token in normalized.split() if len(token) > 2]
        if tokens and all(token in searchable_tokens for token in tokens):
            return True
    return False


def _positive_lr_for_strength(strength: Strength) -> LikelihoodRatioRange:
    if strength == "strong":
        return LikelihoodRatioRange(positive_lr=3.2, negative_lr=0.45, positive_lr_low=2.2, positive_lr_high=4.4, negative_lr_low=0.3, negative_lr_high=0.65)
    if strength == "moderate":
        return LikelihoodRatioRange(positive_lr=1.8, negative_lr=0.7, positive_lr_low=1.35, positive_lr_high=2.35, negative_lr_low=0.55, negative_lr_high=0.85)
    return LikelihoodRatioRange(positive_lr=1.25, negative_lr=0.85, positive_lr_low=1.08, positive_lr_high=1.45, negative_lr_low=0.72, negative_lr_high=0.95)


def _negative_lr_for_strength(strength: Strength) -> LikelihoodRatioRange:
    if strength == "strong":
        return LikelihoodRatioRange(positive_lr=0.4, negative_lr=1.45, positive_lr_low=0.25, positive_lr_high=0.55, negative_lr_low=1.2, negative_lr_high=1.7)
    if strength == "moderate":
        return LikelihoodRatioRange(positive_lr=0.65, negative_lr=1.18, positive_lr_low=0.45, positive_lr_high=0.82, negative_lr_low=1.05, negative_lr_high=1.32)
    return LikelihoodRatioRange(positive_lr=0.85, negative_lr=1.05, positive_lr_low=0.72, positive_lr_high=0.95, negative_lr_low=1.0, negative_lr_high=1.12)


def _test_lr(rule_in_strength: Strength, rule_out_strength: Strength) -> LikelihoodRatioRange:
    positive = _positive_lr_for_strength(rule_in_strength)
    negative = _positive_lr_for_strength(rule_out_strength)
    return LikelihoodRatioRange(
        positive_lr=positive.positive_lr,
        negative_lr=negative.negative_lr,
        positive_lr_low=positive.positive_lr_low,
        positive_lr_high=positive.positive_lr_high,
        negative_lr_low=negative.negative_lr_low,
        negative_lr_high=negative.negative_lr_high,
    )


def generate_open_world_reasoning_plan(
    context: ClinicalDecisionContext,
    settings: Settings,
    *,
    prompt_template: str | None = None,
) -> OpenWorldReasoningPlan | None:
    if not settings.open_world_reasoning_enabled:
        return None
    if not settings.allow_live_llm or settings.default_model_provider != "openai" or not settings.openai_api_key:
        return None

    client = build_openai_client(settings)
    rendered_prompt = render_task_prompt(_default_prompt_template(prompt_template), context.symptoms_free_text or "")
    response = client.responses.parse(
        model=settings.openai_reasoning_model,
        reasoning={"effort": "medium"},
        text_format=OpenWorldReasoningPlan,
        input=[
            {
                "role": "system",
                "content": (
                    "You are PRIORI-X's open-world clinical reasoning expander in an offline research sandbox. "
                    "Your job is to propose diagnoses and next tests for arbitrary cases, even when they are outside the starter ontology. "
                    "Use only clues present in the provided case. Do not invent labs, imaging, or history. "
                    "Return 3 to 6 diagnoses that together cover the likely and dangerous possibilities. "
                    "For each diagnosis, provide reusable clue labels and snake_case finding keys. Use near-verbatim clue wording from the case when possible. "
                    "Reuse familiar slugs like acs, pe, pneumonia, heart_failure, and upper_gi_bleed when they truly fit, "
                    "but introduce new slugs when the case needs them. "
                    "Prior hints should be conservative and usually below 0.25 unless the case is dominated by one diagnosis. "
                    "Suggested tests should be the highest-yield next steps, not treatment orders."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Prompt context:\n{rendered_prompt}\n\n"
                    f"Structured case summary:\n{_context_summary(context)}"
                ),
            },
        ],
    )
    plan = response.output_parsed
    return OpenWorldReasoningPlan(
        specialty=plan.specialty,
        hypotheses=plan.hypotheses[: settings.open_world_max_hypotheses],
        suggested_tests=plan.suggested_tests[: settings.open_world_max_tests],
        notes=plan.notes,
    )


def augment_context_with_open_world_findings(
    context: ClinicalDecisionContext,
    plan: OpenWorldReasoningPlan | None,
) -> ClinicalDecisionContext:
    if plan is None:
        return context

    merged_findings: dict[str, ClinicalFinding] = {finding.key: finding for finding in context.findings}
    for hypothesis in plan.hypotheses:
        for clue in hypothesis.evidence_clues:
            clue_key = _slugify(clue.finding_key or clue.label)
            if clue_key in merged_findings:
                continue
            merged_findings[clue_key] = ClinicalFinding(
                key=clue_key,
                label=clue.label.strip(),
                present=True,
                note=clue.note,
                source_type="llm_inferred",
            )

    return context.model_copy(
        update={
            "specialty": plan.specialty or context.specialty,
            "findings": list(merged_findings.values()),
        }
    )


def _proposal_to_hypothesis(
    proposal: OpenWorldHypothesisProposal,
    *,
    context: ClinicalDecisionContext | None = None,
    prior_cap: float = 0.12,
) -> DiagnosisHypothesis:
    supporting: list[HypothesisEvidence] = []
    contradicting: list[HypothesisEvidence] = []
    for clue in proposal.evidence_clues:
        if context is not None and not _clue_supported_by_context(clue, context):
            continue
        finding_key = _slugify(clue.finding_key or clue.label)
        evidence = HypothesisEvidence(
            finding_key=finding_key,
            label=clue.label.strip(),
            lr=(
                _positive_lr_for_strength(clue.strength)
                if clue.direction == "for"
                else _negative_lr_for_strength(clue.strength)
            ),
            rationale=(clue.note or proposal.rationale).strip(),
            provenance_refs=[OPEN_WORLD_HYPOTHESIS_REF],
            source_type="llm_inferred",
        )
        if clue.direction == "against":
            contradicting.append(evidence)
        else:
            supporting.append(evidence)

    return DiagnosisHypothesis(
        slug=_slugify(proposal.slug or proposal.name),
        name=proposal.name.strip() or proposal.slug.replace("_", " ").title(),
        category=proposal.category.strip() or "general",
        prior=min(max(proposal.prior_hint, 0.01), prior_cap),
        dangerous=proposal.dangerous,
        urgency_weight=proposal.urgency_weight,
        supporting_findings=supporting,
        contradicting_findings=contradicting,
    )


def merge_open_world_hypotheses(
    base_hypotheses: list[DiagnosisHypothesis],
    plan: OpenWorldReasoningPlan | None,
    *,
    context: ClinicalDecisionContext | None = None,
) -> list[DiagnosisHypothesis]:
    if plan is None or not plan.hypotheses:
        return base_hypotheses

    merged: dict[str, DiagnosisHypothesis] = {hypothesis.slug: hypothesis.model_copy(deep=True) for hypothesis in base_hypotheses}
    baseline_finding_keys = {
        evidence.finding_key
        for hypothesis in base_hypotheses
        for evidence in [*hypothesis.supporting_findings, *hypothesis.contradicting_findings]
    }
    for proposal in plan.hypotheses:
        proposal_slug = _slugify(proposal.slug or proposal.name)
        proposal_clue_keys = {
            _slugify(clue.finding_key or clue.label)
            for clue in proposal.evidence_clues
        }
        if proposal_slug in merged:
            prior_cap = 0.35
        elif proposal_clue_keys - baseline_finding_keys:
            prior_cap = 0.12
        else:
            prior_cap = 0.03
        generated = _proposal_to_hypothesis(
            proposal,
            context=context,
            prior_cap=prior_cap,
        )
        if proposal_slug not in merged and not generated.supporting_findings and not generated.contradicting_findings:
            continue
        if generated.slug in merged:
            existing = merged[generated.slug]
            support_keys = {item.finding_key for item in existing.supporting_findings}
            against_keys = {item.finding_key for item in existing.contradicting_findings}
            merged_support = existing.supporting_findings + [
                item for item in generated.supporting_findings if item.finding_key not in support_keys
            ]
            merged_against = existing.contradicting_findings + [
                item for item in generated.contradicting_findings if item.finding_key not in against_keys
            ]
            merged[generated.slug] = existing.model_copy(
                update={
                    "dangerous": existing.dangerous or generated.dangerous,
                    "urgency_weight": max(existing.urgency_weight, generated.urgency_weight),
                    "supporting_findings": merged_support,
                    "contradicting_findings": merged_against,
                }
            )
        else:
            merged[generated.slug] = generated
    return list(merged.values())


def _proposal_to_candidate_test(proposal: OpenWorldTestProposal) -> CandidateTest:
    slug = _slugify(proposal.slug or proposal.name)
    target_diagnoses = [_slugify(item) for item in proposal.target_diagnoses if str(item).strip()]
    lr = _test_lr(proposal.rule_in_strength, proposal.rule_out_strength)
    diagnosis_lrs = {diagnosis_slug: lr for diagnosis_slug in target_diagnoses}
    direct_cost = proposal.direct_cost_hint_usd if proposal.direct_cost_hint_usd is not None else (80.0 if proposal.bedside else 250.0)
    downstream_cost = proposal.downstream_cost_hint_usd or 0.0
    return CandidateTest(
        slug=slug,
        name=proposal.name.strip() or slug.replace("_", " ").title(),
        target_diagnoses=target_diagnoses,
        diagnosis_lrs=diagnosis_lrs,
        direct_cost=direct_cost,
        downstream_cost=downstream_cost,
        invasiveness=proposal.invasiveness,
        radiation=proposal.radiation,
        nephrotoxicity=proposal.nephrotoxicity,
        bleed_risk=proposal.bleed_risk,
        logistic_burden=proposal.logistic_burden,
        actionability=proposal.actionability,
        urgency_modifier=proposal.urgency_modifier,
        bedside=proposal.bedside,
        evidence_note=proposal.rationale.strip(),
        provenance_refs=[OPEN_WORLD_TEST_REF],
        source_type="llm_inferred",
    )


def merge_open_world_candidate_tests(
    base_tests: list[CandidateTest],
    plan: OpenWorldReasoningPlan | None,
) -> list[CandidateTest]:
    if plan is None or not plan.suggested_tests:
        return base_tests

    merged: dict[str, CandidateTest] = {candidate.slug: candidate.model_copy(deep=True) for candidate in base_tests}
    for proposal in plan.suggested_tests:
        generated = _proposal_to_candidate_test(proposal)
        if not generated.target_diagnoses:
            continue
        if generated.slug in merged:
            existing = merged[generated.slug]
            combined_targets = list(dict.fromkeys([*existing.target_diagnoses, *generated.target_diagnoses]))
            combined_lrs = dict(existing.diagnosis_lrs)
            for diagnosis_slug, lr in generated.diagnosis_lrs.items():
                combined_lrs.setdefault(diagnosis_slug, lr)
            combined_refs = list(dict.fromkeys([*existing.provenance_refs, *generated.provenance_refs]))
            note_parts = [part for part in [existing.evidence_note, generated.evidence_note] if part]
            merged[generated.slug] = existing.model_copy(
                update={
                    "target_diagnoses": combined_targets,
                    "diagnosis_lrs": combined_lrs,
                    "evidence_note": " ".join(dict.fromkeys(note_parts)) or None,
                    "provenance_refs": combined_refs,
                }
            )
        else:
            merged[generated.slug] = generated
    return list(merged.values())
