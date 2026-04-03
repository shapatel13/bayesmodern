from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from datasets.reviewed_case_store import REVIEWED_CASE_TAGS
from evidence.cost_catalog import TEST_CATALOG
from evidence.latent_state_catalog import LATENT_STATE_CATALOG
from llm.openai_client import build_openai_client
from llm.structured_output import ResearchReport
from utils.config import Settings


class AIReviewerFeedback(BaseModel):
    summary: str
    gold_diagnosis: str | None = None
    reviewed_mechanism_states: list[str] = Field(default_factory=list)
    reviewed_contributing_processes: list[str] = Field(default_factory=list)
    acceptable_tests: list[str] = Field(default_factory=list)
    gold_triage: Literal["routine", "expedited", "urgent", "emergent"] | None = None
    review_tags: list[str] = Field(default_factory=list)
    preferred_next_action: str | None = None
    mechanism_feedback_summary: str
    review_notes: str
    review_status_recommendation: Literal["approved", "draft"] = "draft"
    reviewer_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def _slugish(value: str | None) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _inventory_aliases(pairs: list[tuple[str, str]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for slug, name in pairs:
        aliases[_slugish(slug)] = slug
        aliases[_slugish(name)] = slug
    return aliases


def _normalize_choices(values: list[str], aliases: dict[str, str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        resolved = aliases.get(_slugish(value))
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(resolved)
    return normalized


def _normalize_tags(values: list[str]) -> list[str]:
    allowed = {tag.lower() for tag in REVIEWED_CASE_TAGS}
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = str(value or "").strip().lower()
        if not tag or tag not in allowed or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def _review_payload(report: ResearchReport) -> dict[str, object]:
    return {
        "context": {
            "specialty": report.context.specialty,
            "findings": [finding.model_dump() for finding in report.context.findings],
            "completed_tests": report.context.completed_tests,
            "medications": report.context.medications,
            "adverse_events": report.context.adverse_events,
            "age_years": report.context.age_years,
            "pregnant": report.context.pregnant,
            "renal_impairment": report.context.renal_impairment,
            "hemodynamic_instability": report.context.hemodynamic_instability,
            "critical_values_present": report.context.critical_values_present,
        },
        "differential": [
            {
                "slug": entry.slug,
                "name": entry.name,
                "posterior": entry.posterior,
                "interval_low": entry.interval_low,
                "interval_high": entry.interval_high,
                "evidence_for": [item.label for item in entry.evidence_for[:4]],
                "evidence_against": [item.label for item in entry.evidence_against[:4]],
            }
            for entry in report.differential.ranked[:5]
        ],
        "mechanism_states": [
            {
                "slug": entry.slug,
                "name": entry.name,
                "posterior": entry.posterior,
                "interval_low": entry.interval_low,
                "interval_high": entry.interval_high,
            }
            for entry in report.mechanism_states.ranked[:6]
        ],
        "mechanism_summary": report.mechanism_states.summary,
        "next_best_tests": [
            {
                "slug": item.slug,
                "name": item.name,
                "score": item.score,
                "target_states": item.target_states,
                "rationale": item.rationale,
            }
            for item in report.next_best_tests[:5]
        ],
        "triage": report.triage.model_dump(),
        "threshold_decision": report.threshold_decision.model_dump(),
        "reasoning_runtime": report.reasoning_runtime.model_dump(),
        "decision_quality": report.decision_quality.model_dump(),
        "contradictions": report.contradictions,
        "provenance_warnings": report.provenance_warnings,
    }


def review_case_with_openai(case_text: str, report: ResearchReport, settings: Settings) -> AIReviewerFeedback:
    if not settings.allow_live_llm:
        raise ValueError("AI review requires PRIORI_ALLOW_LIVE_LLM=true.")
    if not settings.openai_api_key:
        raise ValueError("AI review requires OPENAI_API_KEY.")

    client = build_openai_client(settings)
    diagnosis_aliases = _inventory_aliases([(entry.slug, entry.name) for entry in report.differential.ranked[:8]])
    mechanism_aliases = _inventory_aliases([(slug, definition.name) for slug, definition in LATENT_STATE_CATALOG.items()])
    test_aliases = _inventory_aliases([(slug, test.name) for slug, test in TEST_CATALOG.items()])
    review_payload = _review_payload(report)

    parsed = client.responses.parse(
        model=settings.openai_case_review_model,
        reasoning={"effort": "medium"},
        text_format=AIReviewerFeedback,
        input=[
            {
                "role": "system",
                "content": (
                    "You are the PRIORI-X senior review model for offline case-quality review. "
                    "You are reviewing an existing deterministic Bayesian/mechanistic output, not replacing it. "
                    "Focus on whether the current output missed the dominant physiology, missed an obvious contributor, "
                    "chose a poor next test, used weak calibration, or undercalled urgency. "
                    "Return only the structured schema. "
                    "Do not write disease-specific prompt templates. "
                    "Prefer reusable mechanism-state language and structured next-step feedback. "
                    "If disease labeling is not the main correction, you may leave gold_diagnosis null. "
                    "Use only these review tags when applicable: "
                    + ", ".join(REVIEWED_CASE_TAGS)
                    + ". "
                    "Prefer these known mechanism slugs when applicable: "
                    + ", ".join(LATENT_STATE_CATALOG.keys())
                    + ". "
                    "Prefer these known test slugs when applicable: "
                    + ", ".join(TEST_CATALOG.keys())
                    + "."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Case text:\n"
                    + case_text
                    + "\n\nCurrent PRIORI-X output summary:\n"
                    + json.dumps(review_payload, indent=2)
                    + "\n\nKnown diagnosis aliases:\n"
                    + json.dumps(diagnosis_aliases, indent=2)
                    + "\n\nKnown mechanism aliases:\n"
                    + json.dumps(mechanism_aliases, indent=2)
                    + "\n\nKnown test aliases:\n"
                    + json.dumps(test_aliases, indent=2)
                ),
            },
        ],
    )

    result = parsed.output_parsed
    normalized_diagnosis = diagnosis_aliases.get(_slugish(result.gold_diagnosis)) if result.gold_diagnosis else None
    normalized_mechanisms = _normalize_choices(result.reviewed_mechanism_states, mechanism_aliases)
    normalized_tests = _normalize_choices(result.acceptable_tests, test_aliases)
    normalized_tags = _normalize_tags(result.review_tags)

    return result.model_copy(
        update={
            "gold_diagnosis": normalized_diagnosis or result.gold_diagnosis,
            "reviewed_mechanism_states": normalized_mechanisms,
            "acceptable_tests": normalized_tests,
            "review_tags": normalized_tags,
        }
    )
