from __future__ import annotations

from core.differential import DifferentialEngine
from core.next_best_test import NextBestTestEngine
from core.policy import ReasoningPolicy, get_reasoning_policy
from core.thresholds import DecisionCostModel, calculate_thresholds, explain_threshold_position
from core.triage import assess_triage
from agent.prompt_registry import resolve_prompt_template
from evidence.cost_catalog import default_test_catalog
from evidence.disease_profiles import DISEASE_PROFILES, default_hypotheses
from llm.citation_checker import find_missing_citations
from llm.extraction import extract_context_from_text
from llm.model_router import route_models
from llm.open_world_reasoning import (
    augment_context_with_open_world_findings,
    generate_open_world_reasoning_plan,
    merge_open_world_candidate_tests,
    merge_open_world_hypotheses,
)
from llm.structured_output import ResearchReport
from llm.structured_output import ReasoningRuntimeTrace
from llm.validators import validate_report
from utils.config import Settings, get_settings
from utils.logging import get_logger


logger = get_logger(__name__)


def _should_expand_open_world(differential) -> bool:
    if not differential.ranked:
        return True
    top_entry = differential.ranked[0]
    runner_up = differential.ranked[1].posterior if len(differential.ranked) > 1 else 0.0
    return not (
        top_entry.posterior >= 0.4
        and top_entry.symptom_coverage >= 0.5
        and len(top_entry.evidence_for) >= 2
        and (top_entry.posterior - runner_up) >= 0.08
    )


def _open_world_gate_reason(differential) -> str:
    if not differential.ranked:
        return "No ranked differential available yet, so open-world expansion is allowed."
    top_entry = differential.ranked[0]
    runner_up = differential.ranked[1].posterior if len(differential.ranked) > 1 else 0.0
    margin = top_entry.posterior - runner_up
    if (
        top_entry.posterior >= 0.4
        and top_entry.symptom_coverage >= 0.5
        and len(top_entry.evidence_for) >= 2
        and margin >= 0.08
    ):
        return (
            f"Curated Bayesian pass was already confident: top diagnosis `{top_entry.slug}` "
            f"posterior {top_entry.posterior:.3f}, coverage {top_entry.symptom_coverage:.2f}, "
            f"evidence_for {len(top_entry.evidence_for)}, margin {margin:.3f}."
        )
    return (
        f"Curated Bayesian pass was not yet decisive: top diagnosis `{top_entry.slug}` "
        f"posterior {top_entry.posterior:.3f}, coverage {top_entry.symptom_coverage:.2f}, "
        f"evidence_for {len(top_entry.evidence_for)}, margin {margin:.3f}."
    )


class PRIORIXOrchestrator:
    def __init__(self, settings: Settings | None = None, *, policy_version: str = "v1-deterministic") -> None:
        self.settings = settings or get_settings()
        self.policy = get_reasoning_policy(policy_version)
        self.differential_engine = DifferentialEngine()
        self.next_test_engine = NextBestTestEngine()

    def analyze_text_case(
        self,
        case_id: str,
        note_text: str,
        *,
        policy_version: str | None = None,
        prompt_template: str | None = None,
    ) -> ResearchReport:
        resolved_prompt_template = prompt_template or resolve_prompt_template("active")
        context = extract_context_from_text(
            case_id=case_id,
            note_text=note_text,
            settings=self.settings,
            prompt_template=resolved_prompt_template,
        )
        return self.analyze_context(
            context,
            policy_version=policy_version,
            prompt_template=resolved_prompt_template,
        )

    def analyze_context(self, context, *, policy_version: str | None = None, prompt_template: str | None = None) -> ResearchReport:
        policy: ReasoningPolicy = get_reasoning_policy(policy_version) if policy_version else self.policy
        base_hypotheses = default_hypotheses(context=context)
        base_differential = self.differential_engine.rank(
            context,
            base_hypotheses,
            seed=self.settings.seed,
            policy=policy,
        )
        open_world_plan = None
        differential = base_differential
        hypotheses = base_hypotheses
        gate_reason = (
            "Open-world expansion disabled by configuration."
            if not self.settings.open_world_reasoning_enabled
            else _open_world_gate_reason(base_differential)
        )
        should_expand = self.settings.open_world_reasoning_enabled and (
            not self.settings.open_world_expand_uncertain_only or _should_expand_open_world(base_differential)
        )
        if should_expand:
            try:
                open_world_plan = generate_open_world_reasoning_plan(
                    context,
                    self.settings,
                    prompt_template=prompt_template,
                )
            except Exception as exc:
                logger.warning("Open-world reasoning generation failed; continuing with deterministic hypothesis pool: %s", exc)
            context = augment_context_with_open_world_findings(context, open_world_plan)
            hypotheses = merge_open_world_hypotheses(
                default_hypotheses(context=context),
                open_world_plan,
                context=context,
            )
            differential = self.differential_engine.rank(
                context,
                hypotheses,
                seed=self.settings.seed,
                policy=policy,
            )
        dangerous_mass = sum(
            entry.posterior * DISEASE_PROFILES[entry.slug].urgency_weight
            for entry in differential.ranked
            if DISEASE_PROFILES.get(entry.slug, None) and DISEASE_PROFILES[entry.slug].dangerous
        )
        triage = assess_triage(dangerous_mass, context.hemodynamic_instability, context.critical_values_present)
        urgency_multiplier = 1.0
        if triage.urgency in {"urgent", "emergent"}:
            urgency_multiplier = policy.thresholds.urgent_multiplier
        elif triage.urgency == "expedited":
            urgency_multiplier = policy.thresholds.expedited_multiplier
        thresholds = calculate_thresholds(
            DecisionCostModel(
                benefit_of_treatment=policy.thresholds.benefit_of_treatment,
                harm_of_treatment=policy.thresholds.harm_of_treatment,
                harm_of_missed_disease=policy.thresholds.harm_of_missed_disease,
                harm_of_test=policy.thresholds.harm_of_test,
                discharge_harm=policy.thresholds.discharge_harm,
                icu_overuse_harm=policy.thresholds.icu_overuse_harm,
                urgency_multiplier=urgency_multiplier,
            )
        )
        posterior_map = {entry.slug: entry.posterior for entry in differential.ranked}
        candidates = merge_open_world_candidate_tests(
            default_test_catalog(context=context),
            open_world_plan,
        )
        recommendations = self.next_test_engine.rank(
            posterior_map,
            candidates,
            context,
            thresholds=thresholds,
            policy=policy,
        )[:5]
        threshold_decision = explain_threshold_position(differential.ranked[0].posterior, thresholds)
        report = ResearchReport(
            context=context,
            differential=differential,
            next_best_tests=recommendations,
            triage=triage,
            threshold_decision=threshold_decision,
            reasoning_runtime=ReasoningRuntimeTrace(
                mode="hybrid_open_world" if open_world_plan and (open_world_plan.hypotheses or open_world_plan.suggested_tests) else "curated_only",
                open_world_considered=self.settings.open_world_reasoning_enabled,
                open_world_triggered=bool(open_world_plan and (open_world_plan.hypotheses or open_world_plan.suggested_tests)),
                gate_reason=gate_reason,
                base_top_diagnosis=base_differential.ranked[0].slug if base_differential.ranked else None,
                base_top_posterior=base_differential.ranked[0].posterior if base_differential.ranked else None,
                final_top_diagnosis=differential.ranked[0].slug if differential.ranked else None,
                final_top_posterior=differential.ranked[0].posterior if differential.ranked else None,
                open_world_hypothesis_count=len(open_world_plan.hypotheses) if open_world_plan else 0,
                open_world_test_count=len(open_world_plan.suggested_tests) if open_world_plan else 0,
                notes=(open_world_plan.notes if open_world_plan else []),
            ),
            model_route=route_models(self.settings),
        )
        report.contradictions = validate_report(report)
        report.provenance_warnings = find_missing_citations(report)
        report.differential.model_note = f"{report.differential.model_note} Triage mass {dangerous_mass:.3f}; policy `{policy.version}`."
        if open_world_plan and (open_world_plan.hypotheses or open_world_plan.suggested_tests):
            report.differential.model_note = (
                f"{report.differential.model_note} Open-world reasoning expanded the candidate space with "
                f"{len(open_world_plan.hypotheses)} hypothesis proposals and {len(open_world_plan.suggested_tests)} "
                "test proposals before deterministic Bayesian scoring."
            )
        return report
