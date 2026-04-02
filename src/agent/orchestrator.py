from __future__ import annotations

from core.differential import DifferentialEngine
from core.next_best_test import NextBestTestEngine
from core.policy import ReasoningPolicy, get_reasoning_policy
from core.thresholds import DecisionCostModel, calculate_thresholds, explain_threshold_position
from core.triage import assess_triage
from evidence.cost_catalog import default_test_catalog
from evidence.disease_profiles import DISEASE_PROFILES, default_hypotheses
from llm.citation_checker import find_missing_citations
from llm.extraction import extract_context_from_text
from llm.model_router import route_models
from llm.structured_output import ResearchReport
from llm.validators import validate_report
from utils.config import Settings, get_settings


class PRIORIXOrchestrator:
    def __init__(self, settings: Settings | None = None, *, policy_version: str = "v1-deterministic") -> None:
        self.settings = settings or get_settings()
        self.policy = get_reasoning_policy(policy_version)
        self.differential_engine = DifferentialEngine()
        self.next_test_engine = NextBestTestEngine()

    def analyze_text_case(self, case_id: str, note_text: str, *, policy_version: str | None = None) -> ResearchReport:
        context = extract_context_from_text(case_id=case_id, note_text=note_text, settings=self.settings)
        return self.analyze_context(context, policy_version=policy_version)

    def analyze_context(self, context, *, policy_version: str | None = None) -> ResearchReport:
        policy: ReasoningPolicy = get_reasoning_policy(policy_version) if policy_version else self.policy
        differential = self.differential_engine.rank(
            context,
            default_hypotheses(context=context),
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
        recommendations = self.next_test_engine.rank(
            posterior_map,
            default_test_catalog(context=context),
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
            model_route=route_models(self.settings),
        )
        report.contradictions = validate_report(report)
        report.provenance_warnings = find_missing_citations(report)
        report.differential.model_note = f"{report.differential.model_note} Triage mass {dangerous_mass:.3f}; policy `{policy.version}`."
        return report
