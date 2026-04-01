from __future__ import annotations

from core.differential import DifferentialEngine
from core.next_best_test import NextBestTestEngine
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
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.differential_engine = DifferentialEngine()
        self.next_test_engine = NextBestTestEngine()

    def analyze_text_case(self, case_id: str, note_text: str) -> ResearchReport:
        context = extract_context_from_text(case_id=case_id, note_text=note_text)
        return self.analyze_context(context)

    def analyze_context(self, context) -> ResearchReport:
        differential = self.differential_engine.rank(context, default_hypotheses(), seed=self.settings.seed)
        posterior_map = {entry.slug: entry.posterior for entry in differential.ranked}
        recommendations = self.next_test_engine.rank(posterior_map, default_test_catalog(), context)[:5]
        dangerous_mass = sum(
            entry.posterior * DISEASE_PROFILES[entry.slug].urgency_weight
            for entry in differential.ranked
            if DISEASE_PROFILES.get(entry.slug, None) and DISEASE_PROFILES[entry.slug].dangerous
        )
        triage = assess_triage(dangerous_mass, context.hemodynamic_instability, context.critical_values_present)
        thresholds = calculate_thresholds(
            DecisionCostModel(
                benefit_of_treatment=10.0,
                harm_of_treatment=2.0,
                harm_of_missed_disease=9.0,
                harm_of_test=1.0,
                discharge_harm=2.5,
                urgency_multiplier=1.5 if triage.urgency in {"urgent", "emergent"} else 1.0,
            )
        )
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
        return report
