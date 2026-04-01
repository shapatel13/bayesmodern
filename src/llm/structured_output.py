from __future__ import annotations

from pydantic import BaseModel, Field

from core.models import ClinicalDecisionContext, DifferentialResult, TestRecommendation, TriageAssessment
from core.thresholds import ThresholdDecision


class ModelRoutingDecision(BaseModel):
    parser_model: str
    reasoning_model: str
    verifier_model: str
    mode: str


class ResearchReport(BaseModel):
    context: ClinicalDecisionContext
    differential: DifferentialResult
    next_best_tests: list[TestRecommendation]
    triage: TriageAssessment
    threshold_decision: ThresholdDecision
    contradictions: list[str] = Field(default_factory=list)
    provenance_warnings: list[str] = Field(default_factory=list)
    model_route: ModelRoutingDecision
    disclaimer: str = (
        "Research only. PRIORI-X does not diagnose, treat, or autonomously manage patients."
    )

