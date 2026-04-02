from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.models import ClinicalDecisionContext, DifferentialResult, TestRecommendation, TriageAssessment
from core.thresholds import ThresholdDecision


class ModelRoutingDecision(BaseModel):
    parser_model: str
    reasoning_model: str
    verifier_model: str
    mode: str


class GenerationAuditResult(BaseModel):
    predicted_risk_grade: int = Field(ge=1, le=4)
    recommended_action: Literal["accept", "manual_review", "block"]
    issue_types: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0, le=1, default=0.5)
    reference_available: bool = False
    physician_reference_available: bool = False


class ResearchReport(BaseModel):
    context: ClinicalDecisionContext
    differential: DifferentialResult
    next_best_tests: list[TestRecommendation]
    triage: TriageAssessment
    threshold_decision: ThresholdDecision
    contradictions: list[str] = Field(default_factory=list)
    provenance_warnings: list[str] = Field(default_factory=list)
    generation_audit: GenerationAuditResult | None = None
    model_route: ModelRoutingDecision
    disclaimer: str = (
        "Research only. PRIORI-X does not diagnose, treat, or autonomously manage patients."
    )
