from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.models import ClinicalDecisionContext, DifferentialResult, MechanismStateResult, TestRecommendation, TriageAssessment
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


class ReasoningRuntimeTrace(BaseModel):
    mode: Literal["curated_only", "hybrid_open_world"] = "curated_only"
    open_world_considered: bool = False
    open_world_triggered: bool = False
    gate_reason: str = "Runtime trace unavailable."
    base_top_diagnosis: str | None = None
    base_top_posterior: float | None = None
    final_top_diagnosis: str | None = None
    final_top_posterior: float | None = None
    open_world_hypothesis_count: int = 0
    open_world_test_count: int = 0
    notes: list[str] = Field(default_factory=list)
    special_reasoning_notes: list[str] = Field(default_factory=list)
    test_dependency_notes: list[str] = Field(default_factory=list)


class DecisionQualityAssessment(BaseModel):
    needs_clinician_review: bool = False
    reasons: list[str] = Field(default_factory=list)
    structured_signal_count: int = 0
    top_differential_gap: float = 0.0
    low_signal_case: bool = False
    broad_differential: bool = False
    mixed_mechanism_uncertainty: bool = False


class ResearchReport(BaseModel):
    context: ClinicalDecisionContext
    differential: DifferentialResult
    mechanism_states: MechanismStateResult = Field(default_factory=MechanismStateResult)
    next_best_tests: list[TestRecommendation]
    triage: TriageAssessment
    threshold_decision: ThresholdDecision
    reasoning_runtime: ReasoningRuntimeTrace = Field(default_factory=ReasoningRuntimeTrace)
    decision_quality: DecisionQualityAssessment = Field(default_factory=DecisionQualityAssessment)
    contradictions: list[str] = Field(default_factory=list)
    provenance_warnings: list[str] = Field(default_factory=list)
    generation_audit: GenerationAuditResult | None = None
    model_route: ModelRoutingDecision
    disclaimer: str = (
        "Research only. PRIORI-X does not diagnose, treat, or autonomously manage patients."
    )
