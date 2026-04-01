from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SourceType = Literal["hard_coded", "llm_inferred", "user_supplied"]
UrgencyLevel = Literal["routine", "expedited", "urgent", "emergent"]
CalibrationState = Literal["confident", "moderately_uncertain", "fragile"]


class ClinicalFinding(BaseModel):
    key: str
    label: str
    present: bool | None = None
    value: float | None = None
    units: str | None = None
    note: str | None = None
    source_type: SourceType = "user_supplied"


class LikelihoodRatioRange(BaseModel):
    positive_lr: float = Field(gt=0)
    negative_lr: float = Field(gt=0)
    positive_lr_low: float | None = Field(default=None, gt=0)
    positive_lr_high: float | None = Field(default=None, gt=0)
    negative_lr_low: float | None = Field(default=None, gt=0)
    negative_lr_high: float | None = Field(default=None, gt=0)


class FindingContribution(BaseModel):
    finding_key: str
    label: str
    direction: Literal["for", "against", "neutral"]
    applied_lr: float
    rationale: str
    provenance_refs: list[str] = Field(default_factory=list)
    source_type: SourceType = "hard_coded"


class HypothesisEvidence(BaseModel):
    finding_key: str
    label: str
    lr: LikelihoodRatioRange
    rationale: str
    provenance_refs: list[str] = Field(default_factory=list)
    source_type: SourceType = "hard_coded"


class DiagnosisHypothesis(BaseModel):
    slug: str
    name: str
    category: str = "general"
    prior: float = Field(gt=0, lt=1)
    supporting_findings: list[HypothesisEvidence] = Field(default_factory=list)
    contradicting_findings: list[HypothesisEvidence] = Field(default_factory=list)


class DifferentialEntry(BaseModel):
    slug: str
    name: str
    prior: float
    posterior: float
    interval_low: float
    interval_high: float
    evidence_for: list[FindingContribution]
    evidence_against: list[FindingContribution]
    symptom_coverage: float
    explaining_away: list[str] = Field(default_factory=list)
    calibration_state: CalibrationState
    provenance_badges: list[str] = Field(default_factory=list)


class DifferentialResult(BaseModel):
    ranked: list[DifferentialEntry]
    posterior_mass_top3: float
    model_note: str


class CandidateTest(BaseModel):
    slug: str
    name: str
    target_diagnoses: list[str]
    diagnosis_lrs: dict[str, LikelihoodRatioRange]
    direct_cost: float = Field(ge=0)
    downstream_cost: float = Field(ge=0, default=0)
    invasiveness: float = Field(ge=0, le=1, default=0)
    radiation: float = Field(ge=0, le=1, default=0)
    nephrotoxicity: float = Field(ge=0, le=1, default=0)
    bleed_risk: float = Field(ge=0, le=1, default=0)
    logistic_burden: float = Field(ge=0, le=1, default=0)
    actionability: float = Field(ge=0, le=1, default=0.5)
    urgency_modifier: float = Field(ge=0.5, le=2.0, default=1.0)
    bedside: bool = False
    already_done: bool = False
    provenance_refs: list[str] = Field(default_factory=list)
    source_type: SourceType = "hard_coded"


class TestRecommendation(BaseModel):
    slug: str
    name: str
    score: float
    expected_information_gain: float
    expected_posterior_movement: float
    stewardship_score: float
    disposition: Literal["worth_it_now", "defer", "unnecessary", "already_answered"]
    discriminates_between: list[str]
    rationale: str
    lr_plus: float
    lr_minus: float
    provenance_badges: list[str] = Field(default_factory=list)


class ClinicalDecisionContext(BaseModel):
    case_id: str
    specialty: str = "general_internal_medicine"
    findings: list[ClinicalFinding] = Field(default_factory=list)
    completed_tests: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    symptoms_free_text: str | None = None
    age_years: int | None = Field(default=None, ge=0)
    pregnant: bool = False
    renal_impairment: bool = False
    hemodynamic_instability: bool = False
    critical_values_present: bool = False
    safety_mode: str = "conservative"


class ProbabilityInterval(BaseModel):
    mean: float
    low: float
    high: float


class TriageAssessment(BaseModel):
    urgency: UrgencyLevel
    reasons: list[str]
    admit_threshold_crossed: bool
    icu_threshold_crossed: bool


class TestRiskPenalty(BaseModel):
    total_penalty: float
    reasons: list[str]
