from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from llm.structured_output import ResearchReport


FailureCategory = Literal[
    "wrong_primary_diagnosis",
    "unsafe_recommendation",
    "overtesting",
    "undertesting",
    "cost_insensitive",
    "calibration_failure",
    "urgency_failure",
    "medication_safety_failure",
    "generation_audit_failure",
    "unsupported_evidence_claim",
    "malformed_json",
    "contradiction",
]


class TraceStep(BaseModel):
    name: str
    detail: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardBreakdown(BaseModel):
    diagnostic_correctness: float
    topk_differential_quality: float
    calibration_quality: float
    next_test_quality: float
    stewardship: float
    safety: float
    urgency: float
    provenance: float
    json_validity: float
    consistency: float
    medication_extraction_quality: float | None = None
    adverse_event_quality: float | None = None
    claim_alignment_quality: float | None = None
    generation_audit_quality: float | None = None
    total_reward: float
    reward_profile: str = "diagnostic"
    component_weights: dict[str, float] = Field(default_factory=dict)
    hard_veto: bool = False
    veto_reasons: list[str] = Field(default_factory=list)
    failure_categories: list[FailureCategory] = Field(default_factory=list)


class ExperimentTrace(BaseModel):
    task_id: str
    source_dataset: str
    task_type: str
    task_metadata: dict[str, Any] = Field(default_factory=dict)
    gold_diagnosis: str | None = None
    acceptable_tests: list[str] = Field(default_factory=list)
    gold_triage: str | None = None
    prompt_version: str
    policy_version: str
    model_route: str
    steps: list[TraceStep]
    report: ResearchReport
    reward: RewardBreakdown | None = None


class LightningTransition(BaseModel):
    task_id: str
    state: dict[str, Any]
    action: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
