from __future__ import annotations

from pydantic import BaseModel, Field


class DifferentialPolicy(BaseModel):
    dangerous_context_boost: float = Field(default=0.18, ge=0.0, le=1.0)
    coverage_bonus_strength: float = Field(default=0.12, ge=0.0, le=1.0)
    contradiction_penalty_strength: float = Field(default=0.18, ge=0.0, le=1.0)
    overlap_penalty_strength: float = Field(default=0.15, ge=0.0, le=1.0)
    sample_count: int = Field(default=500, ge=100, le=5000)


class NextTestPolicy(BaseModel):
    movement_weight: float = Field(default=0.9, ge=0.0, le=3.0)
    entropy_weight: float = Field(default=0.5, ge=0.0, le=3.0)
    discrimination_weight: float = Field(default=0.7, ge=0.0, le=3.0)
    threshold_weight: float = Field(default=0.8, ge=0.0, le=3.0)
    actionability_weight: float = Field(default=0.7, ge=0.0, le=3.0)
    cost_weight: float = Field(default=1.8, ge=0.0, le=5.0)
    risk_weight: float = Field(default=1.5, ge=0.0, le=5.0)
    urgency_bonus_weight: float = Field(default=0.25, ge=0.0, le=2.0)
    worth_it_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    defer_threshold: float = Field(default=0.04, ge=0.0, le=1.0)


class ThresholdPolicy(BaseModel):
    benefit_of_treatment: float = Field(default=10.0, gt=0)
    harm_of_treatment: float = Field(default=2.0, gt=0)
    harm_of_missed_disease: float = Field(default=9.0, gt=0)
    harm_of_test: float = Field(default=1.0, gt=0)
    discharge_harm: float = Field(default=2.5, gt=0)
    icu_overuse_harm: float = Field(default=0.7, gt=0)
    urgent_multiplier: float = Field(default=1.5, gt=0)
    expedited_multiplier: float = Field(default=1.15, gt=0)


class ReasoningPolicy(BaseModel):
    version: str
    label: str
    description: str
    differential: DifferentialPolicy = Field(default_factory=DifferentialPolicy)
    next_test: NextTestPolicy = Field(default_factory=NextTestPolicy)
    thresholds: ThresholdPolicy = Field(default_factory=ThresholdPolicy)


POLICY_REGISTRY: dict[str, ReasoningPolicy] = {
    "v1-deterministic": ReasoningPolicy(
        version="v1-deterministic",
        label="Deterministic Baseline",
        description="Balanced starter policy with moderate safety and stewardship pressure.",
    ),
    "v1-balanced-bayesian": ReasoningPolicy(
        version="v1-balanced-bayesian",
        label="Balanced Bayesian",
        description="Stronger evidence coverage and threshold-aware test ranking for general use.",
        differential=DifferentialPolicy(
            dangerous_context_boost=0.22,
            coverage_bonus_strength=0.18,
            contradiction_penalty_strength=0.22,
            overlap_penalty_strength=0.18,
            sample_count=700,
        ),
        next_test=NextTestPolicy(
            movement_weight=1.0,
            entropy_weight=0.8,
            discrimination_weight=0.95,
            threshold_weight=1.0,
            actionability_weight=0.8,
            cost_weight=1.6,
            risk_weight=1.5,
            urgency_bonus_weight=0.3,
            worth_it_threshold=0.11,
            defer_threshold=0.05,
        ),
    ),
    "v1-conservative-safety": ReasoningPolicy(
        version="v1-conservative-safety",
        label="Conservative Safety",
        description="Heavier dangerous-disease prior lift and higher penalties for risky or costly tests.",
        differential=DifferentialPolicy(
            dangerous_context_boost=0.3,
            coverage_bonus_strength=0.14,
            contradiction_penalty_strength=0.25,
            overlap_penalty_strength=0.16,
            sample_count=700,
        ),
        next_test=NextTestPolicy(
            movement_weight=0.9,
            entropy_weight=0.6,
            discrimination_weight=0.7,
            threshold_weight=1.15,
            actionability_weight=0.8,
            cost_weight=2.0,
            risk_weight=2.2,
            urgency_bonus_weight=0.4,
            worth_it_threshold=0.12,
            defer_threshold=0.06,
        ),
        thresholds=ThresholdPolicy(
            benefit_of_treatment=10.0,
            harm_of_treatment=2.0,
            harm_of_missed_disease=10.5,
            harm_of_test=1.1,
            discharge_harm=3.0,
            icu_overuse_harm=0.75,
            urgent_multiplier=1.7,
            expedited_multiplier=1.2,
        ),
    ),
    "v1-stewardship": ReasoningPolicy(
        version="v1-stewardship",
        label="Stewardship First",
        description="Lower-cost, lower-risk test sequencing with tighter disposition thresholds.",
        differential=DifferentialPolicy(
            dangerous_context_boost=0.16,
            coverage_bonus_strength=0.15,
            contradiction_penalty_strength=0.2,
            overlap_penalty_strength=0.2,
            sample_count=600,
        ),
        next_test=NextTestPolicy(
            movement_weight=0.85,
            entropy_weight=0.6,
            discrimination_weight=0.75,
            threshold_weight=0.85,
            actionability_weight=0.75,
            cost_weight=2.4,
            risk_weight=2.0,
            urgency_bonus_weight=0.22,
            worth_it_threshold=0.13,
            defer_threshold=0.06,
        ),
    ),
    "v1-sensitive-triage": ReasoningPolicy(
        version="v1-sensitive-triage",
        label="Sensitive Triage",
        description="Boosts dangerous hypotheses under instability and values decisive threshold movement.",
        differential=DifferentialPolicy(
            dangerous_context_boost=0.35,
            coverage_bonus_strength=0.17,
            contradiction_penalty_strength=0.2,
            overlap_penalty_strength=0.14,
            sample_count=750,
        ),
        next_test=NextTestPolicy(
            movement_weight=1.0,
            entropy_weight=0.65,
            discrimination_weight=0.85,
            threshold_weight=1.25,
            actionability_weight=0.85,
            cost_weight=1.5,
            risk_weight=1.4,
            urgency_bonus_weight=0.45,
            worth_it_threshold=0.09,
            defer_threshold=0.035,
        ),
        thresholds=ThresholdPolicy(
            benefit_of_treatment=10.5,
            harm_of_treatment=2.0,
            harm_of_missed_disease=11.0,
            harm_of_test=1.0,
            discharge_harm=3.2,
            icu_overuse_harm=0.8,
            urgent_multiplier=1.85,
            expedited_multiplier=1.25,
        ),
    ),
}


def get_reasoning_policy(version: str | None) -> ReasoningPolicy:
    normalized = (version or "v1-deterministic").strip().lower()
    if normalized not in POLICY_REGISTRY:
        raise KeyError(f"Unsupported reasoning policy: {version}")
    return POLICY_REGISTRY[normalized]


def list_reasoning_policies() -> list[ReasoningPolicy]:
    return list(POLICY_REGISTRY.values())
