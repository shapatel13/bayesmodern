from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ClinicianStyle = Literal["conservative", "balanced", "aggressive"]


class UtilityProfile(BaseModel):
    mortality_weight: float = Field(default=0.45, ge=0)
    morbidity_weight: float = Field(default=0.25, ge=0)
    stewardship_weight: float = Field(default=0.15, ge=0)
    patient_preference_weight: float = Field(default=0.15, ge=0)
    clinician_style: ClinicianStyle = "balanced"


class StrategyOutcome(BaseModel):
    name: str
    mortality_risk: float = Field(ge=0, le=1)
    morbidity_risk: float = Field(ge=0, le=1)
    resource_cost: float = Field(ge=0)
    patient_preference_alignment: float = Field(ge=0, le=1)


def style_multiplier(style: ClinicianStyle) -> float:
    if style == "conservative":
        return 1.15
    if style == "aggressive":
        return 0.9
    return 1.0


def expected_utility(outcome: StrategyOutcome, profile: UtilityProfile) -> float:
    stewardship_penalty = min(outcome.resource_cost / 10_000.0, 1.0)
    caution = style_multiplier(profile.clinician_style)
    utility = (
        (1.0 - outcome.mortality_risk) * profile.mortality_weight
        + (1.0 - outcome.morbidity_risk) * profile.morbidity_weight
        + (1.0 - stewardship_penalty) * profile.stewardship_weight
        + outcome.patient_preference_alignment * profile.patient_preference_weight
    )
    return utility / caution


def rank_strategies(outcomes: list[StrategyOutcome], profile: UtilityProfile) -> list[tuple[str, float]]:
    scored = [(outcome.name, expected_utility(outcome, profile)) for outcome in outcomes]
    return sorted(scored, key=lambda item: item[1], reverse=True)

