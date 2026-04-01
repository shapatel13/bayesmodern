from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DecisionCostModel(BaseModel):
    benefit_of_treatment: float = Field(gt=0)
    harm_of_treatment: float = Field(gt=0)
    harm_of_missed_disease: float = Field(gt=0)
    harm_of_test: float = Field(gt=0)
    discharge_harm: float = Field(gt=0, default=1.0)
    icu_overuse_harm: float = Field(gt=0, default=0.7)
    urgency_multiplier: float = Field(gt=0, default=1.0)


class DecisionThresholds(BaseModel):
    defer_threshold: float
    test_threshold: float
    treatment_threshold: float
    admit_threshold: float
    icu_threshold: float


class ThresholdDecision(BaseModel):
    action: Literal["observe", "test", "treat", "admit", "icu_consider"]
    clinician_language: str
    plain_language: str


def calculate_thresholds(model: DecisionCostModel) -> DecisionThresholds:
    urgency_adjusted_miss = model.harm_of_missed_disease * model.urgency_multiplier
    test_threshold = model.harm_of_test / (model.benefit_of_treatment + urgency_adjusted_miss)
    treatment_threshold = model.harm_of_treatment / (model.benefit_of_treatment + model.harm_of_treatment)
    defer_threshold = min(test_threshold * 0.5, 0.05)
    admit_threshold = model.discharge_harm / (model.discharge_harm + urgency_adjusted_miss)
    icu_threshold = model.icu_overuse_harm / (model.icu_overuse_harm + urgency_adjusted_miss * 1.25)
    return DecisionThresholds(
        defer_threshold=max(0.01, min(defer_threshold, 0.2)),
        test_threshold=max(0.03, min(test_threshold, 0.6)),
        treatment_threshold=max(0.2, min(treatment_threshold, 0.9)),
        admit_threshold=max(0.1, min(admit_threshold, 0.85)),
        icu_threshold=max(0.15, min(icu_threshold, 0.9)),
    )


def explain_threshold_position(posterior: float, thresholds: DecisionThresholds) -> ThresholdDecision:
    if posterior >= thresholds.icu_threshold:
        return ThresholdDecision(
            action="icu_consider",
            clinician_language="Posterior exceeds ICU escalation threshold given current severity assumptions.",
            plain_language="The current risk is high enough that critical-care level escalation should be considered.",
        )
    if posterior >= thresholds.treatment_threshold:
        return ThresholdDecision(
            action="treat",
            clinician_language="Posterior exceeds treatment threshold; empiric action is favored over further delay.",
            plain_language="The diagnosis looks likely enough that treatment is more reasonable than waiting.",
        )
    if posterior >= thresholds.admit_threshold:
        return ThresholdDecision(
            action="admit",
            clinician_language="Posterior is above the admit threshold even if treatment threshold is not yet crossed.",
            plain_language="The patient is high-risk enough that observation in a higher-acuity setting makes sense.",
        )
    if posterior >= thresholds.test_threshold:
        return ThresholdDecision(
            action="test",
            clinician_language="Posterior lies in the diagnostic workup zone where more information has positive expected value.",
            plain_language="The condition is plausible enough that another test is worth doing now.",
        )
    return ThresholdDecision(
        action="observe",
        clinician_language="Posterior is below testing and treatment thresholds; observation or alternative hypotheses are favored.",
        plain_language="Right now the condition looks unlikely, so watchful waiting or another explanation fits better.",
    )

