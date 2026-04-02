from __future__ import annotations

import math
from collections.abc import Mapping

from core.models import DiagnosisHypothesis, MechanismStateResult


DISEASE_MECHANISM_WEIGHTS: dict[str, dict[str, float]] = {
    "pe": {
        "obstructive_physiology": 0.65,
        "thrombotic_ischemic_tendency": 0.45,
        "low_effective_arterial_volume": 0.12,
    },
    "pneumonia": {
        "distributive_physiology": 0.55,
        "venous_congestion": -0.18,
    },
    "heart_failure": {
        "venous_congestion": 0.7,
        "impaired_contractility": 0.55,
        "fluid_intolerance": 0.4,
        "low_effective_arterial_volume": -0.22,
    },
    "acs": {
        "thrombotic_ischemic_tendency": 0.7,
        "impaired_contractility": 0.35,
        "hemorrhagic_tendency_active_blood_loss": -0.18,
    },
    "upper_gi_bleed": {
        "hemorrhagic_tendency_active_blood_loss": 0.82,
        "low_effective_arterial_volume": 0.35,
        "medication_toxicity_effect": 0.4,
        "thrombotic_ischemic_tendency": -0.15,
    },
}


def _mechanism_signal(mechanism_result: MechanismStateResult) -> Mapping[str, float]:
    return {
        estimate.slug: estimate.posterior - estimate.prior
        for estimate in mechanism_result.ranked
    }


def apply_mechanism_coupling(
    hypotheses: list[DiagnosisHypothesis],
    mechanism_result: MechanismStateResult,
) -> list[DiagnosisHypothesis]:
    mechanism_signal = _mechanism_signal(mechanism_result)
    adjusted: list[DiagnosisHypothesis] = []
    for hypothesis in hypotheses:
        coupling_weights = DISEASE_MECHANISM_WEIGHTS.get(hypothesis.slug, {})
        weighted_signal = sum(
            coupling_weights.get(state_slug, 0.0) * mechanism_signal.get(state_slug, 0.0)
            for state_slug in coupling_weights
        )
        multiplier = math.exp(max(-0.55, min(0.75, weighted_signal)))
        adjusted_prior = min(max(hypothesis.prior * multiplier, 0.02), 0.85)
        adjusted.append(hypothesis.model_copy(update={"prior": adjusted_prior}))
    return adjusted

