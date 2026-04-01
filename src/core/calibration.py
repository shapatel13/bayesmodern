from __future__ import annotations

from collections.abc import Sequence
from math import fabs


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    if len(probabilities) != len(outcomes):
        raise ValueError("Probabilities and outcomes must have the same length.")
    return sum((probability - outcome) ** 2 for probability, outcome in zip(probabilities, outcomes, strict=True)) / len(
        probabilities
    )


def expected_calibration_error(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    bins: int = 10,
) -> float:
    if len(probabilities) != len(outcomes):
        raise ValueError("Probabilities and outcomes must have the same length.")
    bucket_totals = [0 for _ in range(bins)]
    bucket_probs = [0.0 for _ in range(bins)]
    bucket_outcomes = [0.0 for _ in range(bins)]

    for probability, outcome in zip(probabilities, outcomes, strict=True):
        bucket = min(int(probability * bins), bins - 1)
        bucket_totals[bucket] += 1
        bucket_probs[bucket] += probability
        bucket_outcomes[bucket] += outcome

    total = len(probabilities)
    ece = 0.0
    for index in range(bins):
        if bucket_totals[index] == 0:
            continue
        avg_prob = bucket_probs[index] / bucket_totals[index]
        avg_outcome = bucket_outcomes[index] / bucket_totals[index]
        ece += (bucket_totals[index] / total) * fabs(avg_prob - avg_outcome)
    return ece


def classify_calibration(top_probability: float, interval_width: float) -> str:
    if top_probability >= 0.75 and interval_width <= 0.15:
        return "confident"
    if top_probability >= 0.45 and interval_width <= 0.3:
        return "moderately_uncertain"
    return "fragile"

