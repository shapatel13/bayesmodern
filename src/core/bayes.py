from __future__ import annotations

from collections.abc import Iterable, Mapping


EPSILON = 1e-6


def clamp_probability(probability: float) -> float:
    return min(max(probability, EPSILON), 1.0 - EPSILON)


def probability_to_odds(probability: float) -> float:
    bounded = clamp_probability(probability)
    return bounded / (1.0 - bounded)


def odds_to_probability(odds: float) -> float:
    if odds <= 0:
        return EPSILON
    return clamp_probability(odds / (1.0 + odds))


def bayes_update(prior: float, likelihood_ratio: float) -> float:
    if likelihood_ratio <= 0:
        raise ValueError("Likelihood ratio must be positive.")
    return odds_to_probability(probability_to_odds(prior) * likelihood_ratio)


def sequential_bayes_update(prior: float, likelihood_ratios: Iterable[float]) -> float:
    posterior = prior
    for lr in likelihood_ratios:
        posterior = bayes_update(posterior, lr)
    return posterior


def normalize_distribution(raw: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in raw.values())
    if total <= 0:
        raise ValueError("Distribution must contain positive mass.")
    return {key: max(value, 0.0) / total for key, value in raw.items()}


def shannon_entropy(probabilities: Mapping[str, float]) -> float:
    import math

    entropy = 0.0
    for probability in probabilities.values():
        bounded = clamp_probability(probability)
        entropy -= bounded * math.log2(bounded)
    return entropy

