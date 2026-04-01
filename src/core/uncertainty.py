from __future__ import annotations

import numpy as np

from core.bayes import normalize_distribution
from core.models import ProbabilityInterval


def sample_probability(mean: float, low: float | None, high: float | None, rng: np.random.Generator) -> float:
    if low is None or high is None:
        spread = max(mean * 0.1, 0.02)
        low = max(0.001, mean - spread)
        high = min(0.999, mean + spread)
    std = max((high - low) / 3.92, 0.01)
    return float(np.clip(rng.normal(mean, std), 0.001, 0.999))


def sample_lr(center: float, low: float | None, high: float | None, rng: np.random.Generator) -> float:
    if low is None or high is None:
        low = max(0.05, center / 1.5)
        high = max(low + 0.01, center * 1.5)
    log_mean = np.log(center)
    log_std = max((np.log(high) - np.log(low)) / 3.92, 0.05)
    return float(np.exp(rng.normal(log_mean, log_std)))


def summarize_samples(samples: list[float]) -> ProbabilityInterval:
    array = np.asarray(samples)
    return ProbabilityInterval(
        mean=float(array.mean()),
        low=float(np.quantile(array, 0.1)),
        high=float(np.quantile(array, 0.9)),
    )


def normalize_sampled_distributions(samples: list[dict[str, float]]) -> dict[str, ProbabilityInterval]:
    per_key: dict[str, list[float]] = {}
    for sample in samples:
        normalized = normalize_distribution(sample)
        for key, value in normalized.items():
            per_key.setdefault(key, []).append(value)
    return {key: summarize_samples(values) for key, values in per_key.items()}

