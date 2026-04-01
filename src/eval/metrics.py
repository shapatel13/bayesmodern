from __future__ import annotations

from collections.abc import Sequence
from math import log


def exact_match_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    matches = [prediction == target for prediction, target in zip(predictions, targets, strict=True)]
    return sum(matches) / len(matches) if matches else 0.0


def top_k_recall(rankings: Sequence[Sequence[str]], golds: Sequence[str], k: int) -> float:
    hits = [gold in ranking[:k] for ranking, gold in zip(rankings, golds, strict=True)]
    return sum(hits) / len(hits) if hits else 0.0


def average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_log_loss(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    eps = 1e-6
    terms = []
    for probability, outcome in zip(probabilities, outcomes, strict=True):
        bounded = min(max(probability, eps), 1 - eps)
        terms.append(-(outcome * log(bounded) + (1 - outcome) * log(1 - bounded)))
    return average(terms)

