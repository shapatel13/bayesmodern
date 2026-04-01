from __future__ import annotations

from agent.trace_schema import ExperimentTrace
from core.calibration import brier_score, expected_calibration_error


def calibration_summary(traces: list[ExperimentTrace], gold_lookup: dict[str, str]) -> dict[str, float]:
    probabilities: list[float] = []
    outcomes: list[int] = []
    for trace in traces:
        top = trace.report.differential.ranked[0] if trace.report.differential.ranked else None
        if top is None:
            continue
        gold = gold_lookup.get(trace.task_id)
        probabilities.append(top.posterior)
        outcomes.append(1 if gold == top.slug else 0)
    return {
        "brier_score": brier_score(probabilities, outcomes) if probabilities else 0.0,
        "expected_calibration_error": expected_calibration_error(probabilities, outcomes) if probabilities else 0.0,
    }

