from __future__ import annotations

from agent.trace_schema import ExperimentTrace


def evaluate_topk_recall(traces: list[ExperimentTrace], k: int = 3) -> float:
    tasks_with_gold = [trace for trace in traces if trace.reward and "wrong_primary_diagnosis" not in trace.reward.failure_categories]
    if not tasks_with_gold:
        return 0.0
    successes = 0
    for trace in tasks_with_gold:
        if trace.reward and trace.reward.topk_differential_quality > 0:
            successes += 1
    return successes / len(tasks_with_gold)

