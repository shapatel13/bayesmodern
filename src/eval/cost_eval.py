from __future__ import annotations

from agent.trace_schema import ExperimentTrace


def average_recommended_test_cost(traces: list[ExperimentTrace]) -> float:
    costs: list[float] = []
    for trace in traces:
        for recommendation in trace.report.next_best_tests[:1]:
            costs.append(recommendation.direct_cost + recommendation.downstream_cost)
    return sum(costs) / len(costs) if costs else 0.0
