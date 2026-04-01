from __future__ import annotations

from agent.trace_schema import ExperimentTrace


def average_recommended_test_cost(traces: list[ExperimentTrace]) -> float:
    costs: list[float] = []
    for trace in traces:
        if trace.report.next_best_tests:
            top_test = trace.report.next_best_tests[0]
            if "cost=" in top_test.rationale:
                continue
        for recommendation in trace.report.next_best_tests[:1]:
            costs.append(recommendation.score)
    return sum(costs) / len(costs) if costs else 0.0

