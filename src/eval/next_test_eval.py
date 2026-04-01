from __future__ import annotations

from agent.trace_schema import ExperimentTrace


def next_test_hit_rate(traces: list[ExperimentTrace]) -> float:
    scored = [trace for trace in traces if trace.reward]
    if not scored:
        return 0.0
    return sum(trace.reward.next_test_quality for trace in scored if trace.reward) / len(scored)

