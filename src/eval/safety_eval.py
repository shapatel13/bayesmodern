from __future__ import annotations

from agent.trace_schema import ExperimentTrace


def unsafe_recommendation_rate(traces: list[ExperimentTrace]) -> float:
    if not traces:
        return 0.0
    unsafe = 0
    for trace in traces:
        if trace.reward and trace.reward.hard_veto:
            unsafe += 1
    return unsafe / len(traces)

