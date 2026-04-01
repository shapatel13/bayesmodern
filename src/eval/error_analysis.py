from __future__ import annotations

from collections import Counter

from agent.trace_schema import ExperimentTrace


def summarize_failure_categories(traces: list[ExperimentTrace]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for trace in traces:
        if trace.reward:
            counter.update(trace.reward.failure_categories)
    return dict(counter)

