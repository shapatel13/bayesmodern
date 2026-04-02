from __future__ import annotations

from collections import Counter

from agent.trace_schema import ExperimentTrace


def triage_accuracy(traces: list[ExperimentTrace]) -> float:
    eligible = [trace for trace in traces if trace.task_type == "triage" and trace.gold_triage]
    if not eligible:
        return 0.0
    hits = sum(1 for trace in eligible if trace.report.triage.urgency == trace.gold_triage)
    return hits / len(eligible)


def triage_confusion_counts(traces: list[ExperimentTrace]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for trace in traces:
        if trace.task_type != "triage" or not trace.gold_triage:
            continue
        counter[f"{trace.gold_triage}->{trace.report.triage.urgency}"] += 1
    return dict(counter)
