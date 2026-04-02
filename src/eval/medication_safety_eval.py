from __future__ import annotations

from agent.trace_schema import ExperimentTrace


def _normalized(items: list[str]) -> set[str]:
    return {item.strip().lower() for item in items if item.strip()}


def medication_extraction_recall(traces: list[ExperimentTrace]) -> float:
    relevant = [trace for trace in traces if trace.task_type == "medication_safety"]
    if not relevant:
        return 0.0

    scores: list[float] = []
    for trace in relevant:
        gold_medications = _normalized(list(trace.task_metadata.get("gold_medications", [])))
        predicted_medications = _normalized(trace.report.context.medications)
        if not gold_medications:
            scores.append(1.0)
            continue
        scores.append(len(gold_medications & predicted_medications) / len(gold_medications))
    return sum(scores) / len(scores)


def medication_safety_summary(traces: list[ExperimentTrace]) -> dict[str, float]:
    relevant = [trace for trace in traces if trace.task_type == "medication_safety"]
    if not relevant:
        return {
            "medication_recall": 0.0,
            "adverse_event_recall": 0.0,
        }

    medication_scores: list[float] = []
    adverse_event_scores: list[float] = []
    for trace in relevant:
        gold_medications = _normalized(list(trace.task_metadata.get("gold_medications", [])))
        gold_adverse_events = _normalized(list(trace.task_metadata.get("gold_adverse_events", [])))
        predicted_medications = _normalized(trace.report.context.medications)
        predicted_adverse_events = _normalized(trace.report.context.adverse_events)

        medication_scores.append(1.0 if not gold_medications else len(predicted_medications & gold_medications) / len(gold_medications))
        adverse_event_scores.append(1.0 if not gold_adverse_events else len(predicted_adverse_events & gold_adverse_events) / len(gold_adverse_events))

    return {
        "medication_recall": sum(medication_scores) / len(medication_scores),
        "adverse_event_recall": sum(adverse_event_scores) / len(adverse_event_scores),
    }
