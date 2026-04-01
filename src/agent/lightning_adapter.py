from __future__ import annotations

from pathlib import Path

from agent.trace_schema import ExperimentTrace, LightningTransition
from utils.jsonx import dumps_pretty


def traces_to_lightning_transitions(traces: list[ExperimentTrace]) -> list[LightningTransition]:
    transitions: list[LightningTransition] = []
    for trace in traces:
        top = trace.report.differential.ranked[0] if trace.report.differential.ranked else None
        recommendation = trace.report.next_best_tests[0] if trace.report.next_best_tests else None
        transitions.append(
            LightningTransition(
                task_id=trace.task_id,
                state={
                    "prompt_version": trace.prompt_version,
                    "policy_version": trace.policy_version,
                    "triage": trace.report.triage.urgency,
                },
                action={
                    "top_diagnosis": top.slug if top else None,
                    "recommended_test": recommendation.slug if recommendation else None,
                },
                reward=trace.reward.total_reward if trace.reward else 0.0,
                done=True,
                info={
                    "failure_categories": trace.reward.failure_categories if trace.reward else [],
                    "hard_veto": trace.reward.hard_veto if trace.reward else False,
                },
            )
        )
    return transitions


def export_lightning_jsonl(traces: list[ExperimentTrace], output_path: Path) -> None:
    transitions = traces_to_lightning_transitions(traces)
    payload = "\n".join(dumps_pretty(transition.model_dump()) for transition in transitions)
    output_path.write_text(payload, encoding="utf-8")

