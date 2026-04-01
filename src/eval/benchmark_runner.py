from __future__ import annotations

from pathlib import Path

from agent.orchestrator import PRIORIXOrchestrator
from agent.reward_model import CompositeRewardModel
from agent.trace_schema import ExperimentTrace, TraceStep
from datasets.task_builders.common import BenchmarkTask
from eval.calibration_eval import calibration_summary
from eval.error_analysis import summarize_failure_categories
from eval.next_test_eval import next_test_hit_rate
from eval.safety_eval import unsafe_recommendation_rate
from utils.jsonx import dumps_pretty


def run_benchmark(tasks: list[BenchmarkTask], output_dir: Path | None = None) -> list[ExperimentTrace]:
    orchestrator = PRIORIXOrchestrator()
    reward_model = CompositeRewardModel()
    traces: list[ExperimentTrace] = []
    for task in tasks:
        report = orchestrator.analyze_text_case(task.task_id, task.prompt)
        reward = reward_model.score(task, report)
        traces.append(
            ExperimentTrace(
                task_id=task.task_id,
                prompt_version="v1-offline",
                policy_version="v1-deterministic",
                model_route=report.model_route.mode,
                steps=[
                    TraceStep(name="extract", detail="Keyword-based structured extraction"),
                    TraceStep(name="differential", detail="Deterministic Bayesian differential"),
                    TraceStep(name="next_test", detail="Stewardship-aware next-best-test ranking"),
                    TraceStep(name="validate", detail="Contradiction and citation checks"),
                ],
                report=report,
                reward=reward,
            )
        )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "benchmark_traces.json").write_text(dumps_pretty([trace.model_dump() for trace in traces]), encoding="utf-8")
    return traces


def build_markdown_report(traces: list[ExperimentTrace]) -> str:
    gold_lookup = {trace.task_id: trace.report.differential.ranked[0].slug for trace in traces if trace.report.differential.ranked}
    calibration = calibration_summary(traces, gold_lookup)
    lines = [
        "# PRIORI-X Benchmark Report",
        "",
        f"- Cases: {len(traces)}",
        f"- Unsafe recommendation rate: {unsafe_recommendation_rate(traces):.2%}",
        f"- Next-best-test hit rate: {next_test_hit_rate(traces):.2%}",
        f"- Brier score: {calibration['brier_score']:.3f}",
        f"- ECE: {calibration['expected_calibration_error']:.3f}",
        "",
        "## Failure Categories",
    ]
    for key, value in summarize_failure_categories(traces).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)

