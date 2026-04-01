from __future__ import annotations

from pathlib import Path

from agent.orchestrator import PRIORIXOrchestrator
from agent.reward_model import CompositeRewardModel
from agent.trace_schema import ExperimentTrace, TraceStep
from eval.calibration_eval import calibration_summary
from eval.error_analysis import summarize_failure_categories
from eval.next_test_eval import next_test_hit_rate
from eval.safety_eval import unsafe_recommendation_rate
from priorix_tasks.common import BenchmarkTask
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
                source_dataset=task.source_dataset,
                task_type=task.task_type,
                gold_diagnosis=task.gold_diagnosis,
                acceptable_tests=task.acceptable_tests,
                gold_triage=task.gold_triage,
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


def default_demo_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="sample-pe-1",
            source_dataset="synthetic",
            split="test",
            task_type="diagnosis_open",
            prompt="Pleuritic chest pain with tachycardia and hypoxemia after immobility. No fever.",
            gold_diagnosis="pe",
            acceptable_tests=["d_dimer", "cta_pe"],
            gold_triage="urgent",
        ),
        BenchmarkTask(
            task_id="sample-hf-1",
            source_dataset="synthetic",
            split="test",
            task_type="diagnosis_open",
            prompt="Orthopnea, crackles, leg edema, and progressive dyspnea over several days.",
            gold_diagnosis="heart_failure",
            acceptable_tests=["bnp", "cxr"],
            gold_triage="expedited",
        ),
    ]


def build_markdown_report(traces: list[ExperimentTrace]) -> str:
    calibration = calibration_summary(traces)
    lines = [
        "# PRIORI-X Benchmark Report",
        "",
        f"- Cases: {len(traces)}",
        f"- Unsafe recommendation rate: {unsafe_recommendation_rate(traces):.2%}",
        f"- Next-best-test hit rate: {next_test_hit_rate(traces):.2%}",
        f"- Brier score: {calibration['brier_score']:.3f}",
        f"- ECE: {calibration['expected_calibration_error']:.3f}",
        "",
        "## Case Snapshots",
    ]
    for trace in traces:
        top = trace.report.differential.ranked[0] if trace.report.differential.ranked else None
        next_test = trace.report.next_best_tests[0].name if trace.report.next_best_tests else "None"
        lines.append(
            f"- `{trace.task_id}`: top diagnosis `{top.slug if top else 'n/a'}`, triage `{trace.report.triage.urgency}`, next test `{next_test}`, reward `{trace.reward.total_reward if trace.reward else 0.0:.2f}`"
        )
    lines.extend(
        [
            "",
        "## Failure Categories",
        ]
    )
    for key, value in summarize_failure_categories(traces).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    output_dir = Path("artifacts/evals")
    traces = run_benchmark(default_demo_tasks(), output_dir=output_dir)
    report = build_markdown_report(traces)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_benchmark_report.md").write_text(report, encoding="utf-8")
    print(report)
