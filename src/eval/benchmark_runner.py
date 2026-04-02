from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agent.orchestrator import PRIORIXOrchestrator
from agent.reward_model import CompositeRewardModel
from agent.trace_schema import ExperimentTrace, TraceStep
from datasets.loader import load_benchmark_tasks
from eval.calibration_eval import calibration_summary
from eval.error_analysis import summarize_failure_categories
from eval.medication_safety_eval import medication_safety_summary
from eval.metrics import average, safe_log_loss, top_k_recall
from eval.next_test_eval import next_test_hit_rate
from eval.safety_eval import unsafe_recommendation_rate
from eval.triage_eval import triage_accuracy
from priorix_tasks.common import BenchmarkTask
from utils.jsonx import dumps_pretty


class BenchmarkMetricsSummary(BaseModel):
    cases: int
    mean_reward: float
    hard_veto_count: int
    top1_differential_recall: float
    top3_differential_recall: float
    next_best_test_hit_rate: float
    unsafe_recommendation_rate: float
    unsupported_claim_rate: float
    contradiction_rate: float
    urgency_accuracy: float
    medication_recall: float = 0.0
    adverse_event_recall: float = 0.0
    brier_score: float
    expected_calibration_error: float
    log_loss: float
    failure_categories: dict[str, int] = Field(default_factory=dict)
    top_diagnoses: list[str] = Field(default_factory=list)


def run_benchmark(
    tasks: list[BenchmarkTask],
    output_dir: Path | None = None,
    *,
    prompt_version: str = "v1-offline",
    policy_version: str = "v1-deterministic",
) -> list[ExperimentTrace]:
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
                task_metadata=task.metadata,
                gold_diagnosis=task.gold_diagnosis,
                acceptable_tests=task.acceptable_tests,
                gold_triage=task.gold_triage,
                prompt_version=prompt_version,
                policy_version=policy_version,
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
        (output_dir / "benchmark_summary.json").write_text(
            dumps_pretty(summarize_benchmark(traces).model_dump()),
            encoding="utf-8",
        )
    return traces


def run_dataset_benchmark(
    dataset_key: str,
    *,
    split: str | None = None,
    subset: str | None = None,
    limit: int | None = None,
    output_dir: Path | None = None,
    prompt_version: str = "v1-offline",
    policy_version: str = "v1-deterministic",
) -> list[ExperimentTrace]:
    tasks = load_benchmark_tasks(dataset_key, split=split, subset=subset, limit=limit)
    return run_benchmark(
        tasks,
        output_dir=output_dir,
        prompt_version=prompt_version,
        policy_version=policy_version,
    )


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


def summarize_benchmark(traces: list[ExperimentTrace]) -> BenchmarkMetricsSummary:
    ranked_lists = [
        [entry.slug for entry in trace.report.differential.ranked]
        for trace in traces
        if trace.gold_diagnosis and trace.report.differential.ranked
    ]
    gold_diagnoses = [trace.gold_diagnosis for trace in traces if trace.gold_diagnosis and trace.report.differential.ranked]
    top_probabilities = [trace.report.differential.ranked[0].posterior for trace in traces if trace.report.differential.ranked and trace.gold_diagnosis]
    top_outcomes = [
        1 if trace.report.differential.ranked[0].slug == trace.gold_diagnosis else 0
        for trace in traces
        if trace.report.differential.ranked and trace.gold_diagnosis
    ]
    rewards = [trace.reward.total_reward for trace in traces if trace.reward]
    supported_urgency = [trace for trace in traces if trace.gold_triage]
    top_diagnoses = [
        trace.report.differential.ranked[0].slug
        for trace in traces
        if trace.report.differential.ranked
    ]
    calibration = calibration_summary(traces)
    med_safety = medication_safety_summary(traces)

    return BenchmarkMetricsSummary(
        cases=len(traces),
        mean_reward=average(rewards),
        hard_veto_count=sum(1 for trace in traces if trace.reward and trace.reward.hard_veto),
        top1_differential_recall=top_k_recall(ranked_lists, gold_diagnoses, k=1) if gold_diagnoses else 0.0,
        top3_differential_recall=top_k_recall(ranked_lists, gold_diagnoses, k=3) if gold_diagnoses else 0.0,
        next_best_test_hit_rate=next_test_hit_rate(traces),
        unsafe_recommendation_rate=unsafe_recommendation_rate(traces),
        unsupported_claim_rate=sum(1 for trace in traces if trace.report.provenance_warnings) / len(traces) if traces else 0.0,
        contradiction_rate=sum(1 for trace in traces if trace.report.contradictions) / len(traces) if traces else 0.0,
        urgency_accuracy=triage_accuracy(traces) if supported_urgency else 0.0,
        medication_recall=med_safety["medication_recall"],
        adverse_event_recall=med_safety["adverse_event_recall"],
        brier_score=calibration["brier_score"],
        expected_calibration_error=calibration["expected_calibration_error"],
        log_loss=safe_log_loss(top_probabilities, top_outcomes) if top_probabilities else 0.0,
        failure_categories=summarize_failure_categories(traces),
        top_diagnoses=top_diagnoses,
    )


def build_markdown_report(traces: list[ExperimentTrace]) -> str:
    summary = summarize_benchmark(traces)
    lines = [
        "# PRIORI-X Benchmark Report",
        "",
        f"- Cases: {summary.cases}",
        f"- Mean reward: {summary.mean_reward:.3f}",
        f"- Unsafe recommendation rate: {summary.unsafe_recommendation_rate:.2%}",
        f"- Next-best-test hit rate: {summary.next_best_test_hit_rate:.2%}",
        f"- Top-1 differential recall: {summary.top1_differential_recall:.2%}",
        f"- Top-3 differential recall: {summary.top3_differential_recall:.2%}",
        f"- Brier score: {summary.brier_score:.3f}",
        f"- ECE: {summary.expected_calibration_error:.3f}",
        f"- Log loss: {summary.log_loss:.3f}",
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
    for key, value in summary.failure_categories.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    output_dir = Path("artifacts/evals")
    traces = run_benchmark(default_demo_tasks(), output_dir=output_dir)
    report = build_markdown_report(traces)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_benchmark_report.md").write_text(report, encoding="utf-8")
    print(report)
