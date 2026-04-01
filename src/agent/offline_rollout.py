from __future__ import annotations

from pathlib import Path

from agent.trace_schema import ExperimentTrace
from datasets.task_builders.common import BenchmarkTask
from eval.benchmark_runner import build_markdown_report, run_benchmark


def run_offline_rollout(tasks: list[BenchmarkTask], artifacts_dir: Path) -> tuple[list[ExperimentTrace], str]:
    traces = run_benchmark(tasks, output_dir=artifacts_dir)
    report = build_markdown_report(traces)
    (artifacts_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    return traces, report

