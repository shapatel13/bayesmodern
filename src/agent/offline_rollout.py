from __future__ import annotations

from pathlib import Path

from agent.lightning_adapter import export_lightning_bundle
from agent.trace_schema import ExperimentTrace
from datasets.loader import load_train_validation_tasks
from eval.benchmark_runner import build_markdown_report, run_benchmark
from priorix_tasks.common import BenchmarkTask
from utils.config import Settings, get_settings


def run_offline_rollout(
    tasks: list[BenchmarkTask],
    artifacts_dir: Path,
    *,
    validation_tasks: list[BenchmarkTask] | None = None,
    settings: Settings | None = None,
) -> tuple[list[ExperimentTrace], str]:
    settings = settings or get_settings()
    traces = run_benchmark(tasks, output_dir=artifacts_dir)
    report = build_markdown_report(traces)
    (artifacts_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    export_lightning_bundle(
        train_tasks=tasks,
        validation_tasks=validation_tasks or [],
        traces=traces,
        report_markdown=report,
        output_dir=artifacts_dir,
        settings=settings,
    )
    return traces, report


def run_dataset_offline_rollout(
    dataset_key: str,
    artifacts_dir: Path,
    *,
    subset: str | None = None,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    train_split: str | None = None,
    validation_split: str | None = None,
    settings: Settings | None = None,
) -> tuple[list[ExperimentTrace], str]:
    dataset_pair = load_train_validation_tasks(
        dataset_key,
        subset=subset,
        train_limit=train_limit,
        validation_limit=validation_limit,
        train_split=train_split,
        validation_split=validation_split,
    )
    return run_offline_rollout(
        dataset_pair.train,
        artifacts_dir,
        validation_tasks=dataset_pair.validation,
        settings=settings,
    )
