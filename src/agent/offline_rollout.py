from __future__ import annotations

from pathlib import Path

from agent.lightning_adapter import detect_lightning_runtime, export_lightning_bundle
from agent.trace_schema import ExperimentTrace
from datasets.catalog import get_dataset_spec
from datasets.loader import load_curriculum_train_validation_tasks, load_train_validation_tasks
from eval.benchmark_runner import build_markdown_report, run_benchmark, summarize_benchmark
from eval.experiment_registry import ExperimentSummary, create_experiment_dir, write_experiment_summary
from priorix_tasks.common import BenchmarkTask
from utils.config import Settings, get_settings
from utils.dates import utc_now
from utils.ids import make_id
from utils.jsonx import dumps_pretty


def _reward_profiles_for_tasks(tasks: list[BenchmarkTask], validation_tasks: list[BenchmarkTask] | None = None) -> list[str]:
    reward_profiles = {
        str(task.metadata.get("reward_profile_hint", "diagnostic"))
        for task in tasks + (validation_tasks or [])
    }
    return sorted(profile for profile in reward_profiles if profile)


def run_offline_rollout(
    tasks: list[BenchmarkTask],
    artifacts_dir: Path,
    *,
    validation_tasks: list[BenchmarkTask] | None = None,
    settings: Settings | None = None,
    prompt_version: str = "v1-offline",
    policy_version: str = "v1-deterministic",
    curriculum_key: str | None = None,
    component_datasets: list[str] | None = None,
) -> tuple[list[ExperimentTrace], str]:
    settings = settings or get_settings()
    traces = run_benchmark(
        tasks,
        output_dir=artifacts_dir,
        prompt_version=prompt_version,
        policy_version=policy_version,
    )
    report = build_markdown_report(traces)
    summary = summarize_benchmark(traces)
    (artifacts_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    (artifacts_dir / "benchmark_summary.json").write_text(dumps_pretty(summary.model_dump()), encoding="utf-8")
    export_lightning_bundle(
        train_tasks=tasks,
        validation_tasks=validation_tasks or [],
        traces=traces,
        report_markdown=report,
        output_dir=artifacts_dir,
        settings=settings,
        curriculum_key=curriculum_key,
        component_datasets=component_datasets,
        reward_profiles=_reward_profiles_for_tasks(tasks, validation_tasks),
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
    prompt_version: str = "v1-offline",
    policy_version: str = "v1-deterministic",
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
        prompt_version=prompt_version,
        policy_version=policy_version,
    )


def run_dataset_offline_experiment(
    dataset_key: str,
    artifacts_root: Path,
    *,
    subset: str | None = None,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    train_split: str | None = None,
    validation_split: str | None = None,
    settings: Settings | None = None,
    prompt_version: str = "v1-offline",
    policy_version: str = "v1-deterministic",
) -> tuple[ExperimentSummary, list[ExperimentTrace], str]:
    settings = settings or get_settings()
    dataset_pair = load_train_validation_tasks(
        dataset_key,
        subset=subset,
        train_limit=train_limit,
        validation_limit=validation_limit,
        train_split=train_split,
        validation_split=validation_split,
    )
    artifact_dir = create_experiment_dir(artifacts_root, dataset_key)
    traces, report = run_offline_rollout(
        dataset_pair.train,
        artifact_dir,
        validation_tasks=dataset_pair.validation,
        settings=settings,
        prompt_version=prompt_version,
        policy_version=policy_version,
        component_datasets=[dataset_key],
    )
    spec = get_dataset_spec(dataset_key)
    runtime = detect_lightning_runtime(settings)
    experiment_summary = ExperimentSummary(
        experiment_id=make_id("exp"),
        created_at=utc_now().isoformat(),
        dataset_key=dataset_key,
        dataset_hf_id=spec.hf_dataset,
        source_kind="dataset",
        task_family=spec.task_family,
        subset=subset if subset is not None else spec.default_subset,
        train_split=train_split or spec.default_train_split,
        validation_split=validation_split or spec.default_val_split,
        train_cases=len(dataset_pair.train),
        validation_cases=len(dataset_pair.validation),
        component_datasets=[dataset_key],
        component_hf_ids=[spec.hf_dataset] if spec.hf_dataset else [],
        reward_profiles=_reward_profiles_for_tasks(dataset_pair.train, dataset_pair.validation),
        prompt_version=prompt_version,
        policy_version=policy_version,
        artifact_dir=str(artifact_dir),
        benchmark_summary=summarize_benchmark(traces),
        lightning_runtime=runtime,
    )
    write_experiment_summary(experiment_summary, artifact_dir)
    return experiment_summary, traces, report


def run_curriculum_offline_experiment(
    curriculum_key: str,
    artifacts_root: Path,
    *,
    settings: Settings | None = None,
    prompt_version: str = "v1-offline",
    policy_version: str = "v1-deterministic",
    train_cap_per_component: int | None = None,
    validation_cap_per_component: int | None = None,
) -> tuple[ExperimentSummary, list[ExperimentTrace], str]:
    settings = settings or get_settings()
    curriculum_pair = load_curriculum_train_validation_tasks(
        curriculum_key,
        train_cap_per_component=train_cap_per_component,
        validation_cap_per_component=validation_cap_per_component,
    )
    artifact_dir = create_experiment_dir(artifacts_root, curriculum_key)
    traces, report = run_offline_rollout(
        curriculum_pair.train,
        artifact_dir,
        validation_tasks=curriculum_pair.validation,
        settings=settings,
        prompt_version=prompt_version,
        policy_version=policy_version,
        curriculum_key=curriculum_key,
        component_datasets=[component.dataset_key for component in curriculum_pair.components],
    )
    runtime = detect_lightning_runtime(settings)
    experiment_summary = ExperimentSummary(
        experiment_id=make_id("exp"),
        created_at=utc_now().isoformat(),
        dataset_key=curriculum_key,
        dataset_hf_id=None,
        source_kind="curriculum",
        task_family="mixed_feedback",
        subset=None,
        train_split="mixed",
        validation_split="mixed",
        train_cases=len(curriculum_pair.train),
        validation_cases=len(curriculum_pair.validation),
        component_datasets=[component.dataset_key for component in curriculum_pair.components],
        component_hf_ids=[component.hf_dataset for component in curriculum_pair.components if component.hf_dataset],
        reward_profiles=sorted({component.reward_profile for component in curriculum_pair.components}),
        prompt_version=prompt_version,
        policy_version=policy_version,
        artifact_dir=str(artifact_dir),
        benchmark_summary=summarize_benchmark(traces),
        lightning_runtime=runtime,
    )
    write_experiment_summary(experiment_summary, artifact_dir)
    return experiment_summary, traces, report
