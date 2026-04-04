from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.lightning_adapter import build_native_lightning_recipe, detect_lightning_runtime
from agent.prompt_registry import (
    get_active_prompt_record,
    promote_prompt_version,
    register_prompt_candidate,
    reject_prompt_version,
    resolve_prompt_record,
)
from datasets.catalog import BenchmarkDatasetSpec, get_dataset_spec
from datasets.curricula import LightningCurriculumSpec, get_lightning_curriculum
from datasets.loader import load_curriculum_train_validation_tasks, load_train_validation_tasks
from eval.benchmark_runner import build_markdown_report, run_benchmark, summarize_benchmark
from eval.experiment_registry import (
    ExperimentComparison,
    ExperimentSummary,
    compare_experiment_summaries,
    create_experiment_dir,
    write_experiment_summary,
)
from priorix_tasks.common import BenchmarkTask
from utils.config import Settings, get_settings
from utils.dates import utc_now
from utils.ids import make_id
from utils.jsonx import dumps_pretty


class PromptTrainingSummary(BaseModel):
    training_id: str
    created_at: str
    objective_kind: Literal["dataset", "curriculum"]
    objective_key: str
    artifact_dir: str
    status: Literal["blocked", "unchanged", "trained_and_held", "trained_and_promoted"]
    baseline_prompt_version: str
    selected_prompt_version: str
    candidate_prompt_version: str | None = None
    policy_version: str
    baseline_experiment_id: str
    candidate_experiment_id: str | None = None
    promoted: bool = False
    comparison_to_baseline: ExperimentComparison | None = None
    lightning_runtime_mode: str
    notes: list[str] = Field(default_factory=list)


def _get_latest_prompt_template(trainer) -> tuple[str | None, str]:
    latest_update = asyncio.run(trainer.store.get_latest_resources())
    if latest_update is None:
        raise RuntimeError("Microsoft Agent Lightning training completed without publishing updated resources.")
    prompt_resource = latest_update.resources.get("prompt_template")
    if prompt_resource is None or not hasattr(prompt_resource, "template"):
        raise RuntimeError("No prompt_template resource was returned by Microsoft Agent Lightning.")
    return latest_update.resources_id, str(prompt_resource.template)


def _write_training_summary(summary: PromptTrainingSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_training_summary.json").write_text(dumps_pretty(summary.model_dump()), encoding="utf-8")


def _evaluation_summary_for_dataset(
    *,
    spec: BenchmarkDatasetSpec,
    dataset_key: str,
    train_tasks: list[BenchmarkTask],
    validation_tasks: list[BenchmarkTask],
    traces,
    artifact_dir: Path,
    prompt_version: str,
    policy_version: str,
    settings: Settings,
) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=make_id("exp"),
        created_at=utc_now().isoformat(),
        dataset_key=dataset_key,
        dataset_hf_id=spec.hf_dataset,
        source_kind="dataset",
        task_family=spec.task_family,
        subset=spec.default_subset,
        train_split=spec.default_train_split,
        validation_split=spec.default_val_split,
        train_cases=len(train_tasks),
        validation_cases=len(validation_tasks),
        component_datasets=[dataset_key],
        component_hf_ids=[spec.hf_dataset] if spec.hf_dataset else [],
        reward_profiles=sorted({trace.reward.reward_profile for trace in traces if trace.reward}),
        prompt_version=prompt_version,
        policy_version=policy_version,
        artifact_dir=str(artifact_dir),
        benchmark_summary=summarize_benchmark(traces),
        lightning_runtime=detect_lightning_runtime(settings),
    )


def _evaluation_summary_for_curriculum(
    *,
    spec: LightningCurriculumSpec,
    train_tasks: list[BenchmarkTask],
    validation_tasks: list[BenchmarkTask],
    traces,
    artifact_dir: Path,
    prompt_version: str,
    policy_version: str,
    settings: Settings,
) -> ExperimentSummary:
    component_dataset_keys = [component.dataset_key for component in spec.components]
    component_specs = [get_dataset_spec(component.dataset_key) for component in spec.components]
    return ExperimentSummary(
        experiment_id=make_id("exp"),
        created_at=utc_now().isoformat(),
        dataset_key=spec.key,
        dataset_hf_id=None,
        source_kind="curriculum",
        task_family="mixed_feedback",
        subset=None,
        train_split="mixed",
        validation_split="mixed",
        train_cases=len(train_tasks),
        validation_cases=len(validation_tasks),
        component_datasets=component_dataset_keys,
        component_hf_ids=[item.hf_dataset for item in component_specs if item.hf_dataset],
        reward_profiles=sorted({trace.reward.reward_profile for trace in traces if trace.reward}),
        prompt_version=prompt_version,
        policy_version=policy_version,
        artifact_dir=str(artifact_dir),
        benchmark_summary=summarize_benchmark(traces),
        lightning_runtime=detect_lightning_runtime(settings),
    )


def _evaluate_prompt_version(
    *,
    validation_tasks: list[BenchmarkTask],
    artifact_dir: Path,
    prompt_version: str,
    policy_version: str,
    summary_builder,
) -> ExperimentSummary:
    traces = run_benchmark(validation_tasks, output_dir=artifact_dir, prompt_version=prompt_version, policy_version=policy_version)
    report = build_markdown_report(traces)
    (artifact_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    summary = summary_builder(traces=traces, artifact_dir=artifact_dir, prompt_version=prompt_version, policy_version=policy_version)
    write_experiment_summary(summary, artifact_dir)
    return summary


def _blocked_training_summary(
    *,
    objective_kind: Literal["dataset", "curriculum"],
    objective_key: str,
    artifact_dir: Path,
    baseline_prompt_version: str,
    baseline_experiment_id: str,
    policy_version: str,
    runtime_mode: str,
    notes: list[str],
) -> PromptTrainingSummary:
    summary = PromptTrainingSummary(
        training_id=make_id("train"),
        created_at=utc_now().isoformat(),
        objective_kind=objective_kind,
        objective_key=objective_key,
        artifact_dir=str(artifact_dir),
        status="blocked",
        baseline_prompt_version=baseline_prompt_version,
        selected_prompt_version=baseline_prompt_version,
        policy_version=policy_version,
        baseline_experiment_id=baseline_experiment_id,
        lightning_runtime_mode=runtime_mode,
        notes=notes,
    )
    _write_training_summary(summary, artifact_dir)
    return summary


def _train_loaded_tasks(
    *,
    objective_kind: Literal["dataset", "curriculum"],
    objective_key: str,
    train_tasks: list[BenchmarkTask],
    validation_tasks: list[BenchmarkTask],
    artifacts_root: Path,
    settings: Settings,
    prompt_version: str,
    policy_version: str,
    summary_builder,
    n_runners: int,
) -> PromptTrainingSummary:
    baseline_prompt = resolve_prompt_record(prompt_version)
    artifact_dir = create_experiment_dir(artifacts_root, f"{objective_key}_prompt_training")
    runtime = detect_lightning_runtime(settings)
    notes = [
        "Prompt optimization is always offline and benchmark-driven.",
        "Raw app traffic is never used directly for training.",
        "Only held-out validation tasks are used for prompt promotion decisions.",
        f"Lightning training profile: {settings.lightning_training_profile}.",
        (
            "AgentOps tracer disabled for speed and cleaner logs."
            if settings.lightning_disable_agentops
            else "AgentOps tracer left enabled."
        ),
    ]

    baseline_dir = artifact_dir / "baseline"
    baseline_summary = _evaluate_prompt_version(
        validation_tasks=validation_tasks,
        artifact_dir=baseline_dir,
        prompt_version=baseline_prompt.version,
        policy_version=policy_version,
        summary_builder=summary_builder,
    )

    if not runtime.native_training_ready:
        notes.append(runtime.reason)
        if runtime.wsl_distribution_installed is False:
            notes.append("Install a Linux distro first: wsl --install Ubuntu")
        return _blocked_training_summary(
            objective_kind=objective_kind,
            objective_key=objective_key,
            artifact_dir=artifact_dir,
            baseline_prompt_version=baseline_prompt.version,
            baseline_experiment_id=baseline_summary.experiment_id,
            policy_version=policy_version,
            runtime_mode=runtime.mode,
            notes=notes,
        )

    recipe = build_native_lightning_recipe(
        settings=settings,
        prompt_version=baseline_prompt.version,
        policy_version=policy_version,
        train_task_count=len(train_tasks),
        validation_task_count=len(validation_tasks),
        n_runners=n_runners,
    )
    recipe.trainer.fit(recipe.agent, train_tasks, val_dataset=validation_tasks)
    resources_id, candidate_template = _get_latest_prompt_template(recipe.trainer)

    if candidate_template.strip() == baseline_prompt.template.strip():
        summary = PromptTrainingSummary(
            training_id=make_id("train"),
            created_at=utc_now().isoformat(),
            objective_kind=objective_kind,
            objective_key=objective_key,
            artifact_dir=str(artifact_dir),
            status="unchanged",
            baseline_prompt_version=baseline_prompt.version,
            selected_prompt_version=baseline_prompt.version,
            policy_version=policy_version,
            baseline_experiment_id=baseline_summary.experiment_id,
            lightning_runtime_mode=runtime.mode,
            notes=notes + ["Agent Lightning returned the same prompt template as the baseline."],
        )
        _write_training_summary(summary, artifact_dir)
        return summary

    candidate = register_prompt_candidate(
        template=candidate_template,
        label=f"APO candidate for {objective_key}",
        description=f"Generated by Microsoft Agent Lightning APO on {objective_kind} `{objective_key}`.",
        based_on_version=baseline_prompt.version,
        resources_id=resources_id,
        experiment_id=baseline_summary.experiment_id,
        notes=[
            f"Generated at {utc_now().isoformat()}",
            f"Policy version during training: {policy_version}",
        ],
    )

    candidate_dir = artifact_dir / "candidate"
    candidate_summary = _evaluate_prompt_version(
        validation_tasks=validation_tasks,
        artifact_dir=candidate_dir,
        prompt_version=candidate.version,
        policy_version=policy_version,
        summary_builder=summary_builder,
    )
    comparison = compare_experiment_summaries(baseline_summary, candidate_summary)
    promoted = bool(comparison.promotion_gate and comparison.promotion_gate.verdict == "promote")

    if promoted:
        promote_prompt_version(
            candidate.version,
            notes=[
                f"Promoted by training run {artifact_dir.name}.",
                f"Baseline prompt was {baseline_prompt.version}.",
            ],
        )
        status: Literal["trained_and_promoted", "trained_and_held"] = "trained_and_promoted"
        selected_prompt_version = candidate.version
        notes.append("Candidate prompt passed the held-out promotion gate and is now active.")
    else:
        if comparison.promotion_gate and comparison.promotion_gate.verdict == "reject":
            reject_prompt_version(
                candidate.version,
                notes=["Rejected automatically because the held-out safety gate failed."],
            )
        status = "trained_and_held"
        selected_prompt_version = get_active_prompt_record().version
        notes.append("Candidate prompt did not clear the held-out promotion gate and was not activated.")

    summary = PromptTrainingSummary(
        training_id=make_id("train"),
        created_at=utc_now().isoformat(),
        objective_kind=objective_kind,
        objective_key=objective_key,
        artifact_dir=str(artifact_dir),
        status=status,
        baseline_prompt_version=baseline_prompt.version,
        selected_prompt_version=selected_prompt_version,
        candidate_prompt_version=candidate.version,
        policy_version=policy_version,
        baseline_experiment_id=baseline_summary.experiment_id,
        candidate_experiment_id=candidate_summary.experiment_id,
        promoted=promoted,
        comparison_to_baseline=comparison,
        lightning_runtime_mode=runtime.mode,
        notes=notes,
    )
    _write_training_summary(summary, artifact_dir)
    return summary


def train_dataset_prompt(
    dataset_key: str,
    artifacts_root: Path,
    *,
    subset: str | None = None,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    train_split: str | None = None,
    validation_split: str | None = None,
    settings: Settings | None = None,
    prompt_version: str = "active",
    policy_version: str = "v1-deterministic",
    n_runners: int = 1,
) -> PromptTrainingSummary:
    settings = settings or get_settings()
    dataset_pair = load_train_validation_tasks(
        dataset_key,
        subset=subset,
        train_limit=train_limit,
        validation_limit=validation_limit,
        train_split=train_split,
        validation_split=validation_split,
    )
    spec = get_dataset_spec(dataset_key)
    return _train_loaded_tasks(
        objective_kind="dataset",
        objective_key=dataset_key,
        train_tasks=dataset_pair.train,
        validation_tasks=dataset_pair.validation,
        artifacts_root=artifacts_root,
        settings=settings,
        prompt_version=prompt_version,
        policy_version=policy_version,
        n_runners=n_runners,
        summary_builder=lambda **kwargs: _evaluation_summary_for_dataset(
            spec=spec,
            dataset_key=dataset_key,
            train_tasks=dataset_pair.train,
            validation_tasks=dataset_pair.validation,
            settings=settings,
            **kwargs,
        ),
    )


def train_curriculum_prompt(
    curriculum_key: str,
    artifacts_root: Path,
    *,
    settings: Settings | None = None,
    prompt_version: str = "active",
    policy_version: str = "v1-deterministic",
    train_cap_per_component: int | None = None,
    validation_cap_per_component: int | None = None,
    n_runners: int = 1,
) -> PromptTrainingSummary:
    settings = settings or get_settings()
    curriculum_pair = load_curriculum_train_validation_tasks(
        curriculum_key,
        train_cap_per_component=train_cap_per_component,
        validation_cap_per_component=validation_cap_per_component,
    )
    curriculum = get_lightning_curriculum(curriculum_key)
    return _train_loaded_tasks(
        objective_kind="curriculum",
        objective_key=curriculum_key,
        train_tasks=curriculum_pair.train,
        validation_tasks=curriculum_pair.validation,
        artifacts_root=artifacts_root,
        settings=settings,
        prompt_version=prompt_version,
        policy_version=policy_version,
        n_runners=n_runners,
        summary_builder=lambda **kwargs: _evaluation_summary_for_curriculum(
            spec=curriculum,
            train_tasks=curriculum_pair.train,
            validation_tasks=curriculum_pair.validation,
            settings=settings,
            **kwargs,
        ),
    )


def auto_improve_prompt(
    artifacts_root: Path,
    *,
    settings: Settings | None = None,
    policy_version: str = "v1-deterministic",
    train_cap_per_component: int | None = None,
    validation_cap_per_component: int | None = None,
    n_runners: int = 1,
) -> PromptTrainingSummary:
    return train_curriculum_prompt(
        "continuous_improvement_feedback_lab",
        artifacts_root,
        settings=settings,
        prompt_version="active",
        policy_version=policy_version,
        train_cap_per_component=train_cap_per_component,
        validation_cap_per_component=validation_cap_per_component,
        n_runners=n_runners,
    )
