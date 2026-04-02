from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.lightning_adapter import detect_lightning_runtime
from agent.offline_rollout import run_offline_rollout
from core.policy import get_reasoning_policy, list_reasoning_policies
from datasets.catalog import BenchmarkDatasetSpec, get_dataset_spec
from datasets.curricula import LightningCurriculumSpec, get_lightning_curriculum
from datasets.loader import load_curriculum_train_validation_tasks, load_train_validation_tasks
from eval.benchmark_runner import summarize_benchmark
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


class PolicyCandidateOutcome(BaseModel):
    policy_version: str
    label: str
    experiment: ExperimentSummary
    comparison_to_baseline: ExperimentComparison | None = None


class PolicyOptimizationSummary(BaseModel):
    optimization_id: str
    created_at: str
    objective_kind: Literal["dataset", "curriculum"]
    objective_key: str
    artifact_dir: str
    baseline_policy_version: str
    candidate_policy_versions: list[str]
    selected_policy_version: str
    selected_experiment_id: str
    selection_reason: str
    lightning_runtime_mode: str
    outcomes: list[PolicyCandidateOutcome] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _default_candidate_policies(baseline_policy_version: str) -> list[str]:
    return [
        policy.version
        for policy in list_reasoning_policies()
        if policy.version != baseline_policy_version
    ]


def _policy_artifact_dir(root_dir: Path, policy_version: str) -> Path:
    slug = "".join(char if char.isalnum() else "_" for char in policy_version.lower()).strip("_")
    destination = root_dir / slug
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _build_dataset_summary(
    *,
    spec: BenchmarkDatasetSpec,
    dataset_key: str,
    tasks: list[BenchmarkTask],
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
        train_cases=len(tasks),
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


def _build_curriculum_summary(
    *,
    spec: LightningCurriculumSpec,
    tasks: list[BenchmarkTask],
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
        train_cases=len(tasks),
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


def _optimize_loaded_tasks(
    *,
    objective_kind: Literal["dataset", "curriculum"],
    objective_key: str,
    tasks: list[BenchmarkTask],
    validation_tasks: list[BenchmarkTask],
    artifacts_root: Path,
    settings: Settings,
    prompt_version: str,
    baseline_policy_version: str,
    candidate_policy_versions: list[str],
    component_datasets: list[str],
    summary_builder,
) -> PolicyOptimizationSummary:
    root_dir = create_experiment_dir(artifacts_root, f"{objective_key}_policy_search")
    outcomes: list[PolicyCandidateOutcome] = []
    notes = [
        "Policies were evaluated against the same offline task set for fair comparison.",
        "Microsoft Agent Lightning bundles were exported for each policy candidate.",
    ]

    policy_order = [baseline_policy_version, *candidate_policy_versions]
    for policy_version in policy_order:
        policy = get_reasoning_policy(policy_version)
        artifact_dir = _policy_artifact_dir(root_dir, policy_version)
        traces, _report = run_offline_rollout(
            tasks,
            artifact_dir,
            validation_tasks=validation_tasks,
            settings=settings,
            prompt_version=prompt_version,
            policy_version=policy_version,
            curriculum_key=objective_key if objective_kind == "curriculum" else None,
            component_datasets=component_datasets,
        )
        experiment = summary_builder(
            tasks=tasks,
            validation_tasks=validation_tasks,
            traces=traces,
            artifact_dir=artifact_dir,
            prompt_version=prompt_version,
            policy_version=policy_version,
            settings=settings,
        )
        write_experiment_summary(experiment, artifact_dir)
        outcomes.append(
            PolicyCandidateOutcome(
                policy_version=policy.version,
                label=policy.label,
                experiment=experiment,
            )
        )

    baseline_outcome = outcomes[0]
    best_outcome = baseline_outcome
    selection_reason = "Baseline policy retained because no candidate passed promotion gates."

    for index, outcome in enumerate(outcomes[1:], start=1):
        comparison = compare_experiment_summaries(baseline_outcome.experiment, outcome.experiment)
        outcomes[index] = outcome.model_copy(update={"comparison_to_baseline": comparison})
        if comparison.promotion_gate and comparison.promotion_gate.verdict == "promote":
            if outcome.experiment.benchmark_summary.mean_reward > best_outcome.experiment.benchmark_summary.mean_reward:
                best_outcome = outcomes[index]
                selection_reason = (
                    f"Promoted over baseline after improving mean reward to "
                    f"{outcome.experiment.benchmark_summary.mean_reward:.3f} without safety regressions."
                )

    runtime = detect_lightning_runtime(settings)
    if runtime.mode == "export_only":
        notes.append("Native Agent Lightning APO is not available on this host, so bundles are ready for Linux/WSL2 training.")
    else:
        notes.append("This host can continue from exported bundles into native Agent Lightning APO training.")

    optimization = PolicyOptimizationSummary(
        optimization_id=make_id("opt"),
        created_at=utc_now().isoformat(),
        objective_kind=objective_kind,
        objective_key=objective_key,
        artifact_dir=str(root_dir),
        baseline_policy_version=baseline_policy_version,
        candidate_policy_versions=candidate_policy_versions,
        selected_policy_version=best_outcome.policy_version,
        selected_experiment_id=best_outcome.experiment.experiment_id,
        selection_reason=selection_reason,
        lightning_runtime_mode=runtime.mode,
        outcomes=outcomes,
        notes=notes,
    )
    (root_dir / "policy_optimization_summary.json").write_text(
        dumps_pretty(optimization.model_dump()),
        encoding="utf-8",
    )
    return optimization


def optimize_dataset_policy(
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
    baseline_policy_version: str = "v1-deterministic",
    candidate_policy_versions: list[str] | None = None,
) -> PolicyOptimizationSummary:
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
    candidates = candidate_policy_versions or _default_candidate_policies(baseline_policy_version)
    return _optimize_loaded_tasks(
        objective_kind="dataset",
        objective_key=dataset_key,
        tasks=dataset_pair.train,
        validation_tasks=dataset_pair.validation,
        artifacts_root=artifacts_root,
        settings=settings,
        prompt_version=prompt_version,
        baseline_policy_version=baseline_policy_version,
        candidate_policy_versions=candidates,
        component_datasets=[dataset_key],
        summary_builder=lambda **kwargs: _build_dataset_summary(spec=spec, dataset_key=dataset_key, **kwargs),
    )


def optimize_curriculum_policy(
    curriculum_key: str,
    artifacts_root: Path,
    *,
    settings: Settings | None = None,
    prompt_version: str = "active",
    baseline_policy_version: str = "v1-deterministic",
    candidate_policy_versions: list[str] | None = None,
    train_cap_per_component: int | None = None,
    validation_cap_per_component: int | None = None,
) -> PolicyOptimizationSummary:
    settings = settings or get_settings()
    curriculum_pair = load_curriculum_train_validation_tasks(
        curriculum_key,
        train_cap_per_component=train_cap_per_component,
        validation_cap_per_component=validation_cap_per_component,
    )
    spec = get_lightning_curriculum(curriculum_key)
    candidates = candidate_policy_versions or _default_candidate_policies(baseline_policy_version)
    return _optimize_loaded_tasks(
        objective_kind="curriculum",
        objective_key=curriculum_key,
        tasks=curriculum_pair.train,
        validation_tasks=curriculum_pair.validation,
        artifacts_root=artifacts_root,
        settings=settings,
        prompt_version=prompt_version,
        baseline_policy_version=baseline_policy_version,
        candidate_policy_versions=candidates,
        component_datasets=[component.dataset_key for component in curriculum_pair.components],
        summary_builder=lambda **kwargs: _build_curriculum_summary(spec=spec, **kwargs),
    )
