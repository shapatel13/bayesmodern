from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets.catalog import BenchmarkDatasetSpec, get_dataset_spec
from datasets.curricula import CurriculumComponent, LightningCurriculumSpec, get_lightning_curriculum
from datasets.hf_loader import load_dataset_rows, load_local_rows
from priorix_tasks.common import BenchmarkTask
from utils.config import get_settings


@dataclass(frozen=True)
class BenchmarkDatasetPair:
    train: list[BenchmarkTask]
    validation: list[BenchmarkTask]
    spec: BenchmarkDatasetSpec


@dataclass(frozen=True)
class CurriculumComponentLoad:
    dataset_key: str
    hf_dataset: str | None
    task_family: str
    reward_profile: str
    train_cases: int
    validation_cases: int
    requires_credentials: bool


@dataclass(frozen=True)
class BenchmarkCurriculumPair:
    train: list[BenchmarkTask]
    validation: list[BenchmarkTask]
    curriculum: LightningCurriculumSpec
    components: list[CurriculumComponentLoad]
    skipped_components: list[str]


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_local_dataset_file(spec: BenchmarkDatasetSpec, split: str) -> Path | None:
    settings = get_settings()
    raw_path = getattr(settings, spec.local_path_setting, None) if spec.local_path_setting else None

    candidate_paths: list[Path] = []
    if raw_path:
        candidate_paths.append(Path(raw_path))
    if spec.local_repo_relative_dir:
        candidate_paths.append(_REPO_ROOT / spec.local_repo_relative_dir)

    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            return candidate_path
        if not candidate_path.exists():
            continue
        if not candidate_path.is_dir():
            raise FileNotFoundError(f"Configured path for `{spec.key}` does not exist: {candidate_path}")
        for filename in spec.local_split_filenames.get(split, ()):
            split_candidate = candidate_path / filename
            if split_candidate.exists():
                return split_candidate

    if raw_path:
        raise FileNotFoundError(
            f"Unable to resolve local dataset split `{split}` for `{spec.key}`. "
            f"Tried configured path `{raw_path}` and filenames: "
            f"{', '.join(spec.local_split_filenames.get(split, ())) or '<no filenames configured>'}"
        )
    return None


def _load_rows_for_spec(
    spec: BenchmarkDatasetSpec,
    *,
    split: str,
    subset: str | None,
    limit: int | None,
) -> list[dict[str, object]]:
    local_path = _resolve_local_dataset_file(spec, split)
    if local_path is not None:
        return load_local_rows(local_path, limit=limit)

    if spec.hf_dataset is None:
        raise ValueError(
            f"Dataset `{spec.key}` requires local access. Configure {spec.local_path_env} with a local export path."
        )
    try:
        return load_dataset_rows(spec.hf_dataset, split=split, subset=subset, limit=limit)
    except Exception as exc:
        if spec.requires_credentials:
            raise RuntimeError(
                f"Dataset `{spec.key}` is credential-gated. Provide a local export via {spec.local_path_env} "
                f"or ensure Hugging Face access is configured. Original error: {exc}"
            ) from exc
        raise


def load_benchmark_tasks(
    dataset_key: str,
    *,
    split: str | None = None,
    subset: str | None = None,
    limit: int | None = None,
) -> list[BenchmarkTask]:
    spec = get_dataset_spec(dataset_key)
    resolved_split = split or spec.default_eval_split
    resolved_subset = subset if subset is not None else spec.default_subset
    rows = _load_rows_for_spec(spec, split=resolved_split, subset=resolved_subset, limit=limit)
    tasks = [spec.adapter(row, resolved_split) for row in rows]
    return [task for task in tasks if task.prompt.strip()]


def load_train_validation_tasks(
    dataset_key: str,
    *,
    subset: str | None = None,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    train_split: str | None = None,
    validation_split: str | None = None,
) -> BenchmarkDatasetPair:
    spec = get_dataset_spec(dataset_key)
    train_tasks = load_benchmark_tasks(
        dataset_key,
        split=train_split or spec.default_train_split,
        subset=subset,
        limit=train_limit,
    )
    validation_tasks = load_benchmark_tasks(
        dataset_key,
        split=validation_split or spec.default_val_split,
        subset=subset,
        limit=validation_limit,
    )
    return BenchmarkDatasetPair(train=train_tasks, validation=validation_tasks, spec=spec)


def _reward_profile_for_task_type(task_type: str) -> str:
    if task_type == "triage":
        return "triage"
    if task_type == "medication_safety":
        return "medication_safety"
    if task_type == "evidence_verification":
        return "evidence_verification"
    return "diagnostic"


def _annotate_curriculum_tasks(
    tasks: list[BenchmarkTask],
    *,
    curriculum_key: str,
    component: CurriculumComponent,
    spec: BenchmarkDatasetSpec,
) -> list[BenchmarkTask]:
    annotated: list[BenchmarkTask] = []
    for index, task in enumerate(tasks):
        metadata = dict(task.metadata)
        metadata.update(
            {
                "curriculum_key": curriculum_key,
                "curriculum_component": component.dataset_key,
                "curriculum_component_index": index,
                "curriculum_weight": component.weight,
                "reward_profile_hint": _reward_profile_for_task_type(task.task_type),
                "source_hf_dataset": spec.hf_dataset,
                "source_task_family": spec.task_family,
            }
        )
        annotated.append(task.model_copy(update={"metadata": metadata}))
    return annotated


def _weighted_interleave(task_groups: list[list[BenchmarkTask]], weights: list[int]) -> list[BenchmarkTask]:
    queues = [list(group) for group in task_groups]
    schedule: list[int] = []
    for index, weight in enumerate(weights):
        schedule.extend([index] * max(1, weight))

    ordered: list[BenchmarkTask] = []
    while any(queues):
        progressed = False
        for index in schedule:
            if queues[index]:
                ordered.append(queues[index].pop(0))
                progressed = True
        if not progressed:
            break
    return ordered


def load_curriculum_train_validation_tasks(
    curriculum_key: str,
    *,
    train_cap_per_component: int | None = None,
    validation_cap_per_component: int | None = None,
) -> BenchmarkCurriculumPair:
    curriculum = get_lightning_curriculum(curriculum_key)
    train_groups: list[list[BenchmarkTask]] = []
    validation_groups: list[list[BenchmarkTask]] = []
    weights: list[int] = []
    components: list[CurriculumComponentLoad] = []
    skipped_components: list[str] = []

    for component in curriculum.components:
        spec = get_dataset_spec(component.dataset_key)
        try:
            dataset_pair = load_train_validation_tasks(
                component.dataset_key,
                subset=component.subset,
                train_limit=(
                    min(component.train_limit, train_cap_per_component)
                    if train_cap_per_component is not None
                    else component.train_limit
                ),
                validation_limit=(
                    min(component.validation_limit, validation_cap_per_component)
                    if validation_cap_per_component is not None
                    else component.validation_limit
                ),
                train_split=component.train_split,
                validation_split=component.validation_split,
            )
        except Exception as exc:
            if component.optional:
                skipped_components.append(f"{component.dataset_key}: {exc}")
                continue
            raise

        train_tasks = _annotate_curriculum_tasks(
            dataset_pair.train,
            curriculum_key=curriculum.key,
            component=component,
            spec=spec,
        )
        validation_tasks = _annotate_curriculum_tasks(
            dataset_pair.validation,
            curriculum_key=curriculum.key,
            component=component,
            spec=spec,
        )
        train_groups.append(train_tasks)
        validation_groups.append(validation_tasks)
        weights.append(component.weight)
        components.append(
            CurriculumComponentLoad(
                dataset_key=component.dataset_key,
                hf_dataset=spec.hf_dataset,
                task_family=spec.task_family,
                reward_profile=_reward_profile_for_task_type(spec.task_family),
                train_cases=len(train_tasks),
                validation_cases=len(validation_tasks),
                requires_credentials=spec.requires_credentials,
            )
        )

    if not components:
        raise RuntimeError(
            f"Lightning curriculum `{curriculum_key}` could not load any datasets. "
            f"Skipped components: {', '.join(skipped_components) or 'none'}"
        )

    return BenchmarkCurriculumPair(
        train=_weighted_interleave(train_groups, weights),
        validation=_weighted_interleave(validation_groups, weights),
        curriculum=curriculum,
        components=components,
        skipped_components=skipped_components,
    )
