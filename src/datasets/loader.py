from __future__ import annotations

from dataclasses import dataclass

from datasets.catalog import BenchmarkDatasetSpec, get_dataset_spec
from datasets.hf_loader import load_dataset_rows
from priorix_tasks.common import BenchmarkTask


@dataclass(frozen=True)
class BenchmarkDatasetPair:
    train: list[BenchmarkTask]
    validation: list[BenchmarkTask]
    spec: BenchmarkDatasetSpec


def load_benchmark_tasks(
    dataset_key: str,
    *,
    split: str | None = None,
    subset: str | None = None,
    limit: int | None = None,
) -> list[BenchmarkTask]:
    spec = get_dataset_spec(dataset_key)
    if spec.hf_dataset is None:
        raise ValueError(f"Dataset `{dataset_key}` requires local credentialed access and is not HF-loadable.")

    resolved_split = split or spec.default_eval_split
    resolved_subset = subset if subset is not None else spec.default_subset
    rows = load_dataset_rows(spec.hf_dataset, split=resolved_split, subset=resolved_subset, limit=limit)
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
