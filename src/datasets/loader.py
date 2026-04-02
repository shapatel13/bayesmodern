from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets.catalog import BenchmarkDatasetSpec, get_dataset_spec
from datasets.hf_loader import load_dataset_rows, load_local_rows
from priorix_tasks.common import BenchmarkTask
from utils.config import get_settings


@dataclass(frozen=True)
class BenchmarkDatasetPair:
    train: list[BenchmarkTask]
    validation: list[BenchmarkTask]
    spec: BenchmarkDatasetSpec


def _resolve_local_dataset_file(spec: BenchmarkDatasetSpec, split: str) -> Path | None:
    settings = get_settings()
    raw_path = getattr(settings, spec.local_path_setting, None) if spec.local_path_setting else None
    if not raw_path:
        return None

    candidate_path = Path(raw_path)
    if candidate_path.is_file():
        return candidate_path
    if not candidate_path.is_dir():
        raise FileNotFoundError(f"Configured path for `{spec.key}` does not exist: {candidate_path}")

    for filename in spec.local_split_filenames.get(split, ()):
        split_candidate = candidate_path / filename
        if split_candidate.exists():
            return split_candidate
    raise FileNotFoundError(
        f"Unable to resolve local dataset split `{split}` for `{spec.key}` in {candidate_path}. "
        f"Tried: {', '.join(spec.local_split_filenames.get(split, ())) or '<no filenames configured>'}"
    )


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
