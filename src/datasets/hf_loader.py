from __future__ import annotations

import importlib.machinery
import importlib.util
import site
from typing import Any


def _load_external_hf_datasets_module() -> Any:
    for path in site.getsitepackages():
        spec = importlib.machinery.PathFinder.find_spec("datasets", [path])
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError("The external `datasets` package is not installed. Install `.[datasets]` to enable HF loading.")


def load_dataset_rows(
    dataset_name: str,
    split: str,
    subset: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    datasets_module = _load_external_hf_datasets_module()
    dataset = datasets_module.load_dataset(dataset_name, subset, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]

