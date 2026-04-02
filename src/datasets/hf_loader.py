from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


_SRC_ROOT = Path(__file__).resolve().parents[1]


def _sanitized_pythonpath() -> str | None:
    raw_pythonpath = os.environ.get("PYTHONPATH", "")
    if not raw_pythonpath:
        return None
    filtered_parts: list[str] = []
    for part in raw_pythonpath.split(os.pathsep):
        if not part:
            continue
        try:
            if Path(part).resolve() == _SRC_ROOT.resolve():
                continue
        except OSError:
            pass
        filtered_parts.append(part)
    return os.pathsep.join(filtered_parts) if filtered_parts else None


def load_dataset_rows(
    dataset_name: str,
    split: str,
    subset: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    request_payload = {
        "dataset_name": dataset_name,
        "split": split,
        "subset": subset,
        "limit": limit,
    }
    env = os.environ.copy()
    sanitized_pythonpath = _sanitized_pythonpath()
    if sanitized_pythonpath is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = sanitized_pythonpath

    command = (
        "import json, os\n"
        "payload = json.loads(os.environ['PRIORI_HF_DATASET_REQUEST'])\n"
        "from datasets import load_dataset\n"
        "dataset = load_dataset(payload['dataset_name'], payload['subset'], split=payload['split'])\n"
        "limit = payload.get('limit')\n"
        "if limit is not None:\n"
        "    dataset = dataset.select(range(min(limit, len(dataset))))\n"
        "print(json.dumps([dict(row) for row in dataset], ensure_ascii=True))\n"
    )
    env["PRIORI_HF_DATASET_REQUEST"] = json.dumps(request_payload)
    result = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"Hugging Face dataset load failed for `{dataset_name}`: {stderr}")
    return json.loads(result.stdout)


def load_local_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = [dict(row) for row in payload]
        elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            rows = [dict(row) for row in payload["rows"]]
        else:
            raise ValueError(f"Unsupported JSON dataset payload in {path}. Expected a list or object with `rows`.")
    else:
        raise ValueError(f"Unsupported local dataset file type: {path.suffix}")

    if limit is not None:
        rows = rows[:limit]
    return rows
