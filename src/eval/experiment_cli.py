from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from agent.lightning_adapter import detect_lightning_runtime
from agent.offline_rollout import run_dataset_offline_experiment
from datasets.catalog import list_dataset_specs
from eval.experiment_registry import compare_experiment_summaries, list_experiment_summaries
from eval.presets import get_research_preset, list_research_presets
from utils.config import get_settings
from utils.jsonx import dumps_pretty


DEFAULT_ARTIFACTS_ROOT = Path("artifacts/evals/experiments")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRIORI-X offline research experiment CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show dataset catalog size and Microsoft Agent Lightning runtime status.")
    subparsers.add_parser("list-experiments", help="List recorded offline experiments.")
    subparsers.add_parser("list-presets", help="List named specialty benchmark presets.")

    rollout = subparsers.add_parser("run-dataset-rollout", help="Run an offline dataset rollout and export artifacts.")
    rollout.add_argument("dataset_key")
    rollout.add_argument("--subset", default=None)
    rollout.add_argument("--train-limit", type=int, default=8)
    rollout.add_argument("--validation-limit", type=int, default=4)
    rollout.add_argument("--train-split", default=None)
    rollout.add_argument("--validation-split", default=None)
    rollout.add_argument("--prompt-version", default="v1-offline")
    rollout.add_argument("--policy-version", default="v1-deterministic")
    rollout.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

    compare = subparsers.add_parser("compare-experiments", help="Compare two recorded experiments by ID.")
    compare.add_argument("baseline_experiment_id")
    compare.add_argument("candidate_experiment_id")
    compare.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

    preset = subparsers.add_parser("run-preset-rollout", help="Run an offline rollout using a named research preset.")
    preset.add_argument("preset_key")
    preset.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

    return parser


def _resolve_experiment(experiment_id: str, artifacts_root: Path):
    for summary in list_experiment_summaries(artifacts_root):
        if summary.experiment_id == experiment_id:
            return summary
    raise KeyError(f"Experiment `{experiment_id}` was not found in {artifacts_root}.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = get_settings()

    if args.command == "status":
        payload = {
            "dataset_count": len(list_dataset_specs()),
            "datasets": [spec.key for spec in list_dataset_specs()],
            "preset_count": len(list_research_presets()),
            "presets": [preset.key for preset in list_research_presets()],
            "lightning_runtime": detect_lightning_runtime(settings).model_dump(),
            "artifacts_root": str(DEFAULT_ARTIFACTS_ROOT),
        }
        print(dumps_pretty(payload))
        return 0

    artifacts_root = Path(getattr(args, "artifacts_root", DEFAULT_ARTIFACTS_ROOT))

    if args.command == "list-experiments":
        payload = [summary.model_dump() for summary in list_experiment_summaries(artifacts_root)]
        print(dumps_pretty(payload))
        return 0

    if args.command == "list-presets":
        payload = [preset.__dict__ for preset in list_research_presets()]
        print(dumps_pretty(payload))
        return 0

    if args.command == "run-dataset-rollout":
        experiment, _traces, report = run_dataset_offline_experiment(
            args.dataset_key,
            artifacts_root,
            subset=args.subset,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            train_split=args.train_split,
            validation_split=args.validation_split,
            settings=settings,
            prompt_version=args.prompt_version,
            policy_version=args.policy_version,
        )
        payload = {
            "experiment": experiment.model_dump(),
            "report": report,
        }
        print(dumps_pretty(payload))
        return 0

    if args.command == "run-preset-rollout":
        preset = get_research_preset(args.preset_key)
        experiment, _traces, report = run_dataset_offline_experiment(
            preset.dataset_key,
            artifacts_root,
            subset=preset.subset,
            train_limit=preset.train_limit,
            validation_limit=preset.validation_limit,
            settings=settings,
            prompt_version=preset.prompt_version,
            policy_version=preset.policy_version,
        )
        payload = {
            "preset": preset.__dict__,
            "experiment": experiment.model_dump(),
            "report": report,
        }
        print(dumps_pretty(payload))
        return 0

    if args.command == "compare-experiments":
        baseline = _resolve_experiment(args.baseline_experiment_id, artifacts_root)
        candidate = _resolve_experiment(args.candidate_experiment_id, artifacts_root)
        print(dumps_pretty(compare_experiment_summaries(baseline, candidate).model_dump()))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
