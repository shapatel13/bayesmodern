from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from utils.bootstrap import ensure_src_path, prefer_local_package

ensure_src_path(Path(__file__).resolve().parents[1])
prefer_local_package("datasets", Path(__file__).resolve().parents[1] / "datasets")

from agent.lightning_adapter import detect_lightning_runtime
from agent.lightning_train import auto_improve_prompt, train_curriculum_prompt, train_dataset_prompt
from agent.prompt_registry import (
    get_active_prompt_record,
    get_prompt_registry_sync_target,
    list_prompt_records,
    promote_prompt_version,
)
from agent.offline_rollout import run_curriculum_offline_experiment, run_dataset_offline_experiment
from agent.policy_optimizer import optimize_curriculum_policy, optimize_dataset_policy
from core.policy import list_reasoning_policies
from datasets.catalog import list_dataset_specs
from datasets.curricula import get_lightning_curriculum, list_lightning_curricula
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
    subparsers.add_parser("list-curricula", help="List named multi-dataset Lightning feedback curricula.")
    subparsers.add_parser("list-policies", help="List available deterministic reasoning policies for offline optimization.")
    subparsers.add_parser("list-prompts", help="List tracked prompt templates and show the active prompt.")

    promote_prompt = subparsers.add_parser("promote-prompt", help="Promote a candidate prompt version to active.")
    promote_prompt.add_argument("prompt_version")

    rollout = subparsers.add_parser("run-dataset-rollout", help="Run an offline dataset rollout and export artifacts.")
    rollout.add_argument("dataset_key")
    rollout.add_argument("--subset", default=None)
    rollout.add_argument("--train-limit", type=int, default=8)
    rollout.add_argument("--validation-limit", type=int, default=4)
    rollout.add_argument("--train-split", default=None)
    rollout.add_argument("--validation-split", default=None)
    rollout.add_argument("--prompt-version", default="active")
    rollout.add_argument("--policy-version", default="v1-deterministic")
    rollout.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

    compare = subparsers.add_parser("compare-experiments", help="Compare two recorded experiments by ID.")
    compare.add_argument("baseline_experiment_id")
    compare.add_argument("candidate_experiment_id")
    compare.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

    preset = subparsers.add_parser("run-preset-rollout", help="Run an offline rollout using a named research preset.")
    preset.add_argument("preset_key")
    preset.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

    curriculum = subparsers.add_parser(
        "run-curriculum-rollout",
        help="Run an offline rollout using a named multi-dataset Lightning curriculum.",
    )
    curriculum.add_argument("curriculum_key")
    curriculum.add_argument("--prompt-version", default="active")
    curriculum.add_argument("--policy-version", default="v1-deterministic")
    curriculum.add_argument("--train-cap-per-component", type=int, default=None)
    curriculum.add_argument("--validation-cap-per-component", type=int, default=None)
    curriculum.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

    optimize_dataset = subparsers.add_parser(
        "optimize-dataset-policy",
        help="Evaluate baseline and candidate reasoning policies on one dataset and pick the safest improver.",
    )
    optimize_dataset.add_argument("dataset_key")
    optimize_dataset.add_argument("--subset", default=None)
    optimize_dataset.add_argument("--train-limit", type=int, default=8)
    optimize_dataset.add_argument("--validation-limit", type=int, default=4)
    optimize_dataset.add_argument("--train-split", default=None)
    optimize_dataset.add_argument("--validation-split", default=None)
    optimize_dataset.add_argument("--prompt-version", default="active")
    optimize_dataset.add_argument("--baseline-policy-version", default="v1-deterministic")
    optimize_dataset.add_argument("--candidate-policy-version", dest="candidate_policy_versions", action="append")
    optimize_dataset.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

    optimize_curriculum = subparsers.add_parser(
        "optimize-curriculum-policy",
        help="Evaluate baseline and candidate reasoning policies on a multi-dataset Lightning curriculum.",
    )
    optimize_curriculum.add_argument("curriculum_key")
    optimize_curriculum.add_argument("--prompt-version", default="active")
    optimize_curriculum.add_argument("--baseline-policy-version", default="v1-deterministic")
    optimize_curriculum.add_argument("--candidate-policy-version", dest="candidate_policy_versions", action="append")
    optimize_curriculum.add_argument("--train-cap-per-component", type=int, default=None)
    optimize_curriculum.add_argument("--validation-cap-per-component", type=int, default=None)
    optimize_curriculum.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

    train_dataset_prompt_parser = subparsers.add_parser(
        "train-dataset-prompt",
        help="Run Microsoft Agent Lightning prompt optimization on one dataset and safety-gate promotion on held-out validation tasks.",
    )
    train_dataset_prompt_parser.add_argument("dataset_key")
    train_dataset_prompt_parser.add_argument("--subset", default=None)
    train_dataset_prompt_parser.add_argument("--train-limit", type=int, default=8)
    train_dataset_prompt_parser.add_argument("--validation-limit", type=int, default=4)
    train_dataset_prompt_parser.add_argument("--train-split", default=None)
    train_dataset_prompt_parser.add_argument("--validation-split", default=None)
    train_dataset_prompt_parser.add_argument("--prompt-version", default="active")
    train_dataset_prompt_parser.add_argument("--policy-version", default="v1-deterministic")
    train_dataset_prompt_parser.add_argument("--n-runners", type=int, default=2)
    train_dataset_prompt_parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

    train_curriculum_prompt_parser = subparsers.add_parser(
        "train-curriculum-prompt",
        help="Run Microsoft Agent Lightning prompt optimization on a multi-dataset curriculum with held-out promotion gates.",
    )
    train_curriculum_prompt_parser.add_argument("curriculum_key")
    train_curriculum_prompt_parser.add_argument("--prompt-version", default="active")
    train_curriculum_prompt_parser.add_argument("--policy-version", default="v1-deterministic")
    train_curriculum_prompt_parser.add_argument("--train-cap-per-component", type=int, default=None)
    train_curriculum_prompt_parser.add_argument("--validation-cap-per-component", type=int, default=None)
    train_curriculum_prompt_parser.add_argument("--n-runners", type=int, default=2)
    train_curriculum_prompt_parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

    auto_improve = subparsers.add_parser(
        "auto-improve",
        help="Run the default continuous-improvement prompt loop over public QA plus optional reviewed cases.",
    )
    auto_improve.add_argument("--policy-version", default="v1-deterministic")
    auto_improve.add_argument("--train-cap-per-component", type=int, default=None)
    auto_improve.add_argument("--validation-cap-per-component", type=int, default=None)
    auto_improve.add_argument("--n-runners", type=int, default=2)
    auto_improve.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))

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
            "curriculum_count": len(list_lightning_curricula()),
            "curricula": [curriculum.key for curriculum in list_lightning_curricula()],
            "policy_count": len(list_reasoning_policies()),
            "policies": [policy.version for policy in list_reasoning_policies()],
            "active_prompt_version": get_active_prompt_record().version,
            "prompt_registry_sync_target": get_prompt_registry_sync_target(),
            "lightning_training_profile": settings.lightning_training_profile,
            "lightning_disable_agentops": settings.lightning_disable_agentops,
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

    if args.command == "list-curricula":
        payload = [asdict(curriculum) for curriculum in list_lightning_curricula()]
        print(dumps_pretty(payload))
        return 0

    if args.command == "list-policies":
        payload = [policy.model_dump() for policy in list_reasoning_policies()]
        print(dumps_pretty(payload))
        return 0

    if args.command == "list-prompts":
        payload = {
            "active_prompt": get_active_prompt_record().model_dump(),
            "prompts": [prompt.model_dump() for prompt in list_prompt_records()],
        }
        print(dumps_pretty(payload))
        return 0

    if args.command == "promote-prompt":
        registry = promote_prompt_version(args.prompt_version, notes=["Promoted manually from the experiment CLI."])
        print(dumps_pretty(registry.model_dump()))
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

    if args.command == "run-curriculum-rollout":
        curriculum = get_lightning_curriculum(args.curriculum_key)
        experiment, _traces, report = run_curriculum_offline_experiment(
            args.curriculum_key,
            artifacts_root,
            settings=settings,
            prompt_version=args.prompt_version,
            policy_version=args.policy_version,
            train_cap_per_component=args.train_cap_per_component,
            validation_cap_per_component=args.validation_cap_per_component,
        )
        payload = {
            "curriculum": asdict(curriculum),
            "experiment": experiment.model_dump(),
            "report": report,
        }
        print(dumps_pretty(payload))
        return 0

    if args.command == "optimize-dataset-policy":
        optimization = optimize_dataset_policy(
            args.dataset_key,
            artifacts_root,
            subset=args.subset,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            train_split=args.train_split,
            validation_split=args.validation_split,
            settings=settings,
            prompt_version=args.prompt_version,
            baseline_policy_version=args.baseline_policy_version,
            candidate_policy_versions=args.candidate_policy_versions,
        )
        print(dumps_pretty(optimization.model_dump()))
        return 0

    if args.command == "optimize-curriculum-policy":
        optimization = optimize_curriculum_policy(
            args.curriculum_key,
            artifacts_root,
            settings=settings,
            prompt_version=args.prompt_version,
            baseline_policy_version=args.baseline_policy_version,
            candidate_policy_versions=args.candidate_policy_versions,
            train_cap_per_component=args.train_cap_per_component,
            validation_cap_per_component=args.validation_cap_per_component,
        )
        print(dumps_pretty(optimization.model_dump()))
        return 0

    if args.command == "train-dataset-prompt":
        training = train_dataset_prompt(
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
            n_runners=args.n_runners,
        )
        print(dumps_pretty(training.model_dump()))
        return 0

    if args.command == "train-curriculum-prompt":
        training = train_curriculum_prompt(
            args.curriculum_key,
            artifacts_root,
            settings=settings,
            prompt_version=args.prompt_version,
            policy_version=args.policy_version,
            train_cap_per_component=args.train_cap_per_component,
            validation_cap_per_component=args.validation_cap_per_component,
            n_runners=args.n_runners,
        )
        print(dumps_pretty(training.model_dump()))
        return 0

    if args.command == "auto-improve":
        training = auto_improve_prompt(
            artifacts_root,
            settings=settings,
            policy_version=args.policy_version,
            train_cap_per_component=args.train_cap_per_component,
            validation_cap_per_component=args.validation_cap_per_component,
            n_runners=args.n_runners,
        )
        print(dumps_pretty(training.model_dump()))
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
