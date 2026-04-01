from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.orchestrator import PRIORIXOrchestrator
from agent.reward_model import CompositeRewardModel
from agent.trace_schema import ExperimentTrace, LightningTransition
from priorix_tasks.common import BenchmarkTask
from utils.config import Settings, get_settings
from utils.jsonx import dumps_pretty


LightningMode = Literal["native_ready", "export_only"]


class LightningRuntimeStatus(BaseModel):
    mode: LightningMode
    package_available: bool
    package_version: str | None = None
    platform_supported: bool
    native_training_ready: bool
    reason: str


class LightningBundleManifest(BaseModel):
    runtime: LightningRuntimeStatus
    algorithm: str = "apo"
    adapter: str = "TraceToMessages"
    train_tasks_path: str
    validation_tasks_path: str
    transitions_path: str
    traces_path: str
    report_path: str
    prompt_template_baseline: str
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class NativeLightningRecipe:
    agent: Any
    trainer: Any
    runtime: LightningRuntimeStatus


def _running_in_wsl() -> bool:
    return bool(os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"))


def _agentlightning_version() -> str | None:
    try:
        return version("agentlightning")
    except PackageNotFoundError:
        return None


def detect_lightning_runtime(settings: Settings | None = None) -> LightningRuntimeStatus:
    settings = settings or get_settings()
    package_version = _agentlightning_version()
    package_available = package_version is not None
    platform_supported = sys.platform != "win32" or _running_in_wsl()
    llm_ready = settings.allow_live_llm and bool(settings.openai_api_key)
    native_training_ready = package_available and platform_supported and llm_ready

    if not package_available:
        reason = "Microsoft Agent Lightning package not installed; exporting sandbox bundle only."
    elif not platform_supported:
        reason = "Native Microsoft Agent Lightning training should run on Linux or WSL2; exporting sandbox bundle only."
    elif not llm_ready:
        reason = "Native APO training requires offline-eval LLM credentials and PRIORI_ALLOW_LIVE_LLM=true; exporting sandbox bundle only."
    else:
        reason = "Native Microsoft Agent Lightning prompt optimization is ready."

    return LightningRuntimeStatus(
        mode="native_ready" if native_training_ready else "export_only",
        package_available=package_available,
        package_version=package_version,
        platform_supported=platform_supported,
        native_training_ready=native_training_ready,
        reason=reason,
    )


def traces_to_lightning_transitions(traces: list[ExperimentTrace]) -> list[LightningTransition]:
    transitions: list[LightningTransition] = []
    for trace in traces:
        top = trace.report.differential.ranked[0] if trace.report.differential.ranked else None
        recommendation = trace.report.next_best_tests[0] if trace.report.next_best_tests else None
        transitions.append(
            LightningTransition(
                task_id=trace.task_id,
                state={
                    "prompt_version": trace.prompt_version,
                    "policy_version": trace.policy_version,
                    "triage": trace.report.triage.urgency,
                    "task_type": trace.task_type,
                    "source_dataset": trace.source_dataset,
                },
                action={
                    "top_diagnosis": top.slug if top else None,
                    "recommended_test": recommendation.slug if recommendation else None,
                },
                reward=trace.reward.total_reward if trace.reward else 0.0,
                done=True,
                info={
                    "failure_categories": trace.reward.failure_categories if trace.reward else [],
                    "hard_veto": trace.reward.hard_veto if trace.reward else False,
                    "contradictions": trace.report.contradictions,
                    "provenance_warnings": trace.report.provenance_warnings,
                },
            )
        )
    return transitions


def export_lightning_jsonl(traces: list[ExperimentTrace], output_path: Path) -> None:
    transitions = traces_to_lightning_transitions(traces)
    payload = "\n".join(json.dumps(transition.model_dump(), sort_keys=True, ensure_ascii=True) for transition in transitions)
    output_path.write_text(payload, encoding="utf-8")


def export_benchmark_tasks_jsonl(tasks: list[BenchmarkTask], output_path: Path) -> None:
    payload = "\n".join(json.dumps(task.model_dump(), ensure_ascii=True) for task in tasks)
    output_path.write_text(payload, encoding="utf-8")


def baseline_prompt_template() -> str:
    return (
        "You are PRIORI-X, a clinician-facing research workbench operating strictly in offline evaluation mode.\n"
        "Analyze the following benchmark vignette conservatively, preserve uncertainty, and avoid treatment autopilot.\n\n"
        "Case:\n{task}"
    )


def render_prompt_template(prompt_template: Any, task_prompt: str) -> str:
    if hasattr(prompt_template, "format"):
        try:
            return prompt_template.format(task=task_prompt)
        except TypeError:
            pass
    template_text = str(prompt_template)
    if "{task}" in template_text:
        return template_text.format(task=task_prompt)
    return f"{template_text}\n\nCase:\n{task_prompt}".strip()


def export_lightning_bundle(
    *,
    train_tasks: list[BenchmarkTask],
    validation_tasks: list[BenchmarkTask],
    traces: list[ExperimentTrace],
    report_markdown: str,
    output_dir: Path,
    settings: Settings | None = None,
) -> LightningBundleManifest:
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = detect_lightning_runtime(settings)
    train_path = output_dir / "lightning_train_tasks.jsonl"
    validation_path = output_dir / "lightning_validation_tasks.jsonl"
    transitions_path = output_dir / "lightning_transitions.jsonl"
    traces_path = output_dir / "lightning_traces.json"
    report_path = output_dir / "lightning_report.md"
    manifest_path = output_dir / "lightning_bundle_manifest.json"

    export_benchmark_tasks_jsonl(train_tasks, train_path)
    export_benchmark_tasks_jsonl(validation_tasks, validation_path)
    export_lightning_jsonl(traces, transitions_path)
    traces_path.write_text(dumps_pretty([trace.model_dump() for trace in traces]), encoding="utf-8")
    report_path.write_text(report_markdown, encoding="utf-8")

    notes = [
        runtime.reason,
        "The bundle is structured for Microsoft Agent Lightning prompt/policy optimization over benchmark tasks only.",
        "No online learning from real patient traffic is permitted in PRIORI-X.",
    ]
    if runtime.native_training_ready:
        notes.append("This environment can build a native APO trainer via build_native_lightning_recipe().")
    else:
        notes.append("Use the exported task JSONL files and transitions inside Linux/WSL2 to launch native Agent Lightning training.")

    manifest = LightningBundleManifest(
        runtime=runtime,
        train_tasks_path=str(train_path),
        validation_tasks_path=str(validation_path),
        transitions_path=str(transitions_path),
        traces_path=str(traces_path),
        report_path=str(report_path),
        prompt_template_baseline=baseline_prompt_template(),
        notes=notes,
    )
    manifest_path.write_text(dumps_pretty(manifest.model_dump()), encoding="utf-8")
    return manifest


def _import_agentlightning() -> Any:
    return import_module("agentlightning")


def create_lightning_rollout_agent(settings: Settings | None = None) -> Any:
    settings = settings or get_settings()
    agl = _import_agentlightning()
    orchestrator = PRIORIXOrchestrator(settings=settings)
    reward_model = CompositeRewardModel()

    @agl.rollout
    def priori_x_prompt_rollout(task: BenchmarkTask, prompt_template: Any) -> float:
        rendered_prompt = render_prompt_template(prompt_template, task.prompt)
        if hasattr(agl, "emit_object"):
            agl.emit_object({"task_id": task.task_id, "source_dataset": task.source_dataset, "task_type": task.task_type})
        report = orchestrator.analyze_text_case(task.task_id, rendered_prompt)
        reward = reward_model.score(task, report)
        if hasattr(agl, "emit_object"):
            agl.emit_object(
                {
                    "top_diagnosis": report.differential.ranked[0].slug if report.differential.ranked else None,
                    "recommended_test": report.next_best_tests[0].slug if report.next_best_tests else None,
                    "failure_categories": reward.failure_categories,
                }
            )
        if hasattr(agl, "emit_reward"):
            agl.emit_reward(reward.total_reward)
        return reward.total_reward

    return priori_x_prompt_rollout


def build_native_lightning_recipe(
    *,
    settings: Settings | None = None,
    n_runners: int = 4,
) -> NativeLightningRecipe:
    settings = settings or get_settings()
    runtime = detect_lightning_runtime(settings)
    if not runtime.native_training_ready:
        raise RuntimeError(runtime.reason)

    agl = _import_agentlightning()
    from openai import AsyncOpenAI

    agent = create_lightning_rollout_agent(settings=settings)
    algorithm = agl.APO(AsyncOpenAI())
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=n_runners,
        initial_resources={"prompt_template": baseline_prompt_template()},
        adapter=agl.TraceToMessages(),
    )
    return NativeLightningRecipe(agent=agent, trainer=trainer, runtime=runtime)
