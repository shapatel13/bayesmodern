from __future__ import annotations

import json
import os
import sys
import subprocess
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.orchestrator import PRIORIXOrchestrator
from agent.prompt_registry import render_task_prompt, resolve_prompt_record
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
    wsl_distribution_installed: bool | None = None
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
    prompt_version: str = "v1-offline"
    baseline_policy_version: str = "v1-deterministic"
    curriculum_key: str | None = None
    component_datasets: list[str] = Field(default_factory=list)
    reward_profiles: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class NativeLightningRecipe:
    agent: Any
    trainer: Any
    runtime: LightningRuntimeStatus


def _running_in_wsl() -> bool:
    return bool(os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"))


def _windows_wsl_distribution_installed() -> bool | None:
    if sys.platform != "win32" or _running_in_wsl():
        return None
    try:
        result = subprocess.run(
            ["wsl.exe", "--list", "--quiet"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def _agentlightning_version() -> str | None:
    try:
        return version("agentlightning")
    except PackageNotFoundError:
        return None


def _apo_dependency_issue() -> str | None:
    try:
        import_module("agentlightning.algorithm.apo")
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "unknown module"
        return (
            f"Microsoft Agent Lightning APO is installed incompletely; missing dependency `{missing_name}`. "
            "Install the APO extras with `python -m pip install -e \".[lightning]\"` or `python -m pip install \"agentlightning[apo]>=0.3.0\" poml`."
        )
    except Exception as exc:
        return f"Microsoft Agent Lightning APO could not be imported cleanly: {exc}"
    return None


def detect_lightning_runtime(settings: Settings | None = None) -> LightningRuntimeStatus:
    settings = settings or get_settings()
    package_version = _agentlightning_version()
    package_available = package_version is not None
    apo_dependency_issue = _apo_dependency_issue() if package_available else None
    platform_supported = sys.platform != "win32" or _running_in_wsl()
    wsl_distribution_installed = _windows_wsl_distribution_installed()
    llm_ready = settings.allow_live_llm and bool(settings.openai_api_key)
    native_training_ready = package_available and apo_dependency_issue is None and platform_supported and llm_ready

    if not package_available:
        reason = "Microsoft Agent Lightning package not installed; exporting sandbox bundle only."
    elif apo_dependency_issue is not None:
        reason = apo_dependency_issue
    elif sys.platform == "win32" and not _running_in_wsl() and wsl_distribution_installed is False:
        reason = (
            "Native Microsoft Agent Lightning training requires Linux or WSL2, but no WSL distro is installed. "
            "Install Ubuntu with `wsl --install Ubuntu`, then rerun training inside WSL."
        )
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
        wsl_distribution_installed=wsl_distribution_installed,
        reason=reason,
    )


def traces_to_lightning_transitions(traces: list[ExperimentTrace]) -> list[LightningTransition]:
    transitions: list[LightningTransition] = []
    for trace in traces:
        top = trace.report.differential.ranked[0] if trace.report.differential.ranked else None
        top_mechanism = trace.report.mechanism_states.ranked[0] if trace.report.mechanism_states.ranked else None
        recommendation = trace.report.next_best_tests[0] if trace.report.next_best_tests else None
        task_metadata = dict(trace.task_metadata)
        transitions.append(
            LightningTransition(
                task_id=trace.task_id,
                state={
                    "prompt_version": trace.prompt_version,
                    "policy_version": trace.policy_version,
                    "triage": trace.report.triage.urgency,
                    "task_type": trace.task_type,
                    "source_dataset": trace.source_dataset,
                    "curriculum_key": task_metadata.get("curriculum_key"),
                    "curriculum_component": task_metadata.get("curriculum_component"),
                    "reward_profile": trace.reward.reward_profile if trace.reward else "unknown",
                    "reasoning_mode": trace.report.reasoning_runtime.mode,
                    "open_world_triggered": trace.report.reasoning_runtime.open_world_triggered,
                    "needs_clinician_review": trace.report.decision_quality.needs_clinician_review,
                    "low_signal_case": trace.report.decision_quality.low_signal_case,
                    "mixed_mechanism_uncertainty": trace.report.decision_quality.mixed_mechanism_uncertainty,
                },
                action={
                    "top_diagnosis": top.slug if top else None,
                    "top_mechanism": top_mechanism.slug if top_mechanism else None,
                    "active_mechanisms": trace.report.mechanism_states.active_states,
                    "recommended_test": recommendation.slug if recommendation else None,
                    "recommended_test_target_states": recommendation.target_states if recommendation else [],
                    "medications": trace.report.context.medications,
                    "adverse_events": trace.report.context.adverse_events,
                    "generation_audit_action": (
                        trace.report.generation_audit.recommended_action if trace.report.generation_audit else None
                    ),
                    "predicted_risk_grade": (
                        trace.report.generation_audit.predicted_risk_grade if trace.report.generation_audit else None
                    ),
                },
                reward=trace.reward.total_reward if trace.reward else 0.0,
                done=True,
                info={
                    "failure_categories": trace.reward.failure_categories if trace.reward else [],
                    "hard_veto": trace.reward.hard_veto if trace.reward else False,
                    "medication_extraction_quality": trace.reward.medication_extraction_quality if trace.reward else None,
                    "adverse_event_quality": trace.reward.adverse_event_quality if trace.reward else None,
                    "claim_alignment_quality": trace.reward.claim_alignment_quality if trace.reward else None,
                    "generation_audit_quality": trace.reward.generation_audit_quality if trace.reward else None,
                    "reward_profile_hint": task_metadata.get("reward_profile_hint"),
                    "source_hf_dataset": task_metadata.get("source_hf_dataset"),
                    "source_task_family": task_metadata.get("source_task_family"),
                    "gold_risk_grade": task_metadata.get("physician_risk_grade"),
                    "contradictions": trace.report.contradictions,
                    "provenance_warnings": trace.report.provenance_warnings,
                    "mechanism_summary": trace.report.mechanism_states.summary,
                    "mechanism_mixed_physiology": trace.report.mechanism_states.mixed_physiology,
                    "decision_quality_reasons": trace.report.decision_quality.reasons,
                    "structured_signal_count": trace.report.decision_quality.structured_signal_count,
                    "top_differential_gap": trace.report.decision_quality.top_differential_gap,
                    "reasoning_gate_reason": trace.report.reasoning_runtime.gate_reason,
                    "reasoning_notes": trace.report.reasoning_runtime.notes,
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


def render_prompt_template(prompt_template: Any, task_prompt: str) -> str:
    if hasattr(prompt_template, "format"):
        try:
            return prompt_template.format(task=task_prompt)
        except TypeError:
            pass
    template_text = str(prompt_template)
    return render_task_prompt(template_text, task_prompt)


def export_lightning_bundle(
    *,
    train_tasks: list[BenchmarkTask],
    validation_tasks: list[BenchmarkTask],
    traces: list[ExperimentTrace],
    report_markdown: str,
    output_dir: Path,
    settings: Settings | None = None,
    prompt_version: str = "active",
    policy_version: str = "v1-deterministic",
    curriculum_key: str | None = None,
    component_datasets: list[str] | None = None,
    reward_profiles: list[str] | None = None,
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
    prompt_record = resolve_prompt_record(prompt_version)

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
    if curriculum_key:
        notes.append(f"Curriculum `{curriculum_key}` mixes component datasets: {', '.join(component_datasets or [])}.")
    if reward_profiles:
        notes.append(f"Reward profiles covered: {', '.join(reward_profiles)}.")
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
        prompt_template_baseline=prompt_record.template,
        prompt_version=prompt_record.version,
        baseline_policy_version=policy_version,
        curriculum_key=curriculum_key,
        component_datasets=component_datasets or [],
        reward_profiles=reward_profiles or [],
        notes=notes,
    )
    manifest_path.write_text(dumps_pretty(manifest.model_dump()), encoding="utf-8")
    return manifest


def _import_agentlightning() -> Any:
    return import_module("agentlightning")


def _coerce_benchmark_task(task: BenchmarkTask | dict[str, Any]) -> BenchmarkTask:
    if isinstance(task, BenchmarkTask):
        return task
    return BenchmarkTask.model_validate(task)


def create_lightning_rollout_agent(
    settings: Settings | None = None,
    *,
    policy_version: str = "v1-deterministic",
) -> Any:
    settings = settings or get_settings()
    agl = _import_agentlightning()
    reward_model = CompositeRewardModel()

    @agl.rollout
    def priori_x_prompt_rollout(task: BenchmarkTask | dict[str, Any], prompt_template: Any) -> float:
        normalized_task = _coerce_benchmark_task(task)
        orchestrator = PRIORIXOrchestrator(settings=settings, policy_version=policy_version)
        rendered_prompt = render_prompt_template(prompt_template, normalized_task.prompt)
        if hasattr(agl, "emit_object"):
            agl.emit_object(
                {
                    "task_id": normalized_task.task_id,
                    "source_dataset": normalized_task.source_dataset,
                    "task_type": normalized_task.task_type,
                    "policy_version": policy_version,
                }
            )
        report = orchestrator.analyze_text_case(
            normalized_task.task_id,
            normalized_task.prompt,
            policy_version=policy_version,
            prompt_template=str(prompt_template),
        )
        reward = reward_model.score(normalized_task, report)
        if hasattr(agl, "emit_object"):
            agl.emit_object(
                {
                    "top_diagnosis": report.differential.ranked[0].slug if report.differential.ranked else None,
                    "recommended_test": report.next_best_tests[0].slug if report.next_best_tests else None,
                    "rendered_prompt_preview": rendered_prompt[:200],
                    "policy_version": policy_version,
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
    prompt_version: str = "active",
    policy_version: str = "v1-deterministic",
    n_runners: int = 4,
) -> NativeLightningRecipe:
    settings = settings or get_settings()
    runtime = detect_lightning_runtime(settings)
    if not runtime.native_training_ready:
        raise RuntimeError(runtime.reason)

    agl = _import_agentlightning()
    from openai import AsyncOpenAI

    prompt_record = resolve_prompt_record(prompt_version)
    agent = create_lightning_rollout_agent(settings=settings, policy_version=policy_version)
    algorithm = agl.APO(
        AsyncOpenAI(api_key=settings.openai_api_key),
        gradient_model=settings.openai_reasoning_model,
        apply_edit_model=settings.openai_verifier_model,
    )
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=n_runners,
        initial_resources={
            "prompt_template": agl.PromptTemplate(template=prompt_record.template, engine="f-string"),
        },
        adapter=agl.TraceToMessages(),
    )
    return NativeLightningRecipe(agent=agent, trainer=trainer, runtime=runtime)
