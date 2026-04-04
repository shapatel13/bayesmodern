from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from utils.dates import utc_now
from utils.ids import make_id
from utils.jsonx import dumps_pretty


PromptSource = Literal["seed", "lightning_apo", "manual"]
PromptStatus = Literal["active", "candidate", "archived", "rejected"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEED_REGISTRY_PATH = _REPO_ROOT / "artifacts" / "models" / "prompt_registry.seed.json"
_RUNTIME_REGISTRY_PATH = _REPO_ROOT / "artifacts" / "models" / "prompt_registry.json"
_PEER_SYNC_ENV = "PRIORI_PROMPT_REGISTRY_SYNC_PATH"


class PromptRecord(BaseModel):
    version: str
    label: str
    description: str
    template: str
    source: PromptSource
    status: PromptStatus
    created_at: str
    based_on_version: str | None = None
    resources_id: str | None = None
    experiment_id: str | None = None
    notes: list[str] = Field(default_factory=list)


class PromptRegistry(BaseModel):
    active_version: str
    prompts: list[PromptRecord] = Field(default_factory=list)


def _running_in_wsl() -> bool:
    return bool(os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"))


def _peer_registry_candidates() -> list[Path]:
    configured = os.environ.get(_PEER_SYNC_ENV)
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))

    runtime_text = str(_RUNTIME_REGISTRY_PATH)
    if _running_in_wsl():
        default_windows_repo = Path("/mnt/c/Users/msmsh/Downloads/bayesmodern/artifacts/models/prompt_registry.json")
        if str(default_windows_repo) != runtime_text:
            candidates.append(default_windows_repo)
    elif os.name == "nt":
        default_wsl_repo = Path(r"\\wsl.localhost\Ubuntu\home\msmsharad\bayesmodern-wsl\artifacts\models\prompt_registry.json")
        if str(default_wsl_repo) != runtime_text:
            candidates.append(default_wsl_repo)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def get_prompt_registry_sync_target() -> str | None:
    candidates = _peer_registry_candidates()
    if not candidates:
        return None
    return str(candidates[0])


def _ensure_seed_registry() -> None:
    _SEED_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _SEED_REGISTRY_PATH.exists():
        return
    seed = PromptRegistry(
        active_version="v1-offline",
        prompts=[
            PromptRecord(
                version="v1-offline",
                label="Baseline Offline Research Prompt",
                description="Conservative baseline prompt template for offline PRIORI-X benchmark parsing and reasoning.",
                template=(
                    "You are PRIORI-X, a clinician-facing research workbench operating strictly in offline evaluation mode.\n"
                    "Analyze the following benchmark vignette conservatively, preserve uncertainty, cite limitations when appropriate, "
                    "and avoid treatment autopilot.\n\nCase:\n{task}"
                ),
                source="seed",
                status="active",
                created_at="2026-04-02T00:00:00+00:00",
                notes=[
                    "Seed prompt shipped with the repository.",
                    "Use as the baseline for Agent Lightning APO prompt optimization.",
                ],
            )
        ],
    )
    _SEED_REGISTRY_PATH.write_text(dumps_pretty(seed.model_dump()), encoding="utf-8")


def _initialize_runtime_registry() -> None:
    _ensure_seed_registry()
    if _RUNTIME_REGISTRY_PATH.exists():
        return
    _RUNTIME_REGISTRY_PATH.write_text(_SEED_REGISTRY_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def _write_registry_file(path: Path, registry: PromptRegistry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_pretty(registry.model_dump()), encoding="utf-8")


def _merge_prompt_registries(base: PromptRegistry, incoming: PromptRegistry) -> PromptRegistry:
    merged_by_version = {prompt.version: prompt for prompt in base.prompts}
    for prompt in incoming.prompts:
        merged_by_version[prompt.version] = prompt

    ordered_versions: list[str] = []
    for prompt in base.prompts + incoming.prompts:
        if prompt.version not in ordered_versions:
            ordered_versions.append(prompt.version)

    active_version = incoming.active_version if incoming.active_version in merged_by_version else base.active_version
    return PromptRegistry(
        active_version=active_version,
        prompts=[merged_by_version[version] for version in ordered_versions if version in merged_by_version],
    )


def _sync_from_peer_registry() -> None:
    if os.environ.get("PRIORI_DISABLE_PROMPT_REGISTRY_SYNC", "").lower() in {"1", "true", "yes"}:
        return
    if not _RUNTIME_REGISTRY_PATH.exists():
        return

    try:
        local_registry = PromptRegistry.model_validate_json(_RUNTIME_REGISTRY_PATH.read_text(encoding="utf-8"))
        local_mtime = _RUNTIME_REGISTRY_PATH.stat().st_mtime
    except Exception:
        return

    for peer_path in _peer_registry_candidates():
        if not peer_path.exists():
            continue
        try:
            peer_registry = PromptRegistry.model_validate_json(peer_path.read_text(encoding="utf-8"))
            peer_mtime = peer_path.stat().st_mtime
        except Exception:
            continue
        if peer_mtime <= local_mtime:
            continue
        merged = _merge_prompt_registries(local_registry, peer_registry)
        _write_registry_file(_RUNTIME_REGISTRY_PATH, merged)
        local_registry = merged
        local_mtime = _RUNTIME_REGISTRY_PATH.stat().st_mtime


def _mirror_registry_to_peer(registry: PromptRegistry) -> None:
    if os.environ.get("PRIORI_DISABLE_PROMPT_REGISTRY_SYNC", "").lower() in {"1", "true", "yes"}:
        return
    for peer_path in _peer_registry_candidates():
        try:
            if peer_path.exists():
                peer_registry = PromptRegistry.model_validate_json(peer_path.read_text(encoding="utf-8"))
                merged = _merge_prompt_registries(peer_registry, registry)
            else:
                merged = registry
            _write_registry_file(peer_path, merged)
        except Exception:
            continue


def _write_registry(registry: PromptRegistry, *, mirror: bool = True) -> None:
    _write_registry_file(_RUNTIME_REGISTRY_PATH, registry)
    if mirror:
        _mirror_registry_to_peer(registry)


def load_prompt_registry() -> PromptRegistry:
    _initialize_runtime_registry()
    _sync_from_peer_registry()
    return PromptRegistry.model_validate_json(_RUNTIME_REGISTRY_PATH.read_text(encoding="utf-8"))


def list_prompt_records() -> list[PromptRecord]:
    return load_prompt_registry().prompts


def get_prompt_record(version: str) -> PromptRecord:
    normalized = version.strip().lower()
    registry = load_prompt_registry()
    if normalized == "active":
        return get_active_prompt_record()
    for prompt in registry.prompts:
        if prompt.version.lower() == normalized:
            return prompt
    raise KeyError(f"Unknown prompt version: {version}")


def get_active_prompt_record() -> PromptRecord:
    registry = load_prompt_registry()
    for prompt in registry.prompts:
        if prompt.version == registry.active_version:
            return prompt
    raise KeyError(f"Active prompt version `{registry.active_version}` is missing from the registry.")


def resolve_prompt_record(prompt_version: str | None = None) -> PromptRecord:
    if not prompt_version or prompt_version.strip().lower() == "active":
        return get_active_prompt_record()
    return get_prompt_record(prompt_version)


def resolve_prompt_template(prompt_version: str | None = None) -> str:
    return resolve_prompt_record(prompt_version).template


def render_task_prompt(prompt_template: str, task_prompt: str) -> str:
    if "{task}" in prompt_template:
        return prompt_template.replace("{task}", task_prompt)
    return f"{prompt_template}\n\nCase:\n{task_prompt}".strip()


def register_prompt_candidate(
    *,
    template: str,
    label: str,
    description: str,
    based_on_version: str,
    source: PromptSource = "lightning_apo",
    resources_id: str | None = None,
    experiment_id: str | None = None,
    notes: list[str] | None = None,
) -> PromptRecord:
    registry = load_prompt_registry()
    version = f"apo-{make_id('prompt')}"
    record = PromptRecord(
        version=version,
        label=label,
        description=description,
        template=template,
        source=source,
        status="candidate",
        created_at=utc_now().isoformat(),
        based_on_version=based_on_version,
        resources_id=resources_id,
        experiment_id=experiment_id,
        notes=notes or [],
    )
    registry.prompts.append(record)
    _write_registry(registry)
    return record


def promote_prompt_version(version: str, *, notes: list[str] | None = None) -> PromptRegistry:
    registry = load_prompt_registry()
    normalized = version.strip().lower()
    promoted = False
    for index, prompt in enumerate(registry.prompts):
        updated_notes = list(prompt.notes)
        if prompt.version.lower() == normalized:
            promoted = True
            if notes:
                updated_notes.extend(notes)
            registry.prompts[index] = prompt.model_copy(update={"status": "active", "notes": updated_notes})
            registry.active_version = prompt.version
        elif prompt.status == "active":
            registry.prompts[index] = prompt.model_copy(update={"status": "archived"})
    if not promoted:
        raise KeyError(f"Unknown prompt version: {version}")
    _write_registry(registry)
    return registry


def reject_prompt_version(version: str, *, notes: list[str] | None = None) -> PromptRegistry:
    registry = load_prompt_registry()
    normalized = version.strip().lower()
    found = False
    for index, prompt in enumerate(registry.prompts):
        if prompt.version.lower() == normalized:
            found = True
            updated_notes = list(prompt.notes)
            if notes:
                updated_notes.extend(notes)
            registry.prompts[index] = prompt.model_copy(update={"status": "rejected", "notes": updated_notes})
    if not found:
        raise KeyError(f"Unknown prompt version: {version}")
    _write_registry(registry)
    return registry
