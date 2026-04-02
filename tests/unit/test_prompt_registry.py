from __future__ import annotations

import json

from agent import prompt_registry


def test_prompt_registry_bootstraps_and_promotes_candidate(tmp_path, monkeypatch) -> None:
    seed_path = tmp_path / "prompt_registry.seed.json"
    runtime_path = tmp_path / "prompt_registry.json"
    monkeypatch.setattr(prompt_registry, "_SEED_REGISTRY_PATH", seed_path)
    monkeypatch.setattr(prompt_registry, "_RUNTIME_REGISTRY_PATH", runtime_path)

    active = prompt_registry.get_active_prompt_record()
    assert active.version == "v1-offline"

    candidate = prompt_registry.register_prompt_candidate(
        template="Updated template\n\nCase:\n{task}",
        label="Candidate Prompt",
        description="Candidate from a unit test.",
        based_on_version=active.version,
    )
    registry = prompt_registry.promote_prompt_version(candidate.version)

    assert registry.active_version == candidate.version
    promoted = prompt_registry.get_active_prompt_record()
    assert promoted.template.startswith("Updated template")

    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert payload["active_version"] == candidate.version


def test_render_task_prompt_inserts_case_text() -> None:
    rendered = prompt_registry.render_task_prompt("Header\n{task}", "Clinical vignette")
    assert rendered == "Header\nClinical vignette"
