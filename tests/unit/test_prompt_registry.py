from __future__ import annotations

import json
import os

from agent import prompt_registry


def test_prompt_registry_bootstraps_and_promotes_candidate(tmp_path, monkeypatch) -> None:
    seed_path = tmp_path / "prompt_registry.seed.json"
    runtime_path = tmp_path / "prompt_registry.json"
    monkeypatch.setattr(prompt_registry, "_SEED_REGISTRY_PATH", seed_path)
    monkeypatch.setattr(prompt_registry, "_RUNTIME_REGISTRY_PATH", runtime_path)
    monkeypatch.setattr(prompt_registry, "_peer_registry_candidates", lambda: [])

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


def test_render_task_prompt_preserves_other_literal_braces() -> None:
    rendered = prompt_registry.render_task_prompt(
        "Risk buckets {low, medium, high}\nCase:\n{task}",
        "Clinical vignette",
    )
    assert rendered == "Risk buckets {low, medium, high}\nCase:\nClinical vignette"


def test_load_prompt_registry_imports_newer_peer_registry(tmp_path, monkeypatch) -> None:
    seed_path = tmp_path / "prompt_registry.seed.json"
    runtime_path = tmp_path / "prompt_registry.json"
    peer_path = tmp_path / "peer" / "prompt_registry.json"
    monkeypatch.setattr(prompt_registry, "_SEED_REGISTRY_PATH", seed_path)
    monkeypatch.setattr(prompt_registry, "_RUNTIME_REGISTRY_PATH", runtime_path)
    monkeypatch.setenv("PRIORI_PROMPT_REGISTRY_SYNC_PATH", str(peer_path))
    monkeypatch.setattr(prompt_registry, "_peer_registry_candidates", lambda: [peer_path])

    active = prompt_registry.get_active_prompt_record()
    assert active.version == "v1-offline"

    peer_registry = prompt_registry.PromptRegistry(
        active_version="apo-prompt_sync",
        prompts=[
            active.model_copy(update={"status": "archived"}),
            prompt_registry.PromptRecord(
                version="apo-prompt_sync",
                label="Peer Prompt",
                description="Promoted in peer repo.",
                template="Peer template\n\nCase:\n{task}",
                source="lightning_apo",
                status="active",
                created_at="2026-04-04T17:00:00+00:00",
                based_on_version="v1-offline",
                notes=["Imported from peer registry."],
            ),
        ],
    )
    peer_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_registry._write_registry_file(peer_path, peer_registry)
    os.utime(peer_path, (runtime_path.stat().st_atime + 5, runtime_path.stat().st_mtime + 5))

    loaded = prompt_registry.load_prompt_registry()

    assert loaded.active_version == "apo-prompt_sync"
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert payload["active_version"] == "apo-prompt_sync"


def test_promote_prompt_version_mirrors_to_peer_registry(tmp_path, monkeypatch) -> None:
    seed_path = tmp_path / "prompt_registry.seed.json"
    runtime_path = tmp_path / "prompt_registry.json"
    peer_path = tmp_path / "peer" / "prompt_registry.json"
    monkeypatch.setattr(prompt_registry, "_SEED_REGISTRY_PATH", seed_path)
    monkeypatch.setattr(prompt_registry, "_RUNTIME_REGISTRY_PATH", runtime_path)
    monkeypatch.setenv("PRIORI_PROMPT_REGISTRY_SYNC_PATH", str(peer_path))
    monkeypatch.setattr(prompt_registry, "_peer_registry_candidates", lambda: [peer_path])

    active = prompt_registry.get_active_prompt_record()
    candidate = prompt_registry.register_prompt_candidate(
        template="Mirrored candidate\n\nCase:\n{task}",
        label="Mirrored Candidate",
        description="Candidate that should sync to the peer path.",
        based_on_version=active.version,
    )
    registry = prompt_registry.promote_prompt_version(candidate.version)

    assert registry.active_version == candidate.version
    peer_payload = json.loads(peer_path.read_text(encoding="utf-8"))
    assert peer_payload["active_version"] == candidate.version
    mirrored_versions = {prompt["version"] for prompt in peer_payload["prompts"]}
    assert candidate.version in mirrored_versions
