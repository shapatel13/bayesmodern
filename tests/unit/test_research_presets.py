from __future__ import annotations

from eval.presets import get_research_preset, list_research_presets


def test_research_presets_include_specialty_labs() -> None:
    keys = {preset.key for preset in list_research_presets()}
    assert {"ed_triage_lab", "medication_safety_lab", "rare_disease_lab"} <= keys


def test_get_research_preset_returns_expected_dataset() -> None:
    preset = get_research_preset("medication_safety_lab")
    assert preset.dataset_key == "n2c2_2018_track2"
    assert preset.requires_credentials is True
