from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CurriculumAccessMode = Literal["public_hf", "credentialed", "hybrid"]


@dataclass(frozen=True)
class CurriculumComponent:
    dataset_key: str
    train_limit: int
    validation_limit: int
    subset: str | None = None
    train_split: str | None = None
    validation_split: str | None = None
    weight: int = 1
    optional: bool = False
    notes: str = ""


@dataclass(frozen=True)
class LightningCurriculumSpec:
    key: str
    label: str
    description: str
    objective: str
    components: tuple[CurriculumComponent, ...]
    access_mode: CurriculumAccessMode = "public_hf"
    notes: str = ""
    focus_areas: tuple[str, ...] = field(default_factory=tuple)


LIGHTNING_CURRICULA: dict[str, LightningCurriculumSpec] = {
    "broad_medical_feedback_lab": LightningCurriculumSpec(
        key="broad_medical_feedback_lab",
        label="Broad Medical Feedback Lab",
        description=(
            "Public Hugging Face curriculum blending broad medical QA, higher-difficulty reasoning, "
            "evidence verification, and rare-disease differential work."
        ),
        objective=(
            "Generate a Microsoft Agent Lightning-ready feedback curriculum for prompt and policy "
            "optimization across general medical reasoning."
        ),
        access_mode="public_hf",
        components=(
            CurriculumComponent("medmcqa", train_limit=20, validation_limit=10, weight=2),
            CurriculumComponent("medqa", train_limit=20, validation_limit=10, weight=2),
            CurriculumComponent("pubmedqa", train_limit=16, validation_limit=8, subset="pqa_labeled", weight=1),
            CurriculumComponent("findzebra", train_limit=12, validation_limit=6, weight=1),
        ),
        notes="Best starting curriculum for broad offline RL-style prompt optimization using public benchmarks.",
        focus_areas=("diagnostic ranking", "calibration", "evidence grounding", "rare disease coverage"),
    ),
    "diagnostic_reasoning_feedback_lab": LightningCurriculumSpec(
        key="diagnostic_reasoning_feedback_lab",
        label="Diagnostic Reasoning Feedback Lab",
        description="Focused curriculum for broad and difficult diagnosis-style medical multiple-choice reasoning.",
        objective="Strengthen differential ranking, distractor handling, and next-best-step reasoning.",
        access_mode="public_hf",
        components=(
            CurriculumComponent("medmcqa", train_limit=24, validation_limit=12, weight=2),
            CurriculumComponent("medqa", train_limit=24, validation_limit=12, weight=3),
        ),
        notes="Higher emphasis on MedQA to sharpen harder reasoning patterns for offline optimization.",
        focus_areas=("diagnostic ranking", "distractor resistance", "policy repair"),
    ),
    "evidence_rare_feedback_lab": LightningCurriculumSpec(
        key="evidence_rare_feedback_lab",
        label="Evidence And Rare Feedback Lab",
        description="Mixed curriculum for abstention, provenance-aware claims, and open-ended rare-disease reasoning.",
        objective="Improve cautious evidence use, uncertainty calibration, and broad hypothesis generation.",
        access_mode="public_hf",
        components=(
            CurriculumComponent("pubmedqa", train_limit=18, validation_limit=9, subset="pqa_labeled", weight=2),
            CurriculumComponent("findzebra", train_limit=18, validation_limit=9, weight=2),
        ),
        notes="Useful when you want Lightning feedback to reduce unsupported claims and false certainty.",
        focus_areas=("evidence verification", "abstention", "rare disease differential"),
    ),
    "clinical_safety_feedback_lab": LightningCurriculumSpec(
        key="clinical_safety_feedback_lab",
        label="Clinical Safety Feedback Lab",
        description="Credentialed safety curriculum for triage and medication-safety research workflows.",
        objective="Improve under-triage avoidance, medication extraction, and adverse-event detection policies.",
        access_mode="credentialed",
        components=(
            CurriculumComponent("mietic", train_limit=32, validation_limit=16, weight=2),
            CurriculumComponent("n2c2_2018_track2", train_limit=32, validation_limit=16, weight=2),
        ),
        notes="Requires local credentialed data paths and is best used after the public HF curriculum is stable.",
        focus_areas=("triage safety", "medication safety", "hard-veto reduction"),
    ),
}


def get_lightning_curriculum(curriculum_key: str) -> LightningCurriculumSpec:
    normalized = curriculum_key.strip().lower()
    if normalized not in LIGHTNING_CURRICULA:
        raise KeyError(f"Unsupported Lightning curriculum: {curriculum_key}")
    return LIGHTNING_CURRICULA[normalized]


def list_lightning_curricula() -> list[LightningCurriculumSpec]:
    return list(LIGHTNING_CURRICULA.values())
