from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchPreset:
    key: str
    label: str
    dataset_key: str
    description: str
    clinical_mode: str
    task_family: str
    train_limit: int
    validation_limit: int
    subset: str | None = None
    prompt_version: str = "v1-offline"
    policy_version: str = "v1-deterministic"
    requires_credentials: bool = False
    notes: str = ""


RESEARCH_PRESETS: dict[str, ResearchPreset] = {
    "clinical_reasoning_demo_lab": ResearchPreset(
        key="clinical_reasoning_demo_lab",
        label="Clinical Reasoning Demo Lab",
        dataset_key="priorix_demo_cases",
        description="Bundled local demo cases for immediate differential and next-best-test benchmarking.",
        clinical_mode="diagnostic_cockpit",
        task_family="diagnosis_open",
        train_limit=6,
        validation_limit=3,
        notes="Best first-run preset when you want a no-download benchmark immediately.",
    ),
    "core_diagnostic_lab": ResearchPreset(
        key="core_diagnostic_lab",
        label="Core Diagnostic Lab",
        dataset_key="medmcqa",
        description="Broad medical differential and next-best-step benchmark for generalist reasoning.",
        clinical_mode="diagnostic_cockpit",
        task_family="diagnosis_mcq",
        train_limit=24,
        validation_limit=12,
        notes="Strong baseline prompt-optimization track for broad medical knowledge.",
    ),
    "usmle_reasoning_lab": ResearchPreset(
        key="usmle_reasoning_lab",
        label="USMLE Reasoning Lab",
        dataset_key="medqa",
        description="Higher-difficulty clinical reasoning and distractor handling benchmark.",
        clinical_mode="diagnostic_cockpit",
        task_family="diagnosis_mcq",
        train_limit=24,
        validation_limit=12,
        notes="Useful for stronger reasoning prompts and contradiction repair policies.",
    ),
    "rare_disease_lab": ResearchPreset(
        key="rare_disease_lab",
        label="Rare Disease Lab",
        dataset_key="findzebra",
        description="Open differential benchmark for uncommon and syndromic presentations.",
        clinical_mode="rare_disease_workbench",
        task_family="diagnosis_open",
        train_limit=16,
        validation_limit=8,
        notes="Pushes broad hypothesis generation and calibrated uncertainty.",
    ),
    "evidence_verifier_lab": ResearchPreset(
        key="evidence_verifier_lab",
        label="Evidence Verifier Lab",
        dataset_key="pubmedqa",
        description="Biomedical evidence-verification benchmark for abstention and provenance-aware claims.",
        clinical_mode="evidence_lab",
        task_family="evidence_verification",
        train_limit=24,
        validation_limit=12,
        subset="pqa_labeled",
        notes="Good target for calibration and unsupported-claim reduction.",
    ),
    "generation_audit_lab": ResearchPreset(
        key="generation_audit_lab",
        label="Generation Audit Lab",
        dataset_key="medval_bench",
        description="Physician-labeled medical generation audit benchmark for risk grading and output blocking.",
        clinical_mode="generation_audit",
        task_family="generation_audit",
        train_limit=64,
        validation_limit=24,
        requires_credentials=True,
        notes="Requires PRIORI_MEDVAL_BENCH_PATH pointing at the MedVAL-Bench CSV export.",
    ),
    "ed_triage_lab": ResearchPreset(
        key="ed_triage_lab",
        label="ED Triage Lab",
        dataset_key="mietic",
        description="Emergency department acuity and under-triage benchmark built from a credential-gated triage corpus.",
        clinical_mode="triage_cockpit",
        task_family="triage",
        train_limit=32,
        validation_limit=16,
        requires_credentials=True,
        notes="Requires PRIORI_MIETIC_PATH pointing at a local PhysioNet export.",
    ),
    "ed_triage_demo_lab": ResearchPreset(
        key="ed_triage_demo_lab",
        label="ED Triage Demo Lab",
        dataset_key="mietic_demo",
        description="Bundled local triage cases for immediate ED-acuity benchmarking.",
        clinical_mode="triage_cockpit",
        task_family="triage",
        train_limit=6,
        validation_limit=3,
        notes="No extra credentials needed; good for first-morning smoke tests.",
    ),
    "medication_safety_lab": ResearchPreset(
        key="medication_safety_lab",
        label="Medication Safety Lab",
        dataset_key="n2c2_2018_track2",
        description="Medication extraction and adverse-drug-event benchmark for clinician safety workflows.",
        clinical_mode="medication_safety",
        task_family="medication_safety",
        train_limit=32,
        validation_limit=16,
        requires_credentials=True,
        notes="Requires PRIORI_N2C2_2018_TRACK2_PATH pointing at a local export.",
    ),
    "medication_safety_demo_lab": ResearchPreset(
        key="medication_safety_demo_lab",
        label="Medication Safety Demo Lab",
        dataset_key="n2c2_demo",
        description="Bundled medication and adverse-event extraction cases for immediate safety benchmarking.",
        clinical_mode="medication_safety",
        task_family="medication_safety",
        train_limit=6,
        validation_limit=3,
        notes="No extra credentials needed; good for first-morning safety validation.",
    ),
}


def get_research_preset(preset_key: str) -> ResearchPreset:
    normalized = preset_key.strip().lower()
    if normalized not in RESEARCH_PRESETS:
        raise KeyError(f"Unsupported research preset: {preset_key}")
    return RESEARCH_PRESETS[normalized]


def list_research_presets() -> list[ResearchPreset]:
    return list(RESEARCH_PRESETS.values())
