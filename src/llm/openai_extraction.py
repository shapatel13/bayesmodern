from __future__ import annotations

from pydantic import BaseModel, Field

from core.models import ClinicalDecisionContext, ClinicalFinding
from agent.prompt_registry import render_task_prompt
from llm.openai_client import build_openai_client
from utils.config import Settings


SUPPORTED_FINDINGS: dict[str, str] = {
    "pleuritic_chest_pain": "Pleuritic chest pain",
    "tachycardia": "Tachycardia",
    "hypoxemia": "Hypoxemia",
    "fever": "Fever",
    "hemodynamic_instability": "Hemodynamic instability",
    "crackles": "Crackles",
    "orthopnea": "Orthopnea",
    "leg_edema": "Leg edema",
    "elevated_jvp": "Elevated jugular venous pressure",
    "cool_extremities": "Cool extremities",
    "warm_extremities": "Warm extremities",
    "oliguria": "Oliguria",
    "reduced_ef": "Reduced ejection fraction",
    "rv_strain": "RV strain",
    "ecg_ischemia": "Ischemic ECG changes",
    "pressure_chest_pain": "Pressure-like or crushing substernal chest pain",
    "troponin_positive": "Positive troponin",
    "diaphoresis": "Diaphoresis",
    "pain_radiation": "Radiation to arm or jaw",
    "purulent_sputum": "Purulent sputum",
    "anticoagulated": "On anticoagulation",
    "active_gi_bleeding": "Active gastrointestinal bleeding",
    "melena": "Melena",
    "symptomatic_anemia": "Symptomatic anemia",
    "low_hemoglobin": "Low hemoglobin",
    "severe_anemia": "Severe anemia",
    "supratherapeutic_inr": "Supratherapeutic INR",
    "elevated_lactate": "Elevated lactate",
    "bnp_elevated": "Elevated BNP/NT-proBNP",
    "hyponatremia": "Hyponatremia",
}


class OpenAIParsedContext(BaseModel):
    specialty: str = "general_internal_medicine"
    positive_finding_keys: list[str] = Field(default_factory=list)
    completed_tests: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    adverse_events: list[str] = Field(default_factory=list)
    age_years: int | None = None
    pregnant: bool = False
    hemodynamic_instability: bool = False
    critical_values_present: bool = False
    renal_impairment: bool = False
    parser_notes: list[str] = Field(default_factory=list)


def extract_context_with_openai(
    case_id: str,
    note_text: str,
    settings: Settings,
    *,
    prompt_template: str | None = None,
) -> ClinicalDecisionContext:
    client = build_openai_client(settings)
    finding_inventory = ", ".join(f"{key}: {label}" for key, label in SUPPORTED_FINDINGS.items())
    parser_prompt = render_task_prompt(prompt_template, note_text) if prompt_template else note_text
    parsed = client.responses.parse(
        model=settings.openai_parser_model,
        reasoning={"effort": "low"},
        text_format=OpenAIParsedContext,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a conservative clinical note parser for an offline research system. "
                    "Extract only supported structured findings explicitly or strongly implied in the note. "
                    "For ischemic chest pain language, map crushing, substernal pressure, heavy pressure, or arm/jaw radiation "
                    "to the supported chest-pain findings when clearly present. "
                    "For anticoagulation-bleeding language, map warfarin or other anticoagulant exposure, melena, "
                    "hematemesis, GI bleed phrasing, and symptomatic anemia to the supported bleeding findings when clearly present. "
                    "Also extract explicit medication names, explicit adverse-event or harm mentions as short phrases, "
                    "and clearly already-completed tests such as ECG, troponin, BNP, chest radiograph, D-dimer, CTA, INR, CBC, echo, endoscopy, or type and screen. "
                    "Do not diagnose. Do not invent findings. Output only the structured schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Supported findings: {finding_inventory}\n\n"
                    f"Clinical note:\n{parser_prompt}"
                ),
            },
        ],
    )
    result = parsed.output_parsed
    findings = [
        ClinicalFinding(key=key, label=SUPPORTED_FINDINGS[key], present=True, source_type="llm_inferred")
        for key in result.positive_finding_keys
        if key in SUPPORTED_FINDINGS
    ]
    return ClinicalDecisionContext(
        case_id=case_id,
        specialty=result.specialty,
        symptoms_free_text=note_text,
        findings=findings,
        completed_tests=[item.strip().lower() for item in result.completed_tests if item.strip()],
        medications=[item.strip().lower() for item in result.medications if item.strip()],
        adverse_events=[item.strip().lower() for item in result.adverse_events if item.strip()],
        age_years=result.age_years,
        pregnant=result.pregnant,
        renal_impairment=result.renal_impairment,
        hemodynamic_instability=result.hemodynamic_instability,
        critical_values_present=result.critical_values_present,
    )
