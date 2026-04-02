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
    "crackles": "Crackles",
    "orthopnea": "Orthopnea",
    "leg_edema": "Leg edema",
    "pressure_chest_pain": "Pressure-like or crushing substernal chest pain",
    "troponin_positive": "Positive troponin",
    "diaphoresis": "Diaphoresis",
    "pain_radiation": "Radiation to arm or jaw",
    "purulent_sputum": "Purulent sputum",
}


class OpenAIParsedContext(BaseModel):
    specialty: str = "general_internal_medicine"
    positive_finding_keys: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    adverse_events: list[str] = Field(default_factory=list)
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
                    "Also extract explicit medication names and explicit adverse-event or harm mentions as short phrases. "
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
        medications=[item.strip().lower() for item in result.medications if item.strip()],
        adverse_events=[item.strip().lower() for item in result.adverse_events if item.strip()],
        renal_impairment=result.renal_impairment,
        hemodynamic_instability=result.hemodynamic_instability,
        critical_values_present=result.critical_values_present,
    )
