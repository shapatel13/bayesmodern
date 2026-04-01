from __future__ import annotations

from core.models import ClinicalDecisionContext


def build_reasoning_prompt(context: ClinicalDecisionContext) -> str:
    findings = ", ".join(finding.label for finding in context.findings if finding.present)
    return (
        "You are assisting a physician-researcher in an offline evaluation sandbox. "
        "Prioritize Bayesian reasoning, provenance, uncertainty, and safety.\n"
        f"Case context: {context.symptoms_free_text or 'No free-text note provided.'}\n"
        f"Structured findings: {findings or 'No structured positive findings provided.'}"
    )

