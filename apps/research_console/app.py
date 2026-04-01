from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from agent.orchestrator import PRIORIXOrchestrator
from security.secrets import validate_live_llm_config
from ui.charts.probabilities import calibration_figure, differential_figure
from ui.components.report_cards import headline_cards
from ui.viewmodels.cockpit import differential_rows, next_test_rows, provenance_rows
from utils.config import get_settings


def main() -> None:
    settings = get_settings()
    secret_status = validate_live_llm_config(settings)
    orchestrator = PRIORIXOrchestrator(settings)

    st.set_page_config(page_title="PRIORI-X Research Console", page_icon="PX", layout="wide")
    st.markdown(
        """
        <style>
            :root {
                --pri-slate: #0f172a;
                --pri-panel: #f8fafc;
                --pri-accent: #0b7285;
                --pri-amber: #f08c00;
                --pri-green: #2f9e44;
            }
            .pri-shell {
                padding: 1.25rem 1.5rem;
                border-radius: 24px;
                background:
                    radial-gradient(circle at top right, rgba(11,114,133,0.14), transparent 28%),
                    linear-gradient(145deg, #f8fbfd 0%, #eef5f8 100%);
                border: 1px solid rgba(15, 23, 42, 0.08);
                box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
                margin-bottom: 1rem;
            }
            .pri-kpi {
                padding: 0.9rem 1rem;
                border-radius: 18px;
                background: white;
                border: 1px solid rgba(15, 23, 42, 0.08);
                min-height: 110px;
            }
            .pri-kpi-label {
                color: #475569;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            .pri-kpi-value {
                color: var(--pri-slate);
                font-size: 1.2rem;
                font-weight: 700;
                margin-top: 0.3rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="pri-shell">', unsafe_allow_html=True)
    st.title("PRIORI-X")
    st.caption("Academic decision-science lab plus premium clinical cockpit for offline physician-facing research.")

    st.warning(
        "Research only. PRIORI-X is not a clinical autopilot and must not be used for autonomous "
        "patient care decisions."
    )

    sample_case = (
        "52-year-old with pleuritic chest pain, tachycardia, and hypoxemia after recent immobility. "
        "No fever. Concern for pulmonary embolism versus pneumonia."
    )
    with st.sidebar:
        st.header("Research Mode")
        st.markdown(
            "\n".join(
                [
                    f"- Safety mode: `{settings.safety_mode}`",
                    f"- Live external LLMs: `{settings.allow_live_llm}`",
                    f"- Provider: `{settings.default_model_provider}`",
                    f"- OpenAI parser: `{settings.openai_parser_model}`",
                    f"- OpenAI reasoner: `{settings.openai_reasoning_model}`",
                    f"- Provider ready: `{secret_status.provider_ready}`",
                    f"- Trace redaction: `{settings.redact_traces}`",
                    f"- Namespace: `{settings.experiment_namespace}`",
                ]
            )
        )
        if st.button("Load PE Demo"):
            st.session_state["case_text"] = sample_case
        st.caption("Every output remains inspectable, versionable, and explicitly research-only.")

    st.session_state.setdefault("case_text", sample_case)
    case_text = st.text_area("Case Intake", value=st.session_state["case_text"], height=180)
    run = st.button("Analyze Case", type="primary", use_container_width=True)

    if run:
        report = orchestrator.analyze_text_case("console-case", case_text)
        st.session_state["report_json"] = report.model_dump()

    report_json = st.session_state.get("report_json")
    if report_json:
        from llm.structured_output import ResearchReport

        report = ResearchReport.model_validate(report_json)
        kpis = headline_cards(report)
        columns = st.columns(len(kpis))
        for column, (label, value) in zip(columns, kpis, strict=True):
            column.markdown(
                f'<div class="pri-kpi"><div class="pri-kpi-label">{label}</div><div class="pri-kpi-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Diagnostic Cockpit", "Next Best Test", "Safety & Provenance", "Calibration Lab", "Audit Trail"]
        )
        with tab1:
            left, right = st.columns([1.2, 1])
            left.plotly_chart(differential_figure(report), use_container_width=True)
            right.dataframe(pd.DataFrame(differential_rows(report)), use_container_width=True, hide_index=True)
        with tab2:
            st.dataframe(pd.DataFrame(next_test_rows(report)), use_container_width=True, hide_index=True)
            for recommendation in report.next_best_tests[:3]:
                st.info(f"{recommendation.name}: {recommendation.rationale}")
        with tab3:
            st.metric("Triage", report.triage.urgency.title())
            st.metric("Threshold Action", report.threshold_decision.action.replace("_", " ").title())
            st.dataframe(pd.DataFrame(provenance_rows(report)), use_container_width=True, hide_index=True)
            if report.contradictions:
                st.error("\n".join(report.contradictions))
            if report.provenance_warnings:
                st.warning("\n".join(report.provenance_warnings))
        with tab4:
            left, right = st.columns([1, 1])
            left.plotly_chart(calibration_figure(report), use_container_width=True)
            right.json(
                {
                    "top_hypothesis": report.differential.ranked[0].name if report.differential.ranked else None,
                    "calibration_state": report.differential.ranked[0].calibration_state if report.differential.ranked else None,
                    "posterior_mass_top3": round(report.differential.posterior_mass_top3, 3),
                }
            )
        with tab5:
            st.code(json.dumps(report.model_dump(), indent=2), language="json")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
