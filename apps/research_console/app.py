from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.lightning_adapter import detect_lightning_runtime
from agent.offline_rollout import run_dataset_offline_experiment
from agent.orchestrator import PRIORIXOrchestrator
from datasets.catalog import get_dataset_spec, list_dataset_specs
from eval.experiment_registry import compare_experiment_summaries, list_experiment_summaries
from eval.presets import get_research_preset, list_research_presets
from security.secrets import validate_live_llm_config
from ui.charts.probabilities import calibration_figure, differential_figure
from ui.components.report_cards import headline_cards
from ui.viewmodels.cockpit import differential_rows, next_test_rows, provenance_rows
from ui.viewmodels.research_lab import (
    benchmark_summary_cards,
    comparison_rows,
    dataset_catalog_rows,
    experiment_rows,
    preset_rows,
)
from utils.config import get_settings


EXPERIMENTS_ROOT = Path("artifacts/evals/experiments")


def main() -> None:
    settings = get_settings()
    secret_status = validate_live_llm_config(settings)
    lightning_runtime = detect_lightning_runtime(settings)
    orchestrator = PRIORIXOrchestrator(settings)
    dataset_specs = list_dataset_specs()
    research_presets = list_research_presets()
    experiment_summaries = list_experiment_summaries(EXPERIMENTS_ROOT)

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
                    f"- Microsoft Agent Lightning: `{lightning_runtime.mode}`",
                    f"- Trace redaction: `{settings.redact_traces}`",
                    f"- Namespace: `{settings.experiment_namespace}`",
                ]
            )
        )
        if lightning_runtime.mode == "export_only":
            st.info(lightning_runtime.reason)
        else:
            st.success(lightning_runtime.reason)
        if st.button("Load PE Demo"):
            st.session_state["case_text"] = sample_case
        st.caption("Every output remains inspectable, versionable, and explicitly research-only.")

    st.session_state.setdefault("case_text", sample_case)
    case_text = st.text_area("Case Intake", value=st.session_state["case_text"], height=180)
    run = st.button("Analyze Case", type="primary", use_container_width=True)

    if run:
        try:
            report = orchestrator.analyze_text_case("console-case", case_text)
            st.session_state["report_json"] = report.model_dump()
            st.session_state.pop("analysis_error", None)
        except Exception as exc:
            st.session_state["analysis_error"] = str(exc)

    if "analysis_error" in st.session_state:
        st.error(f"Case analysis failed but your prior workbench state was preserved: {st.session_state['analysis_error']}")

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
            if report.context.medications or report.context.adverse_events:
                st.json(
                    {
                        "medications": report.context.medications,
                        "adverse_events": report.context.adverse_events,
                    }
                )
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

    st.divider()
    st.subheader("Research Lab")
    research_metrics = st.columns(5)
    research_metrics[0].metric("Datasets", len(dataset_specs))
    research_metrics[1].metric("Preset Tracks", len(research_presets))
    research_metrics[2].metric("Experiments", len(experiment_summaries))
    research_metrics[3].metric("Lightning Mode", lightning_runtime.mode.replace("_", " ").title())
    research_metrics[4].metric("Native Training Ready", "Yes" if lightning_runtime.native_training_ready else "No")

    st.caption(
        "Offline benchmark runs export Microsoft Agent Lightning-compatible traces and transitions. "
        "No live patient traffic is used for self-improvement."
    )

    manual_col, preset_col, catalog_col = st.columns([1, 1, 1.2])
    with manual_col:
        st.markdown("#### Manual Rollout")
        selected_dataset = st.selectbox("Benchmark Dataset", [spec.key for spec in dataset_specs], key="lab_dataset")
        selected_spec = get_dataset_spec(selected_dataset)
        subset_value = st.text_input(
            "Subset / Config",
            value=selected_spec.default_subset or "",
            help="Leave blank to use the dataset's default configuration.",
        )
        train_limit = st.slider("Train Cases", min_value=1, max_value=32, value=8)
        validation_limit = st.slider("Validation Cases", min_value=0, max_value=16, value=4)
        prompt_version = st.text_input("Prompt Version", value="v1-offline")
        policy_version = st.text_input("Policy Version", value="v1-deterministic")
        if st.button("Run Offline Dataset Rollout", type="primary", use_container_width=True):
            with st.spinner("Running offline benchmark and exporting Lightning bundle..."):
                try:
                    experiment_summary, traces, report = run_dataset_offline_experiment(
                        selected_dataset,
                        EXPERIMENTS_ROOT,
                        subset=subset_value or None,
                        train_limit=train_limit,
                        validation_limit=validation_limit,
                        settings=settings,
                        prompt_version=prompt_version,
                        policy_version=policy_version,
                    )
                    st.session_state["lab_experiment_summary"] = experiment_summary.model_dump()
                    st.session_state["lab_experiment_report"] = report
                    st.session_state["lab_trace_count"] = len(traces)
                    experiment_summaries = list_experiment_summaries(EXPERIMENTS_ROOT)
                    st.session_state.pop("lab_error", None)
                    st.success(f"Stored experiment `{experiment_summary.experiment_id}` in {experiment_summary.artifact_dir}")
                except Exception as exc:
                    st.session_state["lab_error"] = str(exc)

        if "lab_error" in st.session_state:
            st.error(st.session_state["lab_error"])

    with preset_col:
        st.markdown("#### Specialty Tracks")
        selected_preset_key = st.selectbox("Research Preset", [preset.key for preset in research_presets], key="lab_preset")
        selected_preset = get_research_preset(selected_preset_key)
        st.markdown(f"**{selected_preset.label}**")
        st.caption(selected_preset.description)
        st.markdown(
            "\n".join(
                [
                    f"- Dataset: `{selected_preset.dataset_key}`",
                    f"- Mode: `{selected_preset.clinical_mode}`",
                    f"- Task family: `{selected_preset.task_family}`",
                    f"- Train/val: `{selected_preset.train_limit}` / `{selected_preset.validation_limit}`",
                    f"- Credentials required: `{selected_preset.requires_credentials}`",
                ]
            )
        )
        if selected_preset.notes:
            st.info(selected_preset.notes)
        if st.button("Run Specialty Track", use_container_width=True):
            with st.spinner(f"Running {selected_preset.label}..."):
                try:
                    experiment_summary, traces, report = run_dataset_offline_experiment(
                        selected_preset.dataset_key,
                        EXPERIMENTS_ROOT,
                        subset=selected_preset.subset,
                        train_limit=selected_preset.train_limit,
                        validation_limit=selected_preset.validation_limit,
                        settings=settings,
                        prompt_version=selected_preset.prompt_version,
                        policy_version=selected_preset.policy_version,
                    )
                    st.session_state["lab_experiment_summary"] = experiment_summary.model_dump()
                    st.session_state["lab_experiment_report"] = report
                    st.session_state["lab_trace_count"] = len(traces)
                    experiment_summaries = list_experiment_summaries(EXPERIMENTS_ROOT)
                    st.session_state.pop("lab_error", None)
                    st.success(f"Stored experiment `{experiment_summary.experiment_id}` in {experiment_summary.artifact_dir}")
                except Exception as exc:
                    st.session_state["lab_error"] = str(exc)

        st.dataframe(pd.DataFrame(preset_rows(research_presets)), use_container_width=True, hide_index=True)

    with catalog_col:
        st.markdown("#### Dataset Catalog")
        st.dataframe(pd.DataFrame(dataset_catalog_rows(dataset_specs)), use_container_width=True, hide_index=True)

    if "lab_experiment_summary" in st.session_state:
        from eval.experiment_registry import ExperimentSummary

        lab_summary = ExperimentSummary.model_validate(st.session_state["lab_experiment_summary"])
        st.markdown("#### Latest Offline Rollout")
        summary_columns = st.columns(len(benchmark_summary_cards(lab_summary.benchmark_summary)))
        for column, (label, value) in zip(summary_columns, benchmark_summary_cards(lab_summary.benchmark_summary), strict=True):
            column.metric(label, value)
        st.caption(f"Artifacts: `{lab_summary.artifact_dir}`")
        st.markdown(st.session_state.get("lab_experiment_report", ""))
        manifest_path = Path(lab_summary.artifact_dir) / "lightning_bundle_manifest.json"
        if manifest_path.exists():
            st.json(json.loads(manifest_path.read_text(encoding="utf-8")))

    st.markdown("#### Experiment Registry")
    if experiment_summaries:
        st.dataframe(pd.DataFrame(experiment_rows(experiment_summaries)), use_container_width=True, hide_index=True)
    else:
        st.info("No offline experiments have been recorded yet.")

    st.markdown("#### Experiment Compare")
    if len(experiment_summaries) >= 2:
        experiment_ids = [summary.experiment_id for summary in experiment_summaries]
        compare_left, compare_right = st.columns(2)
        baseline_id = compare_left.selectbox("Baseline Experiment", experiment_ids, index=min(1, len(experiment_ids) - 1))
        candidate_id = compare_right.selectbox("Candidate Experiment", experiment_ids, index=0)
        if baseline_id == candidate_id:
            st.warning("Choose two different experiment IDs to compare.")
        else:
            baseline = next(summary for summary in experiment_summaries if summary.experiment_id == baseline_id)
            candidate = next(summary for summary in experiment_summaries if summary.experiment_id == candidate_id)
            comparison = compare_experiment_summaries(baseline, candidate)
            st.dataframe(pd.DataFrame(comparison_rows(comparison)), use_container_width=True, hide_index=True)
            promoted = ", ".join(comparison.promoted_dimensions) or "None"
            regressed = ", ".join(comparison.regressed_dimensions) or "None"
            summary_col_1, summary_col_2 = st.columns(2)
            summary_col_1.info(f"Promoted: {promoted}")
            summary_col_2.warning(f"Regressed: {regressed}")
            if comparison.promotion_gate:
                verdict = comparison.promotion_gate.verdict
                message = (
                    f"Promotion gate: {verdict.upper()}. "
                    f"Rationale: {', '.join(comparison.promotion_gate.rationale) or 'No meaningful improvements yet.'}"
                )
                if verdict == "promote":
                    st.success(message)
                elif verdict == "hold":
                    st.info(message)
                else:
                    st.error(
                        f"{message} Blockers: {', '.join(comparison.promotion_gate.blockers) or 'unspecified regression'}"
                    )
    else:
        st.caption("Two or more experiments are needed before side-by-side comparison becomes available.")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
