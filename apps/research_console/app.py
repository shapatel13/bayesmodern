from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from utils.bootstrap import prefer_local_package

prefer_local_package("datasets", SRC_PATH / "datasets")

import pandas as pd
import streamlit as st

from agent.lightning_adapter import detect_lightning_runtime
from agent.lightning_train import train_curriculum_prompt
from agent.offline_rollout import run_curriculum_offline_experiment, run_dataset_offline_experiment
from agent.orchestrator import PRIORIXOrchestrator
from agent.prompt_registry import get_active_prompt_record, list_prompt_records
from agent.policy_optimizer import optimize_curriculum_policy, optimize_dataset_policy
from core.policy import list_reasoning_policies
from datasets.catalog import get_dataset_spec, list_dataset_specs
from datasets.curricula import get_lightning_curriculum, list_lightning_curricula
from datasets.reviewed_case_store import append_reviewed_case, build_reviewed_case_row, resolve_reviewed_cases_destination
from eval.experiment_registry import compare_experiment_summaries, list_experiment_summaries
from eval.presets import get_research_preset, list_research_presets
from security.secrets import validate_live_llm_config
from ui.charts.probabilities import calibration_figure, differential_figure
from ui.components.report_cards import headline_cards
from ui.viewmodels.cockpit import differential_rows, next_test_rows, provenance_rows
from ui.viewmodels.research_lab import (
    benchmark_summary_cards,
    comparison_rows,
    curriculum_rows,
    dataset_catalog_rows,
    experiment_rows,
    preset_rows,
)
from utils.config import get_settings


EXPERIMENTS_ROOT = Path("artifacts/evals/experiments")
GUIDED_DEMO_CASES: dict[str, str] = {
    "Pulmonary Embolism": (
        "52-year-old with pleuritic chest pain, tachycardia, and hypoxemia after recent immobility. "
        "No fever. Concern for pulmonary embolism versus pneumonia."
    ),
    "Heart Failure": "68-year-old with orthopnea, crackles, bilateral leg edema, and worsening dyspnea over several days.",
    "Medication Safety": "Patient on warfarin with melena and symptomatic anemia after recent dose escalation.",
    "ED Triage": "Crushing chest pain with diaphoresis, hypotension, and concern for immediate resuscitation.",
    "Rare Disease": "Young adult with renal dysfunction, neuropathic pain, and angiokeratoma-like skin lesions.",
}


def _store_rollout_result(experiment_summary, traces, report) -> None:
    st.session_state["lab_experiment_summary"] = experiment_summary.model_dump()
    st.session_state["lab_experiment_report"] = report
    st.session_state["lab_trace_count"] = len(traces)
    st.session_state.pop("lab_error", None)


def _run_preset_rollout(preset_key: str, settings) -> None:
    preset = get_research_preset(preset_key)
    experiment_summary, traces, report = run_dataset_offline_experiment(
        preset.dataset_key,
        EXPERIMENTS_ROOT,
        subset=preset.subset,
        train_limit=preset.train_limit,
        validation_limit=preset.validation_limit,
        settings=settings,
        prompt_version=preset.prompt_version,
        policy_version=preset.policy_version,
    )
    _store_rollout_result(experiment_summary, traces, report)


def main() -> None:
    settings = get_settings()
    secret_status = validate_live_llm_config(settings)
    lightning_runtime = detect_lightning_runtime(settings)
    policy_options = list_reasoning_policies()
    orchestrator = PRIORIXOrchestrator(settings)
    dataset_specs = list_dataset_specs()
    lightning_curricula = list_lightning_curricula()
    active_prompt = get_active_prompt_record()
    prompt_records = list_prompt_records()
    reviewed_cases_destination = resolve_reviewed_cases_destination(settings)
    research_presets = list_research_presets()
    experiment_summaries = list_experiment_summaries(EXPERIMENTS_ROOT)
    api_base_url = os.environ.get("PRIORI_API_BASE_URL")
    startup_command = "python run_priori_x.py"

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
    with st.expander("Quick Handbook", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "**Start here if the app feels overwhelming.**",
                    "",
                    "1. Paste a case into `Case Intake`.",
                    "2. Leave `Case Policy` on `v1-deterministic`.",
                    "3. Click `Analyze Case`.",
                    "4. Read results in this order: `Diagnostic Cockpit` -> `Next Best Test` -> `Safety & Provenance` -> `Calibration Lab`.",
                    "5. Ignore most of `Research Lab` until you want benchmarking or prompt improvement.",
                    "",
                    "**Use each section like this:**",
                    "",
                    "- `Diagnostic Cockpit`: understand the ranked differential and uncertainty.",
                    "- `Next Best Test`: choose the most discriminative, stewardship-aware next step.",
                    "- `Safety & Provenance`: check urgency, contradictions, provenance, medications, and ADE signals.",
                    "- `Calibration Lab`: compare confidence versus fragility, mainly for evaluation.",
                    "- `Audit Trail`: inspect the full structured output and save examples for later review.",
                    "",
                    "**When to optimize prompts:**",
                    "",
                    "- after running benchmark datasets",
                    "- after collecting reviewed misses",
                    "- not after a single raw case",
                    "",
                    "**Best default improvement path:**",
                    "",
                    "- use the `continuous_improvement_feedback_lab` curriculum",
                    "- add de-identified reviewed cases through `PRIORI_REVIEWED_CASES_PATH`",
                    "- run offline prompt improvement only after review",
                    "",
                    "**Full handbook:** `docs/HANDBOOK.md`",
                ]
            )
        )

    sample_case = GUIDED_DEMO_CASES["Pulmonary Embolism"]
    with st.sidebar:
        st.header("Research Mode")
        sidebar_lines = [
            f"- Safety mode: `{settings.safety_mode}`",
            f"- Live external LLMs: `{settings.allow_live_llm}`",
            f"- Provider: `{settings.default_model_provider}`",
            f"- OpenAI parser: `{settings.openai_parser_model}`",
            f"- OpenAI reasoner: `{settings.openai_reasoning_model}`",
            f"- Provider ready: `{secret_status.provider_ready}`",
            f"- Microsoft Agent Lightning: `{lightning_runtime.mode}`",
            f"- Active prompt: `{active_prompt.version}`",
            f"- Trace redaction: `{settings.redact_traces}`",
            f"- Namespace: `{settings.experiment_namespace}`",
        ]
        if api_base_url:
            sidebar_lines.append(f"- API: `{api_base_url}`")
        st.markdown("\n".join(sidebar_lines))
        if lightning_runtime.mode == "export_only":
            st.info(lightning_runtime.reason)
        else:
            st.success(lightning_runtime.reason)
        selected_case_policy = st.selectbox(
            "Case Policy",
            [policy.version for policy in policy_options],
            index=0,
            help="Switch the deterministic reasoning policy used for live case analysis.",
        )
        selected_demo_case = st.selectbox("Starter Case Template", list(GUIDED_DEMO_CASES.keys()), key="guided_demo_case")
        if st.button("Load Starter Case"):
            st.session_state["case_text"] = GUIDED_DEMO_CASES[selected_demo_case]
        st.caption("Every output remains inspectable, versionable, and explicitly research-only.")

    st.session_state.setdefault("case_text", sample_case)
    case_text = st.text_area("Case Intake", value=st.session_state["case_text"], height=180)
    run = st.button("Analyze Case", type="primary", use_container_width=True)

    if run:
        try:
            report = orchestrator.analyze_text_case("console-case", case_text, policy_version=selected_case_policy)
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

        with st.expander("Save This Case To Reviewed Cases", expanded=False):
            st.caption(
                "Use this to turn a good or bad run into a reviewed case for offline Lightning improvement. "
                f"Saved rows go to `{reviewed_cases_destination}` and will be picked up by the reviewed-cases curriculum."
            )
            suggested_top = report.differential.ranked[0].slug if report.differential.ranked else ""
            suggested_tests = [recommendation.slug for recommendation in report.next_best_tests[:5]]
            default_tests = suggested_tests[:2]
            with st.form("reviewed_case_capture_form", clear_on_submit=False):
                reviewed_diagnosis = st.text_input(
                    "Reviewed Diagnosis",
                    value=suggested_top,
                    help="Enter your reviewed gold diagnosis. Edit the suggestion if the model was wrong.",
                )
                reviewed_tests = st.multiselect(
                    "Acceptable Tests",
                    options=suggested_tests,
                    default=default_tests,
                    help="Choose tests that would count as acceptable next steps for this case.",
                )
                urgency_options = ["routine", "expedited", "urgent", "emergent"]
                urgency_index = urgency_options.index(report.triage.urgency) if report.triage.urgency in urgency_options else 2
                reviewed_triage = st.selectbox(
                    "Reviewed Triage",
                    urgency_options,
                    index=urgency_index,
                )
                review_status = st.selectbox(
                    "Review Status",
                    ["approved", "draft"],
                    index=0,
                    help="Use `approved` when you are comfortable using this case in offline improvement runs.",
                )
                reviewer_id = st.text_input("Reviewer ID", value="local_reviewer")
                review_notes = st.text_area(
                    "Review Notes",
                    value=(
                        f"Captured from PRIORI-X. Model top diagnosis was `{suggested_top or 'unknown'}`; "
                        f"current top tests were {', '.join(suggested_tests[:3]) or 'none'}."
                    ),
                    height=120,
                )
                save_reviewed_case = st.form_submit_button("Save To Reviewed Cases", use_container_width=True)

            if save_reviewed_case:
                try:
                    row = build_reviewed_case_row(
                        note_text=case_text,
                        gold_diagnosis=reviewed_diagnosis,
                        acceptable_tests=reviewed_tests,
                        gold_triage=reviewed_triage,
                        review_status=review_status,
                        reviewer_id=reviewer_id,
                        review_notes=review_notes,
                        suggested_top_diagnosis=suggested_top or None,
                        suggested_next_tests=suggested_tests,
                        policy_version=selected_case_policy,
                        prompt_version=active_prompt.version,
                    )
                    destination = append_reviewed_case(row, settings)
                    st.success(
                        f"Saved reviewed case `{row['id']}` to {destination}. "
                        "It can now feed the reviewed-cases and continuous-improvement curricula."
                    )
                except Exception as exc:
                    st.error(f"Failed to save reviewed case: {exc}")

    st.divider()
    st.subheader("Research Lab")
    research_metrics = st.columns(6)
    research_metrics[0].metric("Datasets", len(dataset_specs))
    research_metrics[1].metric("Preset Tracks", len(research_presets))
    research_metrics[2].metric("Lightning Curricula", len(lightning_curricula))
    research_metrics[3].metric("Experiments", len(experiment_summaries))
    research_metrics[4].metric("Tracked Prompts", len(prompt_records))
    research_metrics[5].metric("Native Training Ready", "Yes" if lightning_runtime.native_training_ready else "No")

    st.caption(
        "Offline benchmark runs export Microsoft Agent Lightning-compatible traces and transitions. "
        "No live patient traffic is used for self-improvement."
    )
    st.info(
        "Public Hugging Face tracks like MedMCQA, MedQA, PubMedQA, and FindZebra can now be mixed into a single "
        "Lightning feedback curriculum for offline reward-bearing policy optimization."
    )
    st.markdown("#### Prompt Improvement")
    improvement_left, improvement_right = st.columns([1, 1.4])
    with improvement_left:
        st.markdown(f"**Active Prompt:** `{active_prompt.version}`")
        st.caption(active_prompt.description)
        if active_prompt.notes:
            st.info(active_prompt.notes[0])
        st.caption(
            "Prompt updates are offline only. Raw app traffic is never trained on directly; reviewed cases must be "
            "added to the reviewed-case dataset first."
        )
    with improvement_right:
        improvement_curriculum = st.selectbox(
            "Improvement Curriculum",
            [curriculum.key for curriculum in lightning_curricula],
            index=(
                [curriculum.key for curriculum in lightning_curricula].index("continuous_improvement_feedback_lab")
                if any(curriculum.key == "continuous_improvement_feedback_lab" for curriculum in lightning_curricula)
                else 0
            ),
            key="improvement_curriculum",
        )
        improvement_policy = st.selectbox(
            "Improvement Policy",
            [policy.version for policy in policy_options],
            index=0,
            key="improvement_policy",
        )
        improvement_train_cap = st.slider("Improvement Train Cap / Component", min_value=1, max_value=32, value=8)
        improvement_validation_cap = st.slider("Improvement Validation Cap / Component", min_value=0, max_value=16, value=4)
        if st.button("Run Prompt Improvement", use_container_width=True):
            with st.spinner("Running offline prompt improvement loop..."):
                try:
                    training_summary = train_curriculum_prompt(
                        improvement_curriculum,
                        EXPERIMENTS_ROOT,
                        settings=settings,
                        prompt_version="active",
                        policy_version=improvement_policy,
                        train_cap_per_component=improvement_train_cap,
                        validation_cap_per_component=improvement_validation_cap,
                        n_runners=1,
                    )
                    st.session_state["prompt_training_summary"] = training_summary.model_dump()
                    st.session_state.pop("prompt_training_error", None)
                except Exception as exc:
                    st.session_state["prompt_training_error"] = str(exc)
    if "prompt_training_error" in st.session_state:
        st.error(st.session_state["prompt_training_error"])
    if "prompt_training_summary" in st.session_state:
        training_summary = st.session_state["prompt_training_summary"]
        verdict = training_summary.get("status", "blocked")
        if verdict == "trained_and_promoted":
            st.success("A candidate prompt cleared the held-out promotion gate and is now active.")
        elif verdict == "trained_and_held":
            st.warning("Prompt training completed, but the candidate did not clear the promotion gate.")
        elif verdict == "blocked":
            st.info("Prompt training is configured but currently blocked on platform/runtime setup.")
        st.json(training_summary)

    demo_presets = [
        preset
        for preset in research_presets
        if preset.key in {"clinical_reasoning_demo_lab", "ed_triage_demo_lab", "medication_safety_demo_lab"}
    ]
    st.markdown("#### Fast Start")
    morning_col_1, morning_col_2, morning_col_3 = st.columns(3)
    morning_col_1.info(
        "Use `Pulmonary Embolism`, `Heart Failure`, or `Medication Safety` in the starter-case selector for a fast first run."
    )
    morning_col_2.info(
        "Run a built-in smoke track if you want to verify the stack end to end before switching to Hugging Face or local clinical datasets."
    )
    morning_col_3.info(
        "If OpenAI is unavailable, PRIORI-X still falls back to the deterministic engine and preserves your prior console state."
    )

    if demo_presets:
        st.markdown("#### Built-In Smoke Tracks")
        demo_columns = st.columns(len(demo_presets))
        for column, preset in zip(demo_columns, demo_presets, strict=True):
            if column.button(preset.label, use_container_width=True):
                with st.spinner(f"Running {preset.label}..."):
                    try:
                        _run_preset_rollout(preset.key, settings)
                        experiment_summaries = list_experiment_summaries(EXPERIMENTS_ROOT)
                        st.success(f"Completed {preset.label}.")
                    except Exception as exc:
                        st.session_state["lab_error"] = str(exc)

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
        prompt_version = st.text_input("Prompt Version", value="active")
        policy_version = st.selectbox(
            "Policy Version",
            [policy.version for policy in policy_options],
            index=0,
            key="manual_rollout_policy_version",
        )
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
        preset_policy_version = st.selectbox(
            "Preset Policy Override",
            [policy.version for policy in policy_options],
            index=0,
            key="preset_policy_version",
            help="Use a different deterministic policy while keeping the preset's dataset and prompt defaults.",
        )
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
                        policy_version=preset_policy_version,
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

    st.markdown("#### Lightning Feedback Curricula")
    curriculum_col, curriculum_table_col = st.columns([1, 1.4])
    with curriculum_col:
        selected_curriculum_key = st.selectbox(
            "Lightning Curriculum",
            [curriculum.key for curriculum in lightning_curricula],
            key="lightning_curriculum",
        )
        selected_curriculum = get_lightning_curriculum(selected_curriculum_key)
        curriculum_train_cap = st.number_input(
            "Train Cap Per Dataset",
            min_value=1,
            max_value=64,
            value=8,
            step=1,
            help="Use a smaller cap for a faster public-HF curriculum pass.",
        )
        curriculum_validation_cap = st.number_input(
            "Validation Cap Per Dataset",
            min_value=0,
            max_value=32,
            value=4,
            step=1,
            help="Set to 0 to skip validation export if you only want a quick training bundle.",
        )
        st.markdown(f"**{selected_curriculum.label}**")
        st.caption(selected_curriculum.description)
        st.markdown(
            "\n".join(
                [
                    f"- Access: `{selected_curriculum.access_mode}`",
                    f"- Objective: {selected_curriculum.objective}",
                    f"- Datasets: `{', '.join(component.dataset_key for component in selected_curriculum.components)}`",
                    f"- Focus: `{', '.join(selected_curriculum.focus_areas)}`",
                ]
            )
        )
        if selected_curriculum.notes:
            st.info(selected_curriculum.notes)
        if st.button("Run Lightning Feedback Curriculum", use_container_width=True):
            with st.spinner(f"Running {selected_curriculum.label}..."):
                try:
                    experiment_summary, traces, report = run_curriculum_offline_experiment(
                        selected_curriculum.key,
                        EXPERIMENTS_ROOT,
                        settings=settings,
                        train_cap_per_component=int(curriculum_train_cap),
                        validation_cap_per_component=int(curriculum_validation_cap),
                    )
                    _store_rollout_result(experiment_summary, traces, report)
                    experiment_summaries = list_experiment_summaries(EXPERIMENTS_ROOT)
                    st.success(f"Stored curriculum experiment `{experiment_summary.experiment_id}`.")
                except Exception as exc:
                    st.session_state["lab_error"] = str(exc)
    with curriculum_table_col:
        st.dataframe(pd.DataFrame(curriculum_rows(lightning_curricula)), use_container_width=True, hide_index=True)

    st.markdown("#### Offline Policy Search")
    search_col_1, search_col_2 = st.columns([1, 1.2])
    with search_col_1:
        search_mode = st.radio("Optimization Target", ["Dataset", "Curriculum"], horizontal=True)
        baseline_policy_version = st.selectbox(
            "Baseline Policy",
            [policy.version for policy in policy_options],
            index=0,
            key="policy_search_baseline",
        )
        candidate_policy_versions = [
            policy.version
            for policy in policy_options
            if policy.version != baseline_policy_version
        ]
        if search_mode == "Dataset":
            if st.button("Optimize Selected Dataset Policy", use_container_width=True):
                with st.spinner("Running offline policy optimization on the selected dataset..."):
                    try:
                        optimization = optimize_dataset_policy(
                            selected_dataset,
                            EXPERIMENTS_ROOT,
                            subset=subset_value or None,
                            train_limit=train_limit,
                            validation_limit=validation_limit,
                            settings=settings,
                            prompt_version=prompt_version,
                            baseline_policy_version=baseline_policy_version,
                            candidate_policy_versions=candidate_policy_versions,
                        )
                        st.session_state["lab_policy_optimization"] = optimization.model_dump()
                        st.success(f"Selected policy `{optimization.selected_policy_version}`.")
                    except Exception as exc:
                        st.session_state["lab_error"] = str(exc)
        else:
            if st.button("Optimize Selected Curriculum Policy", use_container_width=True):
                with st.spinner("Running offline policy optimization on the selected curriculum..."):
                    try:
                        optimization = optimize_curriculum_policy(
                            selected_curriculum.key,
                            EXPERIMENTS_ROOT,
                            settings=settings,
                            baseline_policy_version=baseline_policy_version,
                            candidate_policy_versions=candidate_policy_versions,
                            train_cap_per_component=int(curriculum_train_cap),
                            validation_cap_per_component=int(curriculum_validation_cap),
                        )
                        st.session_state["lab_policy_optimization"] = optimization.model_dump()
                        st.success(f"Selected policy `{optimization.selected_policy_version}`.")
                    except Exception as exc:
                        st.session_state["lab_error"] = str(exc)
    with search_col_2:
        if "lab_policy_optimization" in st.session_state:
            optimization = st.session_state["lab_policy_optimization"]
            st.json(optimization)
        else:
            st.info("Run dataset or curriculum policy optimization to compare deterministic Bayesian policies offline.")

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

    st.markdown("#### System")
    system_col_1, system_col_2 = st.columns([1, 1])
    system_col_1.markdown("**Launcher**")
    system_col_1.code(startup_command, language="bash")
    system_col_1.code(f"{startup_command} --check", language="bash")
    if api_base_url:
        system_col_1.success(f"Integrated API available at `{api_base_url}`")
    else:
        system_col_1.info("Use the launcher command above to start the API and Streamlit workbench together.")

    system_col_2.markdown("**Runtime**")
    system_col_2.json(
        {
            "provider_ready": secret_status.provider_ready,
            "allow_live_llm": secret_status.allow_live_llm,
            "missing_required_secrets": secret_status.missing_required_secrets,
            "lightning_mode": lightning_runtime.mode,
            "mietic_path": settings.mietic_path,
            "n2c2_2018_track2_path": settings.n2c2_2018_track2_path,
            "medval_bench_path": settings.medval_bench_path,
        }
    )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
