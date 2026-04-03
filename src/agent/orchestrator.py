from __future__ import annotations

from core.differential import DifferentialEngine
from core.latent_states import LatentStateEngine
from core.mechanism_coupling import apply_mechanism_coupling
from core.next_best_test import NextBestTestEngine
from core.policy import ReasoningPolicy, get_reasoning_policy
from core.thresholds import DecisionCostModel, ThresholdDecision, calculate_thresholds, explain_threshold_position
from core.triage import assess_triage
from agent.prompt_registry import resolve_prompt_template
from evidence.cost_catalog import default_test_catalog
from evidence.disease_profiles import DISEASE_PROFILES, default_hypotheses
from llm.citation_checker import find_missing_citations
from llm.extraction import extract_context_from_text
from llm.model_router import route_models
from llm.open_world_reasoning import (
    augment_context_with_open_world_findings,
    generate_open_world_reasoning_plan,
    merge_open_world_candidate_tests,
    merge_open_world_hypotheses,
)
from llm.structured_output import ResearchReport
from llm.structured_output import DecisionQualityAssessment
from llm.structured_output import ReasoningRuntimeTrace
from llm.validators import validate_report
from utils.config import Settings, get_settings
from utils.logging import get_logger


logger = get_logger(__name__)


def _has_present_finding(context, key: str) -> bool:
    return any(finding.key == key and finding.present for finding in context.findings)


def _find_differential_entry(differential, slug: str):
    return next((entry for entry in differential.ranked if entry.slug == slug), None)


def _hsv_threshold_refinement(context, differential, threshold_decision: ThresholdDecision) -> ThresholdDecision:
    hsv_entry = _find_differential_entry(differential, "hsv_encephalitis")
    if hsv_entry is None:
        return threshold_decision

    timed_negative_pcr = _has_present_finding(context, "hsv_pcr_negative_timed")
    repeat_negative_pcr = _has_present_finding(context, "repeat_hsv_pcr_negative_dependent")
    acyclovir_exposure = "acyclovir" in {medication.strip().lower() for medication in context.medications}
    nephrotoxicity_signal = _has_present_finding(context, "acyclovir_associated_aki") or (
        acyclovir_exposure and (_has_present_finding(context, "creatinine_elevated") or context.renal_impairment)
    )

    if not (timed_negative_pcr and acyclovir_exposure and nephrotoxicity_signal):
        return threshold_decision

    if repeat_negative_pcr and hsv_entry.posterior <= 0.05:
        return ThresholdDecision(
            action="observe",
            clinician_language=(
                "MRI remains only moderate positive support for HSV, while two appropriately timed negative CSF HSV PCRs "
                "dominate the evidence. In this research model the residual HSV posterior has fallen below the modeled "
                "treatment threshold, so worsening acyclovir-associated renal risk now outweighs expected antiviral benefit "
                "unless another HSV-specific signal emerges."
            ),
            plain_language=(
                "The MRI still raises concern, but two well-timed negative spinal-fluid HSV PCR tests now outweigh it. "
                "Because kidney injury is worsening on acyclovir, continuing treatment no longer clearly looks worth it "
                "unless new HSV-specific evidence appears."
            ),
        )

    if repeat_negative_pcr and hsv_entry.posterior <= 0.1:
        return ThresholdDecision(
            action="test",
            clinician_language=(
                "HSV probability has fallen into a gray zone after two timed negative CSF HSV PCRs. Continued acyclovir is "
                "no longer clearly favored over nephrotoxicity, so any further treatment should depend on sample timing, "
                "quality, and whether stronger HSV-specific evidence remains."
            ),
            plain_language=(
                "The chance of HSV now looks low enough that continuing acyclovir is uncertain rather than clearly beneficial. "
                "Double-check test timing and whether another diagnosis fits better."
            ),
        )

    return threshold_decision


def _hsv_reasoning_notes(context, differential) -> tuple[list[str], list[str]]:
    hsv_entry = _find_differential_entry(differential, "hsv_encephalitis")
    if hsv_entry is None:
        return [], []

    note_parts: list[str] = []
    dependency_parts: list[str] = []
    if _has_present_finding(context, "temporal_lobe_mri_pattern"):
        note_parts.append("Temporal-lobe MRI pattern is treated as only moderate positive support for HSV, not decisive support.")
    if _has_present_finding(context, "no_csf_pleocytosis"):
        note_parts.append("Absence of pleocytosis is modeled as weakly negative to near-neutral evidence.")
    if _has_present_finding(context, "eeg_without_classic_temporal_features"):
        note_parts.append("Absence of classic EEG features is modeled as weak negative evidence only.")
    if _has_present_finding(context, "hsv_pcr_negative_timed"):
        note_parts.append("A timed negative CSF HSV PCR is dominant negative evidence in this model.")
    if _has_present_finding(context, "repeat_hsv_pcr_negative_dependent"):
        note_parts.append("The second negative HSV PCR is treated as partially dependent rather than fully independent to avoid overstating certainty.")
        dependency_parts.append(
            "Repeat negative HSV PCRs are treated as partially dependent rather than fully independent evidence by default; "
            "the second negative test adds reduced incremental weight unless timing and sample quality are exceptionally strong."
        )
    if _has_present_finding(context, "acyclovir_associated_aki"):
        note_parts.append("Rising creatinine during acyclovir raises treatment-harm concern even if HSV is not fully excluded.")
    return note_parts, dependency_parts


def _should_expand_open_world(differential) -> bool:
    if not differential.ranked:
        return True
    top_entry = differential.ranked[0]
    runner_up = differential.ranked[1].posterior if len(differential.ranked) > 1 else 0.0
    return not (
        top_entry.posterior >= 0.4
        and top_entry.symptom_coverage >= 0.5
        and len(top_entry.evidence_for) >= 2
        and (top_entry.posterior - runner_up) >= 0.08
    )


def _open_world_gate_reason(differential) -> str:
    if not differential.ranked:
        return "No ranked differential available yet, so open-world expansion is allowed."
    top_entry = differential.ranked[0]
    runner_up = differential.ranked[1].posterior if len(differential.ranked) > 1 else 0.0
    margin = top_entry.posterior - runner_up
    if (
        top_entry.posterior >= 0.4
        and top_entry.symptom_coverage >= 0.5
        and len(top_entry.evidence_for) >= 2
        and margin >= 0.08
    ):
        return (
            f"Curated Bayesian pass was already confident: top diagnosis `{top_entry.slug}` "
            f"posterior {top_entry.posterior:.3f}, coverage {top_entry.symptom_coverage:.2f}, "
            f"evidence_for {len(top_entry.evidence_for)}, margin {margin:.3f}."
        )
    return (
        f"Curated Bayesian pass was not yet decisive: top diagnosis `{top_entry.slug}` "
        f"posterior {top_entry.posterior:.3f}, coverage {top_entry.symptom_coverage:.2f}, "
        f"evidence_for {len(top_entry.evidence_for)}, margin {margin:.3f}."
    )


def _decision_quality(context, differential, mechanism_result) -> DecisionQualityAssessment:
    structured_signal_count = sum(1 for finding in context.findings if finding.present)
    top_entry = differential.ranked[0] if differential.ranked else None
    runner_up = differential.ranked[1] if len(differential.ranked) > 1 else None
    top_gap = (top_entry.posterior - runner_up.posterior) if top_entry and runner_up else (top_entry.posterior if top_entry else 0.0)
    reasons: list[str] = []
    low_signal_case = structured_signal_count < 2
    broad_differential = bool(top_entry and (top_entry.posterior < 0.38 or top_gap < 0.08 or top_entry.symptom_coverage < 0.25))
    mixed_mechanism_uncertainty = mechanism_result.mixed_physiology and (not mechanism_result.active_states or (top_entry and top_entry.posterior < 0.52))
    if low_signal_case:
        reasons.append("Very few structured findings were extracted from the case.")
    if top_entry and top_entry.posterior < 0.38:
        reasons.append("No disease hypothesis reached a strong posterior probability.")
    if top_entry and top_entry.symptom_coverage < 0.25:
        reasons.append("The top diagnosis is supported by limited structured evidence coverage.")
    if runner_up and top_gap < 0.08:
        reasons.append("The leading diagnoses remain tightly clustered.")
    if mixed_mechanism_uncertainty:
        reasons.append("Mechanism layer suggests mixed physiology that still needs clinician review.")
    if context.hemodynamic_instability and top_entry and top_entry.posterior < 0.5:
        reasons.append("High-acuity physiology is present without a decisive lead diagnosis.")
    return DecisionQualityAssessment(
        needs_clinician_review=bool(reasons),
        reasons=reasons,
        structured_signal_count=structured_signal_count,
        top_differential_gap=max(top_gap, 0.0),
        low_signal_case=low_signal_case,
        broad_differential=broad_differential,
        mixed_mechanism_uncertainty=mixed_mechanism_uncertainty,
    )


class PRIORIXOrchestrator:
    def __init__(self, settings: Settings | None = None, *, policy_version: str = "v1-deterministic") -> None:
        self.settings = settings or get_settings()
        self.policy = get_reasoning_policy(policy_version)
        self.differential_engine = DifferentialEngine()
        self.mechanism_engine = LatentStateEngine()
        self.next_test_engine = NextBestTestEngine()

    def analyze_text_case(
        self,
        case_id: str,
        note_text: str,
        *,
        policy_version: str | None = None,
        prompt_template: str | None = None,
    ) -> ResearchReport:
        resolved_prompt_template = prompt_template or resolve_prompt_template("active")
        context = extract_context_from_text(
            case_id=case_id,
            note_text=note_text,
            settings=self.settings,
            prompt_template=resolved_prompt_template,
        )
        return self.analyze_context(
            context,
            policy_version=policy_version,
            prompt_template=resolved_prompt_template,
        )

    def analyze_context(self, context, *, policy_version: str | None = None, prompt_template: str | None = None) -> ResearchReport:
        policy: ReasoningPolicy = get_reasoning_policy(policy_version) if policy_version else self.policy
        base_hypotheses = default_hypotheses(context=context)
        mechanism_result = self.mechanism_engine.infer(
            context,
            seed=self.settings.seed,
            enable_dag_refinement=self.settings.mechanism_dag_enabled,
        )
        coupled_hypotheses = apply_mechanism_coupling(base_hypotheses, mechanism_result)
        base_differential = self.differential_engine.rank(
            context,
            coupled_hypotheses,
            seed=self.settings.seed,
            policy=policy,
        )
        open_world_plan = None
        differential = base_differential
        hypotheses = coupled_hypotheses
        gate_reason = (
            "Open-world expansion disabled by configuration."
            if not self.settings.open_world_reasoning_enabled
            else _open_world_gate_reason(base_differential)
        )
        should_expand = self.settings.open_world_reasoning_enabled and (
            not self.settings.open_world_expand_uncertain_only or _should_expand_open_world(base_differential)
        )
        if should_expand:
            try:
                open_world_plan = generate_open_world_reasoning_plan(
                    context,
                    self.settings,
                    prompt_template=prompt_template,
                )
            except Exception as exc:
                logger.warning("Open-world reasoning generation failed; continuing with deterministic hypothesis pool: %s", exc)
            context = augment_context_with_open_world_findings(context, open_world_plan)
            mechanism_result = self.mechanism_engine.infer(
                context,
                seed=self.settings.seed,
                enable_dag_refinement=self.settings.mechanism_dag_enabled,
            )
            hypotheses = apply_mechanism_coupling(
                merge_open_world_hypotheses(
                default_hypotheses(context=context),
                open_world_plan,
                context=context,
                ),
                mechanism_result,
            )
            differential = self.differential_engine.rank(
                context,
                hypotheses,
                seed=self.settings.seed,
                policy=policy,
            )
        dangerous_mass = sum(
            entry.posterior * DISEASE_PROFILES[entry.slug].urgency_weight
            for entry in differential.ranked
            if DISEASE_PROFILES.get(entry.slug, None) and DISEASE_PROFILES[entry.slug].dangerous
        )
        triage = assess_triage(dangerous_mass, context.hemodynamic_instability, context.critical_values_present)
        urgency_multiplier = 1.0
        if triage.urgency in {"urgent", "emergent"}:
            urgency_multiplier = policy.thresholds.urgent_multiplier
        elif triage.urgency == "expedited":
            urgency_multiplier = policy.thresholds.expedited_multiplier
        thresholds = calculate_thresholds(
            DecisionCostModel(
                benefit_of_treatment=policy.thresholds.benefit_of_treatment,
                harm_of_treatment=policy.thresholds.harm_of_treatment,
                harm_of_missed_disease=policy.thresholds.harm_of_missed_disease,
                harm_of_test=policy.thresholds.harm_of_test,
                discharge_harm=policy.thresholds.discharge_harm,
                icu_overuse_harm=policy.thresholds.icu_overuse_harm,
                urgency_multiplier=urgency_multiplier,
            )
        )
        posterior_map = {entry.slug: entry.posterior for entry in differential.ranked}
        candidates = merge_open_world_candidate_tests(
            default_test_catalog(context=context),
            open_world_plan,
        )
        recommendations = self.next_test_engine.rank(
            posterior_map,
            candidates,
            context,
            mechanism_result=mechanism_result,
            thresholds=thresholds,
            policy=policy,
        )[:5]
        threshold_decision = explain_threshold_position(differential.ranked[0].posterior, thresholds)
        threshold_decision = _hsv_threshold_refinement(context, differential, threshold_decision)
        report = ResearchReport(
            context=context,
            differential=differential,
            mechanism_states=mechanism_result,
            next_best_tests=recommendations,
            triage=triage,
            threshold_decision=threshold_decision,
            reasoning_runtime=ReasoningRuntimeTrace(
                mode="hybrid_open_world" if open_world_plan and (open_world_plan.hypotheses or open_world_plan.suggested_tests) else "curated_only",
                open_world_considered=self.settings.open_world_reasoning_enabled,
                open_world_triggered=bool(open_world_plan and (open_world_plan.hypotheses or open_world_plan.suggested_tests)),
                gate_reason=gate_reason,
                base_top_diagnosis=base_differential.ranked[0].slug if base_differential.ranked else None,
                base_top_posterior=base_differential.ranked[0].posterior if base_differential.ranked else None,
                final_top_diagnosis=differential.ranked[0].slug if differential.ranked else None,
                final_top_posterior=differential.ranked[0].posterior if differential.ranked else None,
                open_world_hypothesis_count=len(open_world_plan.hypotheses) if open_world_plan else 0,
                open_world_test_count=len(open_world_plan.suggested_tests) if open_world_plan else 0,
                notes=(open_world_plan.notes if open_world_plan else []),
            ),
            decision_quality=_decision_quality(context, differential, mechanism_result),
            model_route=route_models(self.settings),
        )
        report.contradictions = validate_report(report)
        report.provenance_warnings = find_missing_citations(report)
        report.differential.model_note = f"{report.differential.model_note} Triage mass {dangerous_mass:.3f}; policy `{policy.version}`."
        if mechanism_result.ranked:
            top_states = ", ".join(mechanism_result.active_states[:2]) or mechanism_result.summary
            report.differential.model_note = (
                f"{report.differential.model_note} Mechanism layer summary: {top_states}."
            )
        hsv_notes, hsv_dependency_notes = _hsv_reasoning_notes(context, differential)
        if hsv_notes:
            report.reasoning_runtime.special_reasoning_notes.extend(hsv_notes)
            report.differential.model_note = f"{report.differential.model_note} {' '.join(hsv_notes)}"
        if hsv_dependency_notes:
            report.reasoning_runtime.test_dependency_notes.extend(hsv_dependency_notes)
        if open_world_plan and (open_world_plan.hypotheses or open_world_plan.suggested_tests):
            report.differential.model_note = (
                f"{report.differential.model_note} Open-world reasoning expanded the candidate space with "
                f"{len(open_world_plan.hypotheses)} hypothesis proposals and {len(open_world_plan.suggested_tests)} "
                "test proposals before deterministic Bayesian scoring."
            )
        return report
