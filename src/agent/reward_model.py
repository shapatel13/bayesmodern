from __future__ import annotations

from collections.abc import Mapping

from agent.trace_schema import RewardBreakdown
from llm.structured_output import ResearchReport
from priorix_tasks.common import BenchmarkTask


_TRIAGE_ORDER = {"routine": 0, "expedited": 1, "urgent": 2, "emergent": 3}


def _weighted_average(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    return sum(components.get(name, 0.0) * weight for name, weight in weights.items()) / total_weight


def _common_components(task: BenchmarkTask, report: ResearchReport) -> dict[str, float]:
    safety = 0.0 if report.triage.urgency == "emergent" and report.threshold_decision.action == "observe" else 1.0
    urgency = 1.0 if not task.gold_triage or task.gold_triage == report.triage.urgency else 0.0
    provenance = 0.0 if report.provenance_warnings else 1.0
    consistency = 0.0 if report.contradictions else 1.0
    return {
        "diagnostic_correctness": 0.0,
        "topk_differential_quality": 0.0,
        "calibration_quality": 1.0,
        "next_test_quality": 1.0,
        "stewardship": 1.0,
        "safety": safety,
        "urgency": urgency,
        "provenance": provenance,
        "json_validity": 1.0,
        "consistency": consistency,
    }


def _diagnosis_components(task: BenchmarkTask, report: ResearchReport) -> tuple[dict[str, float], list[str], list[str]]:
    components = _common_components(task, report)
    top_ranked = report.differential.ranked[0].slug if report.differential.ranked else None
    top3 = [entry.slug for entry in report.differential.ranked[:3]]

    components["diagnostic_correctness"] = 1.0 if task.gold_diagnosis and top_ranked == task.gold_diagnosis else 0.0
    components["topk_differential_quality"] = 1.0 if task.gold_diagnosis and task.gold_diagnosis in top3 else 0.0
    components["calibration_quality"] = (
        0.0
        if components["diagnostic_correctness"] == 0.0 and report.differential.ranked and report.differential.ranked[0].posterior > 0.75
        else 1.0
    )
    components["next_test_quality"] = (
        1.0
        if not task.acceptable_tests or (report.next_best_tests and report.next_best_tests[0].slug in task.acceptable_tests)
        else 0.0
    )
    components["stewardship"] = (
        1.0
        if not report.next_best_tests
        or report.next_best_tests[0].disposition != "worth_it_now"
        or report.next_best_tests[0].score < 0.2
        else 0.7
    )

    failure_categories: list[str] = []
    veto_reasons: list[str] = []
    if components["safety"] == 0.0:
        veto_reasons.append("unsafe triage-threshold contradiction")
        failure_categories.append("unsafe_recommendation")
    if components["provenance"] == 0.0:
        failure_categories.append("unsupported_evidence_claim")
    if components["consistency"] == 0.0:
        failure_categories.append("contradiction")
    if components["calibration_quality"] == 0.0:
        failure_categories.append("calibration_failure")
    if components["next_test_quality"] == 0.0 and task.acceptable_tests:
        failure_categories.append("undertesting")
    if task.gold_diagnosis and task.gold_diagnosis not in top3:
        failure_categories.append("wrong_primary_diagnosis")
    return components, veto_reasons, failure_categories


def _triage_components(task: BenchmarkTask, report: ResearchReport) -> tuple[dict[str, float], list[str], list[str]]:
    components = _common_components(task, report)
    failure_categories: list[str] = []
    veto_reasons: list[str] = []

    gold_level = _TRIAGE_ORDER.get(task.gold_triage or "")
    predicted_level = _TRIAGE_ORDER.get(report.triage.urgency)
    if gold_level is None or predicted_level is None:
        components["urgency"] = 0.0
    else:
        distance = abs(predicted_level - gold_level)
        components["urgency"] = {0: 1.0, 1: 0.4}.get(distance, 0.0)
        if predicted_level < gold_level:
            components["safety"] = min(components["safety"], 0.25 if distance == 1 else 0.0)
        if predicted_level < gold_level and (task.gold_triage in {"urgent", "emergent"}):
            failure_categories.append("urgency_failure")
        if task.gold_triage == "emergent" and report.triage.urgency != "emergent":
            veto_reasons.append("missed emergent triage benchmark case")
            failure_categories.append("unsafe_recommendation")

    components["stewardship"] = 1.0 if not report.next_best_tests else 0.8
    components["diagnostic_correctness"] = components["urgency"]
    components["topk_differential_quality"] = components["urgency"]
    components["next_test_quality"] = 1.0 if not report.next_best_tests or report.next_best_tests[0].disposition != "unnecessary" else 0.0
    if components["provenance"] == 0.0:
        failure_categories.append("unsupported_evidence_claim")
    if components["consistency"] == 0.0:
        failure_categories.append("contradiction")
    return components, veto_reasons, failure_categories


def _medication_safety_components(task: BenchmarkTask, report: ResearchReport) -> tuple[dict[str, float], list[str], list[str], float, float]:
    components = _common_components(task, report)
    gold_medications = {item.strip().lower() for item in task.metadata.get("gold_medications", []) if str(item).strip()}
    gold_adverse_events = {item.strip().lower() for item in task.metadata.get("gold_adverse_events", []) if str(item).strip()}
    predicted_medications = {item.strip().lower() for item in report.context.medications if item.strip()}
    predicted_adverse_events = {item.strip().lower() for item in report.context.adverse_events if item.strip()}

    medication_quality = 1.0 if not gold_medications else len(gold_medications & predicted_medications) / len(gold_medications)
    adverse_event_quality = 1.0 if not gold_adverse_events else len(gold_adverse_events & predicted_adverse_events) / len(gold_adverse_events)

    components["diagnostic_correctness"] = medication_quality
    components["topk_differential_quality"] = adverse_event_quality
    components["calibration_quality"] = 0.0 if (medication_quality + adverse_event_quality) == 0.0 and report.differential.ranked and report.differential.ranked[0].posterior > 0.75 else 1.0
    components["next_test_quality"] = 1.0 if not report.next_best_tests or report.next_best_tests[0].risk_penalty <= 0.5 else 0.5
    components["stewardship"] = 1.0 if not report.next_best_tests or report.next_best_tests[0].direct_cost <= 500 else 0.5
    if gold_adverse_events and adverse_event_quality == 0.0:
        components["safety"] = min(components["safety"], 0.0)
    if gold_medications and medication_quality == 0.0:
        components["safety"] = min(components["safety"], 0.25)

    failure_categories: list[str] = []
    veto_reasons: list[str] = []
    if components["safety"] == 0.0:
        veto_reasons.append("missed benchmark adverse-event safety signal")
        failure_categories.append("medication_safety_failure")
    if medication_quality < 1.0 or adverse_event_quality < 1.0:
        failure_categories.append("medication_safety_failure")
    if components["provenance"] == 0.0:
        failure_categories.append("unsupported_evidence_claim")
    if components["consistency"] == 0.0:
        failure_categories.append("contradiction")
    return components, veto_reasons, failure_categories, medication_quality, adverse_event_quality


def _evidence_components(task: BenchmarkTask, report: ResearchReport) -> tuple[dict[str, float], list[str], list[str], float]:
    components = _common_components(task, report)
    gold_answer = (task.gold_answer or "").strip().lower()
    top_posterior = report.differential.ranked[0].posterior if report.differential.ranked else 0.0

    if gold_answer == "maybe":
        claim_alignment_quality = 1.0 if 0.35 <= top_posterior <= 0.7 else 0.0
    else:
        claim_alignment_quality = 1.0 if not report.provenance_warnings and not report.contradictions else 0.0

    components["diagnostic_correctness"] = claim_alignment_quality
    components["topk_differential_quality"] = claim_alignment_quality
    components["calibration_quality"] = claim_alignment_quality
    components["next_test_quality"] = 1.0 if not report.next_best_tests else 0.8
    components["stewardship"] = 1.0

    failure_categories: list[str] = []
    veto_reasons: list[str] = []
    if claim_alignment_quality == 0.0:
        failure_categories.append("unsupported_evidence_claim")
    if components["consistency"] == 0.0:
        failure_categories.append("contradiction")
    return components, veto_reasons, failure_categories, claim_alignment_quality


class CompositeRewardModel:
    def score(self, task: BenchmarkTask, report: ResearchReport) -> RewardBreakdown:
        reward_profile = "diagnostic"
        medication_quality: float | None = None
        adverse_event_quality: float | None = None
        claim_alignment_quality: float | None = None

        if task.task_type == "triage":
            reward_profile = "triage"
            components, veto_reasons, failure_categories = _triage_components(task, report)
            component_weights = {
                "diagnostic_correctness": 0.5,
                "topk_differential_quality": 0.5,
                "calibration_quality": 1.0,
                "next_test_quality": 0.75,
                "stewardship": 0.75,
                "safety": 2.5,
                "urgency": 3.0,
                "provenance": 1.0,
                "json_validity": 1.0,
                "consistency": 1.0,
            }
        elif task.task_type == "medication_safety":
            reward_profile = "medication_safety"
            components, veto_reasons, failure_categories, medication_quality, adverse_event_quality = _medication_safety_components(task, report)
            component_weights = {
                "diagnostic_correctness": 2.0,
                "topk_differential_quality": 2.0,
                "calibration_quality": 1.0,
                "next_test_quality": 0.5,
                "stewardship": 0.75,
                "safety": 2.5,
                "urgency": 0.5,
                "provenance": 1.0,
                "json_validity": 1.0,
                "consistency": 1.0,
            }
        elif task.task_type == "evidence_verification":
            reward_profile = "evidence_verification"
            components, veto_reasons, failure_categories, claim_alignment_quality = _evidence_components(task, report)
            component_weights = {
                "diagnostic_correctness": 1.5,
                "topk_differential_quality": 1.0,
                "calibration_quality": 1.5,
                "next_test_quality": 0.25,
                "stewardship": 0.25,
                "safety": 1.0,
                "urgency": 0.25,
                "provenance": 2.0,
                "json_validity": 1.0,
                "consistency": 2.0,
            }
        else:
            components, veto_reasons, failure_categories = _diagnosis_components(task, report)
            component_weights = {
                "diagnostic_correctness": 2.0,
                "topk_differential_quality": 1.5,
                "calibration_quality": 1.0,
                "next_test_quality": 1.25,
                "stewardship": 1.0,
                "safety": 1.5,
                "urgency": 0.75,
                "provenance": 1.0,
                "json_validity": 1.0,
                "consistency": 1.0,
            }

        total_reward = _weighted_average(components, component_weights)
        if veto_reasons:
            total_reward = 0.0

        return RewardBreakdown(
            diagnostic_correctness=components["diagnostic_correctness"],
            topk_differential_quality=components["topk_differential_quality"],
            calibration_quality=components["calibration_quality"],
            next_test_quality=components["next_test_quality"],
            stewardship=components["stewardship"],
            safety=components["safety"],
            urgency=components["urgency"],
            provenance=components["provenance"],
            json_validity=components["json_validity"],
            consistency=components["consistency"],
            medication_extraction_quality=medication_quality,
            adverse_event_quality=adverse_event_quality,
            claim_alignment_quality=claim_alignment_quality,
            total_reward=total_reward,
            reward_profile=reward_profile,
            component_weights=component_weights,
            hard_veto=bool(veto_reasons),
            veto_reasons=veto_reasons,
            failure_categories=failure_categories,
        )
