from __future__ import annotations

from agent.trace_schema import RewardBreakdown
from datasets.task_builders.common import BenchmarkTask
from llm.structured_output import ResearchReport


class CompositeRewardModel:
    def score(self, task: BenchmarkTask, report: ResearchReport) -> RewardBreakdown:
        top_ranked = report.differential.ranked[0].slug if report.differential.ranked else None
        top3 = [entry.slug for entry in report.differential.ranked[:3]]
        diagnostic_correctness = 1.0 if task.gold_diagnosis and top_ranked == task.gold_diagnosis else 0.0
        topk_quality = 1.0 if task.gold_diagnosis and task.gold_diagnosis in top3 else 0.0
        calibration_quality = 0.0 if diagnostic_correctness == 0.0 and report.differential.ranked and report.differential.ranked[0].posterior > 0.75 else 1.0
        next_test_quality = 1.0 if not task.acceptable_tests or (report.next_best_tests and report.next_best_tests[0].slug in task.acceptable_tests) else 0.0
        stewardship = 1.0 if not report.next_best_tests or report.next_best_tests[0].disposition != "worth_it_now" or report.next_best_tests[0].score < 0.2 else 0.7
        safety = 0.0 if report.triage.urgency == "emergent" and report.threshold_decision.action == "observe" else 1.0
        urgency = 1.0 if not task.gold_triage or task.gold_triage == report.triage.urgency else 0.0
        provenance = 0.0 if report.provenance_warnings else 1.0
        json_validity = 1.0
        consistency = 0.0 if report.contradictions else 1.0

        veto_reasons: list[str] = []
        failure_categories: list[str] = []
        if safety == 0.0:
            veto_reasons.append("unsafe triage-threshold contradiction")
            failure_categories.append("unsafe_recommendation")
        if provenance == 0.0:
            failure_categories.append("unsupported_evidence_claim")
        if consistency == 0.0:
            failure_categories.append("contradiction")
        if calibration_quality == 0.0:
            failure_categories.append("calibration_failure")
        if next_test_quality == 0.0 and task.acceptable_tests:
            failure_categories.append("undertesting")
        if task.gold_diagnosis and task.gold_diagnosis not in top3:
            failure_categories.append("wrong_primary_diagnosis")

        total_reward = sum(
            [
                diagnostic_correctness,
                topk_quality,
                calibration_quality,
                next_test_quality,
                stewardship,
                safety,
                urgency,
                provenance,
                json_validity,
                consistency,
            ]
        ) / 10.0
        if veto_reasons:
            total_reward = 0.0

        return RewardBreakdown(
            diagnostic_correctness=diagnostic_correctness,
            topk_differential_quality=topk_quality,
            calibration_quality=calibration_quality,
            next_test_quality=next_test_quality,
            stewardship=stewardship,
            safety=safety,
            urgency=urgency,
            provenance=provenance,
            json_validity=json_validity,
            consistency=consistency,
            total_reward=total_reward,
            hard_veto=bool(veto_reasons),
            veto_reasons=veto_reasons,
            failure_categories=failure_categories,
        )

