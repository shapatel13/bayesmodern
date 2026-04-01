from __future__ import annotations

from fastapi import APIRouter

from agent.orchestrator import PRIORIXOrchestrator
from apps.api.schemas.analysis import AnalysisRequest
from eval.benchmark_runner import build_markdown_report, default_demo_tasks, run_benchmark
from llm.structured_output import ResearchReport

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=ResearchReport)
def analyze_case(request: AnalysisRequest) -> ResearchReport:
    orchestrator = PRIORIXOrchestrator()
    return orchestrator.analyze_text_case(case_id=request.case_id, note_text=request.note_text)


@router.post("/benchmark/sample")
def run_sample_benchmark() -> dict[str, object]:
    traces = run_benchmark(default_demo_tasks())
    return {
        "cases": len(traces),
        "report": build_markdown_report(traces),
    }
