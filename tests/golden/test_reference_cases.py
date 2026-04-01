from agent.orchestrator import PRIORIXOrchestrator


def test_golden_pe_case_structure() -> None:
    report = PRIORIXOrchestrator().analyze_text_case(
        "golden-pe",
        "Pleuritic chest pain, tachycardia, and hypoxemia after travel. No fever.",
    )
    assert report.differential.ranked[0].slug == "pe"
    assert report.next_best_tests[0].slug in {"d_dimer", "cta_pe"}
    assert report.triage.urgency in {"urgent", "emergent"}


def test_golden_hf_case_structure() -> None:
    report = PRIORIXOrchestrator().analyze_text_case(
        "golden-hf",
        "Progressive dyspnea with orthopnea, diffuse crackles, and leg edema.",
    )
    top3 = {entry.slug for entry in report.differential.ranked[:3]}
    assert "heart_failure" in top3
    assert report.next_best_tests[0].slug in {"bnp", "cxr"}

