from __future__ import annotations

from functools import lru_cache

from evidence.provenance import SourceRecord
from evidence.registry_loader import load_default_registry_bundle


LITERATURE_REGISTRY: dict[str, SourceRecord] = {
    "rule:wells_pe": SourceRecord(
        source_id="rule:wells_pe",
        title="Wells Criteria for Pulmonary Embolism",
        kind="clinical_rule",
        citation="Starter registry entry summarizing the Wells PE rule family.",
        notes="Used as a clinician-facing heuristic provenance pointer for PE-oriented features.",
    ),
    "study:d_dimer_pe": SourceRecord(
        source_id="study:d_dimer_pe",
        title="D-dimer performance in suspected PE",
        kind="primary_literature",
        citation="Starter registry entry summarizing D-dimer discrimination in suspected PE cohorts.",
        notes="Approximate LR ranges are intentionally conservative and should be locally curated.",
    ),
    "study:cta_pe": SourceRecord(
        source_id="study:cta_pe",
        title="CT pulmonary angiography for PE confirmation",
        kind="primary_literature",
        citation="Starter registry entry summarizing cross-sectional imaging confirmation for PE.",
        notes="Used for high-yield but higher-cost/risk testing metadata.",
    ),
    "study:cxr_pneumonia": SourceRecord(
        source_id="study:cxr_pneumonia",
        title="Chest radiography in community-acquired pneumonia",
        kind="primary_literature",
        citation="Starter registry entry summarizing chest radiograph discrimination for pneumonia.",
        notes="Appropriate for bedside vs radiographic differentiation logic.",
    ),
    "study:bnp_hf": SourceRecord(
        source_id="study:bnp_hf",
        title="BNP for acute decompensated heart failure",
        kind="primary_literature",
        citation="Starter registry entry summarizing BNP performance in acute heart failure workups.",
        notes="Included to support low-radiation discriminatory testing.",
    ),
    "registry:acs_symptom_profile": SourceRecord(
        source_id="registry:acs_symptom_profile",
        title="Starter ACS symptom and biomarker profile",
        kind="registry_note",
        citation="Internal starter registry entry for ACS-oriented findings.",
        notes="Replace with curated local evidence tables for benchmark-grade evaluation.",
    ),
    "cost:starter_us_hospital": SourceRecord(
        source_id="cost:starter_us_hospital",
        title="Starter US hospital cost tier assumptions",
        kind="cost_model",
        citation="Internal starter cost model for offline stewardship simulations.",
        notes="Replace with institutional cost catalogs before serious benchmarking.",
    ),
    "risk:contrast_ckd": SourceRecord(
        source_id="risk:contrast_ckd",
        title="Contrast nephrotoxicity heuristic",
        kind="risk_heuristic",
        citation="Internal risk heuristic for contrast-mediated renal risk penalties.",
        notes="Used for conservative research-mode penalties in CKD contexts.",
    ),
}


@lru_cache(maxsize=1)
def _seed_registry_source_ids() -> frozenset[str]:
    return frozenset(entry.id for entry in load_default_registry_bundle())


def is_known_provenance_ref(reference: str) -> bool:
    if reference in LITERATURE_REGISTRY:
        return True
    if reference.startswith("registry:"):
        return reference.split(":", maxsplit=1)[1] in _seed_registry_source_ids()
    return False
