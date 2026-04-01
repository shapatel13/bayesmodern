from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from core.models import CandidateTest, ClinicalDecisionContext, HypothesisEvidence, LikelihoodRatioRange
from evidence.query import RegistryResolution, resolve_registry_entry
from evidence.registry_loader import AgeGroup, EvidenceStatus, LREntry, load_default_registry_bundle


@dataclass(frozen=True)
class RegistryBinding:
    condition: str
    condition_group: str | None
    test_or_finding: str


@dataclass(frozen=True)
class TestRegistryBinding(RegistryBinding):
    diagnosis_slug: str


HYPOTHESIS_REGISTRY_BINDINGS: dict[str, dict[str, RegistryBinding]] = {
    "acs": {
        "troponin_positive": RegistryBinding(
            condition="Acute coronary syndrome",
            condition_group="cardiovascular",
            test_or_finding="High-sensitivity troponin",
        )
    }
}


TEST_REGISTRY_BINDINGS: dict[str, list[TestRegistryBinding]] = {
    "d_dimer": [
        TestRegistryBinding(
            diagnosis_slug="pe",
            condition="Pulmonary embolism",
            condition_group="thromboembolic",
            test_or_finding="D-dimer",
        )
    ],
    "cta_pe": [
        TestRegistryBinding(
            diagnosis_slug="pe",
            condition="Pulmonary embolism",
            condition_group="thromboembolic",
            test_or_finding="CT pulmonary angiography",
        )
    ],
    "cxr": [
        TestRegistryBinding(
            diagnosis_slug="pneumonia",
            condition="Community-acquired pneumonia",
            condition_group="infectious_pulmonary",
            test_or_finding="Chest radiograph",
        )
    ],
    "bnp": [
        TestRegistryBinding(
            diagnosis_slug="heart_failure",
            condition="Acute decompensated heart failure / volume overload",
            condition_group="cardiopulmonary",
            test_or_finding="BNP or NT-proBNP",
        )
    ],
}


@lru_cache(maxsize=1)
def _cached_seed_registry() -> tuple[LREntry, ...]:
    return tuple(load_default_registry_bundle())


def _active_registry(entries: list[LREntry] | None = None) -> tuple[LREntry, ...]:
    if entries is None:
        return _cached_seed_registry()
    return tuple(entries)


def _infer_age_group(context: ClinicalDecisionContext | None) -> AgeGroup | None:
    if context is None or context.age_years is None:
        return None
    if context.age_years < 18:
        return AgeGroup.PEDIATRIC
    if context.age_years >= 65:
        return AgeGroup.GERIATRIC
    return AgeGroup.ADULT


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _note_from_resolution(resolution: RegistryResolution) -> str | None:
    selected = resolution.selected
    note_parts: list[str] = []
    if resolution.message:
        note_parts.append(resolution.message)
    if selected is not None and selected.notes_for_model:
        note_parts.append(selected.notes_for_model)
    joined = " ".join(_unique_strings(note_parts)).strip()
    return joined or None


def _registry_ref(entry: LREntry) -> str:
    return f"registry:{entry.id}"


def likelihood_ratio_from_registry_entry(entry: LREntry) -> LikelihoodRatioRange | None:
    lr_positive = entry.lr_positive
    lr_negative = entry.lr_negative

    if lr_positive is None and entry.sensitivity is not None and entry.specificity is not None:
        specificity_gap = 1.0 - entry.specificity
        if specificity_gap > 0:
            lr_positive = entry.sensitivity / specificity_gap
    if lr_negative is None and entry.sensitivity is not None and entry.specificity is not None:
        if entry.specificity > 0:
            lr_negative = (1.0 - entry.sensitivity) / entry.specificity

    if lr_positive is None or lr_negative is None:
        return None

    return LikelihoodRatioRange(
        positive_lr=lr_positive,
        negative_lr=lr_negative,
        positive_lr_low=entry.ci_95_lower_lr_positive,
        positive_lr_high=entry.ci_95_upper_lr_positive,
        negative_lr_low=entry.ci_95_lower_lr_negative,
        negative_lr_high=entry.ci_95_upper_lr_negative,
    )


def enrich_hypothesis_evidence(
    disease_slug: str,
    evidence_items: list[HypothesisEvidence],
    *,
    context: ClinicalDecisionContext | None = None,
    registry_entries: list[LREntry] | None = None,
) -> list[HypothesisEvidence]:
    registry = _active_registry(registry_entries)
    age_group = _infer_age_group(context)
    bindings = HYPOTHESIS_REGISTRY_BINDINGS.get(disease_slug, {})
    enriched: list[HypothesisEvidence] = []

    for evidence in evidence_items:
        updated = evidence.model_copy(deep=True)
        binding = bindings.get(evidence.finding_key)
        if binding is None:
            enriched.append(updated)
            continue

        resolution = resolve_registry_entry(
            registry,
            condition=binding.condition,
            condition_group=binding.condition_group,
            test_or_finding=binding.test_or_finding,
            age_group=age_group,
        )
        selected = resolution.selected
        if selected is not None:
            updated.provenance_refs = _unique_strings([*updated.provenance_refs, _registry_ref(selected)])
            if selected.evidence_status == EvidenceStatus.SOURCED:
                registry_lr = likelihood_ratio_from_registry_entry(selected)
                if registry_lr is not None:
                    updated.lr = registry_lr
                    updated.source_type = "evidence_registry"
        note = _note_from_resolution(resolution)
        if note:
            updated.rationale = f"{updated.rationale} {note}".strip()
        enriched.append(updated)

    return enriched


def enrich_candidate_test(
    candidate: CandidateTest,
    *,
    context: ClinicalDecisionContext | None = None,
    registry_entries: list[LREntry] | None = None,
) -> CandidateTest:
    registry = _active_registry(registry_entries)
    age_group = _infer_age_group(context)
    bindings = TEST_REGISTRY_BINDINGS.get(candidate.slug, [])
    updated = candidate.model_copy(deep=True)
    note_fragments: list[str] = []
    override_applied = False

    for binding in bindings:
        resolution = resolve_registry_entry(
            registry,
            condition=binding.condition,
            condition_group=binding.condition_group,
            test_or_finding=binding.test_or_finding,
            age_group=age_group,
        )
        selected = resolution.selected
        if selected is None:
            continue

        updated.provenance_refs = _unique_strings([*updated.provenance_refs, _registry_ref(selected)])
        note = _note_from_resolution(resolution)
        if note:
            note_fragments.append(note)

        if selected.evidence_status != EvidenceStatus.SOURCED:
            continue

        registry_lr = likelihood_ratio_from_registry_entry(selected)
        if registry_lr is None:
            continue

        updated.diagnosis_lrs[binding.diagnosis_slug] = registry_lr
        override_applied = True

        if selected.direct_cost_usd is not None:
            updated.direct_cost = selected.direct_cost_usd
        if selected.downstream_cascade_cost_usd is not None:
            updated.downstream_cost = selected.downstream_cascade_cost_usd

    if note_fragments:
        updated.evidence_note = " ".join(_unique_strings(note_fragments))
    if override_applied:
        updated.source_type = "evidence_registry"
    return updated
