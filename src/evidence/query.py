from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from evidence.registry_loader import (
    AgeGroup,
    EvidenceStatus,
    LREntry,
    SOURCE_QUALITY_SCORE,
    Setting,
    SourceType,
)


EVIDENCE_STATUS_SCORE: dict[EvidenceStatus, int] = {
    EvidenceStatus.SOURCED: 4,
    EvidenceStatus.ESTIMATED: 3,
    EvidenceStatus.SEED_ONLY: 2,
    EvidenceStatus.DEPRECATED: 1,
}


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.lower().replace("/", " ").replace("-", " ").split())


def _condition_matches(entry: LREntry, condition: str | None) -> bool:
    if condition is None:
        return True
    normalized_condition = _normalize(condition)
    return normalized_condition in {_normalize(entry.condition), _normalize(entry.condition_group)}


def _test_matches(entry: LREntry, test_or_finding: str | None) -> bool:
    if test_or_finding is None:
        return True
    requested = _normalize(test_or_finding)
    candidate = _normalize(entry.test_or_finding)
    return requested == candidate or requested in candidate or candidate in requested


def _setting_matches(entry: LREntry, setting: Setting | str | None) -> bool:
    if setting is None:
        return True
    requested = setting.value if isinstance(setting, Setting) else setting
    return requested in [item.value for item in entry.setting]


def _age_group_matches(entry: LREntry, age_group: AgeGroup | str | None) -> bool:
    if age_group is None:
        return True
    requested = age_group.value if isinstance(age_group, AgeGroup) else age_group
    return entry.population.age_group.value == requested


def _source_quality_matches(entry: LREntry, minimum_source_quality: SourceType | str | None) -> bool:
    if minimum_source_quality is None:
        return True
    minimum = minimum_source_quality if isinstance(minimum_source_quality, SourceType) else SourceType(minimum_source_quality)
    return entry.provenance_score >= SOURCE_QUALITY_SCORE[minimum]


def sort_registry_entries(entries: Iterable[LREntry]) -> list[LREntry]:
    return sorted(
        entries,
        key=lambda entry: (
            EVIDENCE_STATUS_SCORE[entry.evidence_status],
            entry.provenance_score,
            entry.publication_year or 0,
            entry.status_updated_at,
        ),
        reverse=True,
    )


def query_registry(
    entries: Iterable[LREntry],
    *,
    condition: str | None = None,
    test_or_finding: str | None = None,
    setting: Setting | str | None = None,
    age_group: AgeGroup | str | None = None,
    minimum_source_quality: SourceType | str | None = None,
) -> list[LREntry]:
    filtered = [
        entry
        for entry in entries
        if _condition_matches(entry, condition)
        and _test_matches(entry, test_or_finding)
        and _setting_matches(entry, setting)
        and _age_group_matches(entry, age_group)
        and _source_quality_matches(entry, minimum_source_quality)
        and entry.evidence_status != EvidenceStatus.DEPRECATED
    ]
    return sort_registry_entries(filtered)


@dataclass(frozen=True)
class RegistryResolution:
    selected: LREntry | None
    candidates: list[LREntry]
    selection_reason: str | None
    message: str | None
    confidence_label: str


def _is_exact_condition(entry: LREntry, condition: str) -> bool:
    return _normalize(entry.condition) == _normalize(condition)


def _is_related_syndrome(entry: LREntry, condition_group: str | None) -> bool:
    return condition_group is not None and _normalize(entry.condition_group) == _normalize(condition_group)


def _is_exact_setting(entry: LREntry, setting: Setting | str | None) -> bool:
    if setting is None:
        return True
    requested = setting.value if isinstance(setting, Setting) else setting
    entry_settings = [item.value for item in entry.setting]
    return requested in entry_settings and len(entry_settings) == 1


def _is_broader_setting(entry: LREntry, setting: Setting | str | None) -> bool:
    if setting is None:
        return False
    requested = setting.value if isinstance(setting, Setting) else setting
    entry_settings = [item.value for item in entry.setting]
    return any(item in {"mixed", "unknown"} for item in entry_settings) or (
        requested in entry_settings and len(entry_settings) > 1
    )


def resolve_registry_entry(
    entries: Iterable[LREntry],
    *,
    condition: str,
    test_or_finding: str,
    setting: Setting | str | None = None,
    age_group: AgeGroup | str | None = None,
    condition_group: str | None = None,
    minimum_source_quality: SourceType | str | None = None,
) -> RegistryResolution:
    candidates = query_registry(
        entries,
        condition=None,
        test_or_finding=test_or_finding,
        setting=None,
        age_group=age_group,
        minimum_source_quality=minimum_source_quality,
    )

    exact_condition_sourced = [
        entry
        for entry in candidates
        if _is_exact_condition(entry, condition) and entry.evidence_status == EvidenceStatus.SOURCED
    ]
    exact_setting = [entry for entry in exact_condition_sourced if _is_exact_setting(entry, setting)]
    if exact_setting:
        ranked = sort_registry_entries(exact_setting)
        return RegistryResolution(
            selected=ranked[0],
            candidates=ranked,
            selection_reason="exact_condition_exact_setting_sourced",
            message=None,
            confidence_label=ranked[0].confidence_label,
        )

    broader_setting = [entry for entry in exact_condition_sourced if _is_broader_setting(entry, setting)]
    if broader_setting:
        ranked = sort_registry_entries(broader_setting)
        return RegistryResolution(
            selected=ranked[0],
            candidates=ranked,
            selection_reason="exact_condition_broader_setting_sourced",
            message=None,
            confidence_label=ranked[0].confidence_label,
        )

    related_syndrome = [
        entry
        for entry in candidates
        if _is_related_syndrome(entry, condition_group) and entry.evidence_status == EvidenceStatus.SOURCED
    ]
    if related_syndrome:
        ranked = sort_registry_entries(related_syndrome)
        return RegistryResolution(
            selected=ranked[0],
            candidates=ranked,
            selection_reason="related_syndrome_sourced",
            message=None,
            confidence_label=ranked[0].confidence_label,
        )

    estimated = [
        entry
        for entry in candidates
        if _is_exact_condition(entry, condition) and entry.evidence_status == EvidenceStatus.ESTIMATED
    ]
    if estimated:
        ranked = sort_registry_entries(estimated)
        return RegistryResolution(
            selected=ranked[0],
            candidates=ranked,
            selection_reason="estimated_fallback",
            message="Estimated evidence selected; label as lower confidence in the UI.",
            confidence_label=ranked[0].confidence_label,
        )

    seed_only = [
        entry
        for entry in candidates
        if _is_exact_condition(entry, condition) and entry.evidence_status == EvidenceStatus.SEED_ONLY
    ]
    if seed_only:
        ranked = sort_registry_entries(seed_only)
        return RegistryResolution(
            selected=ranked[0],
            candidates=ranked,
            selection_reason="seed_only_fallback",
            message="No sourced LR data available yet for this condition/test; using seed registry only.",
            confidence_label=ranked[0].confidence_label,
        )

    return RegistryResolution(
        selected=None,
        candidates=[],
        selection_reason=None,
        message="No registry entry found for the requested condition/test combination.",
        confidence_label="unavailable",
    )
