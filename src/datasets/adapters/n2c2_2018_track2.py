from __future__ import annotations

from collections.abc import Iterable

from datasets.transforms.normalize import as_text
from priorix_tasks.common import BenchmarkTask


def _coalesce_text(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(part for part in (as_text(item) for item in value) if part).strip()
    return as_text(value)


def _flatten_passages(row: dict[str, object]) -> str:
    passages = row.get("passages")
    if not isinstance(passages, list):
        return ""
    chunks: list[str] = []
    for passage in passages:
        if isinstance(passage, dict):
            text_value = passage.get("text")
            if isinstance(text_value, list):
                chunks.extend(part for part in (_coalesce_text(item) for item in text_value) if part)
            else:
                text = _coalesce_text(text_value)
                if text:
                    chunks.append(text)
    return "\n".join(chunks).strip()


def _normalize_label(value: object) -> str:
    return as_text(value).lower().replace("_", " ").replace("-", " ")


def _extract_entity_text(entity: object) -> str:
    if isinstance(entity, dict):
        if "text" in entity:
            return _coalesce_text(entity.get("text"))
        if "mention" in entity:
            return _coalesce_text(entity.get("mention"))
    return _coalesce_text(entity)


def _extract_typed_entities(entities: object, *, needles: Iterable[str]) -> list[str]:
    if not isinstance(entities, list):
        return []
    normalized_needles = tuple(needle.lower() for needle in needles)
    extracted: list[str] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        entity_type = _normalize_label(entity.get("type") or entity.get("label") or entity.get("category"))
        if not any(needle in entity_type for needle in normalized_needles):
            continue
        text = _extract_entity_text(entity)
        if text:
            extracted.append(text)
    return sorted(set(extracted))


def normalize_n2c2_2018_track2_row(row: dict[str, object], split: str) -> BenchmarkTask:
    prompt = (
        _coalesce_text(row.get("note") or row.get("text") or row.get("sentence") or row.get("document_text"))
        or _flatten_passages(row)
    )

    medications = sorted(
        set(
            item
            for item in [
                *(_coalesce_text(item) for item in row.get("medications", []) if isinstance(row.get("medications"), list)),
                _coalesce_text(row.get("medication") or row.get("drug")),
                *_extract_typed_entities(row.get("entities"), needles=("drug", "medication")),
            ]
            if item
        )
    )
    adverse_events = sorted(
        set(
            item
            for item in [
                *(_coalesce_text(item) for item in row.get("adverse_events", []) if isinstance(row.get("adverse_events"), list)),
                _coalesce_text(row.get("adverse_event") or row.get("ade") or row.get("reaction")),
                *_extract_typed_entities(row.get("entities"), needles=("adverse", "ade", "reason")),
            ]
            if item
        )
    )

    relations: list[str] = []
    raw_relations = row.get("relations")
    if isinstance(raw_relations, list):
        for relation in raw_relations:
            if isinstance(relation, dict):
                relation_type = _normalize_label(relation.get("type") or relation.get("label"))
                arg1 = _coalesce_text(relation.get("arg1") or relation.get("head") or relation.get("source"))
                arg2 = _coalesce_text(relation.get("arg2") or relation.get("tail") or relation.get("target"))
                if relation_type or arg1 or arg2:
                    relations.append(" | ".join(part for part in [relation_type, arg1, arg2] if part))
            else:
                text = _coalesce_text(relation)
                if text:
                    relations.append(text)

    return BenchmarkTask(
        task_id=f"n2c2-2018-track2-{row.get('id', row.get('doc_id', row.get('document_id', 'unknown')))}",
        source_dataset="n2c2_2018_track2",
        split=split,
        task_type="medication_safety",
        prompt=prompt,
        gold_answer=relations[0] if relations else None,
        metadata={
            "gold_medications": medications,
            "gold_adverse_events": adverse_events,
            "gold_relations": relations,
            "credential_gated": True,
        },
    )
