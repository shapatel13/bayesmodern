from __future__ import annotations

from pathlib import Path

from evidence.registry_loader import LREntry, load_registry


class RegistryValidationError(ValueError):
    """Raised when the LR registry violates structural or semantic rules."""


def validate_registry_entries(entries: list[LREntry]) -> list[LREntry]:
    seen_ids: set[str] = set()
    for entry in entries:
        if entry.id in seen_ids:
            raise RegistryValidationError(f"Duplicate LR registry id detected: {entry.id}")
        seen_ids.add(entry.id)
    return entries


def validate_registry_file(path: str | Path) -> list[LREntry]:
    return validate_registry_entries(load_registry(path))

