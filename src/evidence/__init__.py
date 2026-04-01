"""Evidence registries and provenance-aware domain knowledge."""

from evidence.query import RegistryResolution, query_registry, resolve_registry_entry
from evidence.registry_loader import DEFAULT_CSV_PATH, DEFAULT_JSONL_PATH, LREntry, load_registry, load_seed_registry
from evidence.registry_validator import RegistryValidationError, validate_registry_entries, validate_registry_file

__all__ = [
    "DEFAULT_CSV_PATH",
    "DEFAULT_JSONL_PATH",
    "LREntry",
    "RegistryResolution",
    "RegistryValidationError",
    "load_registry",
    "load_seed_registry",
    "query_registry",
    "resolve_registry_entry",
    "validate_registry_entries",
    "validate_registry_file",
]
