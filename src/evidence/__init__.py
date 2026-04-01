"""Evidence registries and provenance-aware domain knowledge."""

from evidence.integration import enrich_candidate_test, enrich_hypothesis_evidence, likelihood_ratio_from_registry_entry
from evidence.query import RegistryResolution, query_registry, resolve_registry_entry
from evidence.registry_loader import (
    DEFAULT_CSV_PATH,
    DEFAULT_CURATED_CSV_PATH,
    DEFAULT_CURATED_JSONL_PATH,
    DEFAULT_JSONL_PATH,
    LREntry,
    load_curated_registry,
    load_default_registry_bundle,
    load_registry,
    load_seed_registry,
)
from evidence.registry_validator import RegistryValidationError, validate_registry_entries, validate_registry_file

__all__ = [
    "DEFAULT_CSV_PATH",
    "DEFAULT_CURATED_CSV_PATH",
    "DEFAULT_CURATED_JSONL_PATH",
    "DEFAULT_JSONL_PATH",
    "LREntry",
    "RegistryResolution",
    "RegistryValidationError",
    "enrich_candidate_test",
    "enrich_hypothesis_evidence",
    "likelihood_ratio_from_registry_entry",
    "load_curated_registry",
    "load_default_registry_bundle",
    "load_registry",
    "load_seed_registry",
    "query_registry",
    "resolve_registry_entry",
    "validate_registry_entries",
    "validate_registry_file",
]
