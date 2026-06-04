"""
WHAT: Maps raw extracted metadata strings into strict canonical IDs.
WHY: Raw string variants (e.g., 'NCT Delhi' vs 'Delhi') break strict filtering.
HOW: Loads registry.yaml and maps via dictionary lookup.
"""
import os
import yaml
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.yaml")

class MetadataCanonicalizer:
    def __init__(self, registry_path: str = REGISTRY_PATH):
        self.registry = self._load_registry(registry_path)
        self.jurisdiction_map = self._build_reverse_map(self.registry.get("jurisdictions", {}))
        self.doc_type_map = self._build_reverse_map(self.registry.get("document_types", {}))

    def _load_registry(self, path: str) -> dict:
        if not os.path.exists(path):
            logger.warning(f"Registry not found at {path}")
            return {}
        with open(path, "r") as f:
            try:
                return yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Failed to parse registry: {e}")
                return {}

    def _build_reverse_map(self, section: Dict[str, list]) -> Dict[str, str]:
        """Flattens lists of aliases into a dict mapping alias -> ID"""
        reverse_map = {}
        for canonical_id, aliases in section.items():
            for alias in aliases:
                reverse_map[alias.lower().strip()] = canonical_id
        return reverse_map

    def get_jurisdiction_id(self, raw_string: str) -> Optional[str]:
        if not raw_string:
            return None
        return self.jurisdiction_map.get(raw_string.lower().strip())

    def get_document_type_id(self, raw_string: str) -> Optional[str]:
        if not raw_string:
            return None
        return self.doc_type_map.get(raw_string.lower().strip())

# Singleton
_canonicalizer = None
def get_canonicalizer() -> MetadataCanonicalizer:
    global _canonicalizer
    if _canonicalizer is None:
        _canonicalizer = MetadataCanonicalizer()
    return _canonicalizer
