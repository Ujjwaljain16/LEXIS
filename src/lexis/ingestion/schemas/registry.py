import logging
from typing import Optional, Dict, Any
from .definitions import GENERIC_SCHEMA

logger = logging.getLogger(__name__)

class FeatureSchemaRegistry:
    """
    Registry for document-type specific feature schemas.
    Follows Open/Closed Principle: new document types can be added without modifying the core extraction engine.
    """
    def __init__(self):
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, str] = {}
        self._instructions: Dict[str, str] = {}
        
    def register(
        self,
        doc_type: str,
        schema: dict,
        version: str = "v1",
        extraction_instructions: Optional[str] = None
    ):
        """Register a schema for a specific document type."""
        # Merge with Generic Schema as the base layer
        combined_schema = {
            "type": "object",
            "properties": {**GENERIC_SCHEMA["properties"], **schema.get("properties", {})},
            "required": list(set(GENERIC_SCHEMA.get("required", []) + schema.get("required", [])))
        }
        
        doc_type = doc_type.lower()
        self._schemas[doc_type] = combined_schema
        self._versions[doc_type] = version
        self._instructions[doc_type] = extraction_instructions or ""
        logger.debug("Registered schema for '%s' (version: %s)", doc_type, version)

    def get_schema(self, doc_type: str) -> dict:
        """Returns the schema for the doc type, or GENERIC_SCHEMA if unknown."""
        doc_type = doc_type.lower()
        if doc_type not in self._schemas:
            logger.debug("Using fallback GENERIC_SCHEMA for unknown doc_type='%s'", doc_type)
            return GENERIC_SCHEMA
            
        logger.debug("Using registered feature schema for doc_type='%s'", doc_type)
        return self._schemas[doc_type]

    def get_version(self, doc_type: str) -> str:
        """Returns the version of the schema, defaults to 'v1' for generic."""
        return self._versions.get(doc_type.lower(), "v1")
        
    def get_instructions(self, doc_type: str) -> str:
        """Returns any specific extraction instructions for the doc type."""
        return self._instructions.get(doc_type.lower(), "")
