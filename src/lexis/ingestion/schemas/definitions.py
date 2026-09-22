"""
Schema definitions for Feature Extraction.
These are decoupled from the extraction engine to allow easy extensions.
"""

GENERIC_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "string"}},
        "concepts": {"type": "array", "items": {"type": "string"}},
        "dates": {"type": "array", "items": {"type": "string"}},
        "amounts": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "relationships": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["entities", "concepts", "dates", "amounts", "risks", "relationships"]
}

LEGAL_SCHEMA = {
    "type": "object",
    "properties": {
        "parties": {"type": "array", "items": {"type": "string"}},
        "obligations": {"type": "array", "items": {"type": "string"}},
        "governing_law": {"type": "array", "items": {"type": "string"}},
        "termination": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["parties", "obligations", "governing_law", "termination"]
}

FINANCIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "financial_metrics": {"type": "array", "items": {"type": "string"}},
        "material_events": {"type": "array", "items": {"type": "string"}},
        "forward_looking": {"type": "boolean"},
    },
    "required": ["financial_metrics", "material_events", "forward_looking"]
}
