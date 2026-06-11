"""
Centralized prompt definitions for LEXIS to ensure consistency and easy tweaking.
"""

ELEMENT_CHECK_PROMPT = """You are a strict legal verification judge.
Does the query scenario satisfy the constitutive elements of this legal clause?

Query: "{query}"

Required Elements (ALL must be satisfied):
{required_elements}

Optional Elements (used for threshold scoring):
{optional_elements}

Output ONLY valid JSON matching this schema:
{{
    "required": {{"element_1": true/false, ...}},
    "optional": {{"element_1": true/false, ...}}
}}
"""

EVIDENCE_EXTRACTION_PROMPT = """You are a legal research analyst.
Extract exactly WHY this text answers the query. 
If it does NOT answer the query, output exactly: "NO_EVIDENCE".

Query: "{query}"

Text: "{text}"

Output your reasoning succinctly.
"""

CONCEPT_EXTRACTION_PROMPT = """Extract up to 3 key conceptual entities from the user's query. 
Output ONLY valid JSON matching this schema: {"concepts": ["concept1", "concept2"]} 
"""
