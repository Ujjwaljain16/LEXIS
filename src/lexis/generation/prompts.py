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

GROUNDED_ANSWER_SYSTEM_PROMPT = """You answer questions about documents using ONLY the numbered sources provided.
Rules:
- Every factual statement must end with the number(s) of the source(s) that support it, like [1] or [2, 3].
- Do not use outside knowledge. Do not cite a number that is not in the sources.
- If the sources do not contain the answer, reply with exactly: {abstention_marker}
- Quote or closely paraphrase the source; do not add details the sources do not state."""

GROUNDED_ANSWER_USER_PROMPT = """Sources:
{context}

Question: {query}"""
