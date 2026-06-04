"""
WHAT: Decomposes complex multi-hop or comparative queries into independent subqueries.
WHY: Dense retrieval fails on compound queries. Breaking them down drastically improves recall.
HOW: LLM parses the query into a JSON array of search-optimized subqueries.
"""
import json
import logging
from typing import List
from litellm import acompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class DecomposedQuery(BaseModel):
    original_query: str
    subqueries: List[str]
    is_complex: bool

DECOMPOSITION_PROMPT = """You are a master legal and financial research assistant.
Your task is to decompose a complex, compound, or comparative query into a set of independent, focused subqueries.
These subqueries will be fed into a vector search engine to find evidence.

Original Query: "{query}"

Rules:
1. Break the query into 2-4 atomic questions.
2. Each subquery must be fully self-contained (resolve pronouns and implicit context).
3. Frame them as natural search queries.
4. Output ONLY a valid JSON object matching this schema:
{{
    "subqueries": ["independent query 1", "independent query 2"]
}}
"""

async def decompose_query(query: str, model: str = "gemini/gemini-2.5-flash") -> DecomposedQuery:
    """
    Splits a complex query into independent subqueries using an LLM.
    Returns the original query in the list as well, to ensure global context isn't lost.
    """
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": DECOMPOSITION_PROMPT.format(query=query)}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content.strip()
        data = json.loads(raw_content)
        
        subqueries = data.get("subqueries", [])
        
        # Always include the original query as the first item to maintain global context
        if query not in subqueries:
            subqueries.insert(0, query)
            
        return DecomposedQuery(
            original_query=query,
            subqueries=subqueries,
            is_complex=len(subqueries) > 1
        )
        
    except Exception as e:
        logger.error(f"Query decomposition failed: {e}")
        # Fail safe: return the original query as a single subquery
        return DecomposedQuery(
            original_query=query,
            subqueries=[query],
            is_complex=False
        )
