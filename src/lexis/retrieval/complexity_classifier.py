"""
WHAT: Routes queries to either standard retrieval or deep mode decomposition.
WHY: Avoids LLM taxation on simple queries; correctly routes multi-hop questions.
HOW: Fast rule heuristics first, falling back to an LLM classifier if ambiguous.
"""
import logging
from enum import Enum
from typing import Literal
from pydantic import BaseModel
from litellm import acompletion

logger = logging.getLogger(__name__)

class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"

class ClassifierResult(BaseModel):
    complexity: QueryComplexity
    reasoning: str
    confidence: float
    used_llm: bool

COMPLEX_KEYWORDS = {
    "compare", "differences", "versus", "vs", "similarities",
    "conflict", "contradict", "supersede", "reconcile", "summarize across"
}

LOGICAL_CONNECTORS = {"and", "or", "but", "however", "although", "unless", "except"}

def _fast_heuristic_classifier(query: str) -> ClassifierResult | None:
    """
    Stage 1: Rule heuristics.
    Returns result if highly confident, else None (to trigger LLM escalation).
    """
    query_lower = query.lower()
    words = query_lower.split()
    
    # 1. Extremely short queries are almost always simple entity/fact lookups
    if len(words) <= 5:
        return ClassifierResult(
            complexity=QueryComplexity.SIMPLE,
            reasoning="Query is 5 words or less; typical fact lookup.",
            confidence=0.95,
            used_llm=False
        )
        
    # 2. Check for explicit multi-document comparison keywords
    if any(kw in query_lower for kw in COMPLEX_KEYWORDS):
        return ClassifierResult(
            complexity=QueryComplexity.COMPLEX,
            reasoning="Contains explicit comparison/synthesis keywords.",
            confidence=0.90,
            used_llm=False
        )
        
    # 3. High logical complexity (long query + multiple connectors)
    connector_count = sum(1 for w in words if w in LOGICAL_CONNECTORS)
    if len(words) > 20 and connector_count >= 3:
        return ClassifierResult(
            complexity=QueryComplexity.COMPLEX,
            reasoning="Long query with multiple logical connectors.",
            confidence=0.85,
            used_llm=False
        )
        
    # Ambiguous. Escalate to LLM.
    return None

async def _llm_classifier(query: str, model: str = "gemini/gemini-2.5-flash") -> ClassifierResult:
    """
    Stage 2: LLM escalation for ambiguous queries.
    """
    prompt = f"""Analyze the complexity of this legal/financial research query.
A SIMPLE query asks for a specific fact, definition, or clause from a single logical location.
A COMPLEX query requires comparing multiple documents, synthesizing scattered evidence, resolving conflicts, or deep multi-hop reasoning.

Query: "{query}"

Output exactly one word: SIMPLE or COMPLEX.
"""
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        decision = response.choices[0].message.content.strip().upper()
        
        complexity = QueryComplexity.COMPLEX if "COMPLEX" in decision else QueryComplexity.SIMPLE
        
        return ClassifierResult(
            complexity=complexity,
            reasoning="LLM escalation decision.",
            confidence=0.90,
            used_llm=True
        )
    except Exception as e:
        logger.error(f"LLM Complexity Classifier failed: {e}")
        # Fail safe to SIMPLE to avoid blocking the user
        return ClassifierResult(
            complexity=QueryComplexity.SIMPLE,
            reasoning=f"LLM fallback failed: {e}",
            confidence=0.1,
            used_llm=True
        )

async def classify_query(query: str, llm_model: str = "gemini/gemini-2.5-flash") -> ClassifierResult:
    """
    Main entry point. Runs fast heuristics, then LLM if necessary.
    """
    fast_result = _fast_heuristic_classifier(query)
    if fast_result:
        return fast_result
        
    logger.debug(f"Query ambiguous, escalating to LLM classifier: '{query}'")
    return await _llm_classifier(query, model=llm_model)
