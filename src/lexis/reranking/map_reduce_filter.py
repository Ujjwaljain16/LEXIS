"""
WHAT: Cost-controlled, failure-isolated Deep Mode Map Reduce.
WHY: Transforms candidate chunks into high-signal Evidence Nodes.
HOW: Uses EvidenceSummaryCache, runs async LLM maps, and builds a ResearchGraph.
"""
import hashlib
import asyncio
import logging
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel, Field
from litellm import acompletion

from ..cache.base import CachedEvidence
from ..cache.redis_cache import get_cache

logger = logging.getLogger(__name__)

# --- Data Structures ---

class ResearchBudget(BaseModel):
    max_subqueries: int = 4
    max_map_calls: int = 20
    max_cost_usd: float = 0.5

class ResearchNode(BaseModel):
    query: str
    evidence_text: str
    citations: List[str]
    confidence_score: float

class ResearchGraph(BaseModel):
    nodes: List[ResearchNode] = Field(default_factory=list)
    edges: List[Dict] = Field(default_factory=list)

class ResearchSession(BaseModel):
    session_id: str
    query: str
    graph: ResearchGraph
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# --- Prompts ---

EVIDENCE_EXTRACTION_PROMPT = """You are a legal research analyst.
Extract exactly WHY this text answers the query. 
If it does NOT answer the query, output exactly: "NO_EVIDENCE".

Query: "{query}"

Text: "{text}"

Output your reasoning succinctly.
"""

# --- Core Logic ---

def _hash_query(query: str) -> str:
    return hashlib.md5(query.lower().encode()).hexdigest()

async def _map_single_chunk(query: str, query_hash: str, chunk: Dict, model: str) -> Optional[ResearchNode]:
    """Processes a single chunk, utilizing the EvidenceSummaryCache."""
    cache = get_cache()
    payload = chunk.get("payload", {})
    chunk_id = payload.get("chunk_id", "unknown")
    text = payload.get("content", "")
    pqac_key = payload.get("pqac_key", "unknown")
    
    # 1. Cache Check
    cached = await cache.get(chunk_id, query_hash)
    if cached:
        if cached.reason == "NO_EVIDENCE":
            return None
        return ResearchNode(
            query=query,
            evidence_text=cached.reason,
            citations=[pqac_key],
            confidence_score=cached.score
        )
        
    # 2. LLM Call
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": EVIDENCE_EXTRACTION_PROMPT.format(query=query, text=text)}],
            temperature=0.0
        )
        reason = response.choices[0].message.content.strip()
        
        # 3. Cache Miss Update
        await cache.set(CachedEvidence(
            chunk_id=chunk_id,
            query_hash=query_hash,
            reason=reason,
            score=chunk.get("cross_encoder_score", 0.5),
            timestamp=datetime.utcnow()
        ))
        
        if reason == "NO_EVIDENCE":
            return None
            
        return ResearchNode(
            query=query,
            evidence_text=reason,
            citations=[pqac_key],
            confidence_score=chunk.get("cross_encoder_score", 0.5)
        )
        
    except Exception as e:
        logger.error(f"Map extraction failed for chunk {chunk_id}: {e}")
        raise e  # Let return_exceptions=True catch this

async def map_reduce_deep_mode(
    session_id: str,
    query: str,
    subqueries: List[str],
    chunks: List[Dict],
    budget: ResearchBudget,
    model: str = "gemini/gemini-2.5-flash",
    min_evidence_threshold: float = 0.6
) -> ResearchSession:
    """
    Executes the Map phase under strict budget constraints and builds the ResearchGraph.
    """
    if len(subqueries) > budget.max_subqueries:
        subqueries = subqueries[:budget.max_subqueries]
        
    if len(chunks) > budget.max_map_calls:
        chunks = chunks[:budget.max_map_calls]
        
    query_hash = _hash_query(query)
    
    # Fire async map calls with failure isolation
    tasks = [_map_single_chunk(query, query_hash, c, model) for c in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Bucket successes vs failures
    nodes = []
    failures = 0
    for res in results:
        if isinstance(res, Exception):
            failures += 1
        elif res is not None:
            nodes.append(res)
            
    # Check minimum evidence threshold
    total_maps = len(chunks)
    success_rate = (total_maps - failures) / max(1, total_maps)
    
    if success_rate < min_evidence_threshold:
        logger.warning(f"Deep Mode Map Phase aborted: Success rate {success_rate:.2f} < {min_evidence_threshold}")
        # Fallback to returning an empty graph so synthesis knows it failed
        return ResearchSession(
            session_id=session_id,
            query=query,
            graph=ResearchGraph()
        )
        
    graph = ResearchGraph(nodes=nodes)
    
    # TODO: In production, save ResearchSession to PostgreSQL here
    # session_db.save(session)
    
    return ResearchSession(
        session_id=session_id,
        query=query,
        graph=graph
    )
