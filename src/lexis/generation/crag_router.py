"""
WHAT: Route to web search when local corpus confidence is too low.
WHY: Protects against corpus gaps; prevents hallucination from "pretending" to know.
HOW: Uses Brave Search API. Modifies retrieved web chunk scores via TrustScorer.
"""
import os
import httpx
import logging
from typing import List, Dict
from .trust_scorer import get_scorer

logger = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

async def get_web_context(query: str, n_results: int = 3) -> List[Dict]:
    """Fetch web results from Brave Search and apply trust multipliers."""
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        # Graceful disable
        logger.warning("No Brave Search API key — skipping web fallback")
        return []

    trust_scorer = get_scorer()
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                BRAVE_SEARCH_URL,
                headers={"X-Subscription-Token": api_key},
                params={"q": query, "count": 10, "search_lang": "en"},
                timeout=5.0,
            )
            results = response.json().get("web", {}).get("results", [])

            # Apply trust and freshness scoring
            trusted_chunks = []
            for r in results:
                url = r.get("url", "")
                trust_score = trust_scorer.get_trust_score(url)
                
                # Freshness scoring: if result has a recent 'age' or 'date', boost it.
                # Brave search gives 'page_age'. For now, naive freshness heuristic:
                age = r.get("page_age", "")
                freshness_score = 1.0 if ("day" in age or "hour" in age or "minute" in age) else (0.8 if "month" in age else 0.6)
                
                # Base retrieval score (heuristic mock since Brave doesn't give vector scores)
                base_score = 0.8
                final_score = base_score * trust_score * freshness_score
                
                trusted_chunks.append({
                    "payload": {
                        "content": r.get("description", ""),
                        "source_file": url,
                        "title": r.get("title", ""),
                        "doc_type": "web_search",
                        "age": age
                    },
                    "cross_encoder_score": final_score,
                    "trust_score": trust_score,
                    "freshness_score": freshness_score
                })
                
            # Sort by final score
            trusted_chunks.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
            return trusted_chunks[:n_results]
            
        except Exception as e:
            logger.error(f"CRAG Web Fallback failed: {e}")
            return []

async def route_crag(
    query: str, 
    local_chunks: List[Dict], 
    calibrated_confidence: float,
    confidence_threshold: float = 0.4
) -> List[Dict]:
    """
    If local confidence is below threshold, fetch web chunks and merge them into the pool.
    """
    if calibrated_confidence >= confidence_threshold:
        return local_chunks
        
    logger.info(f"CRAG triggered (Confidence {calibrated_confidence:.2f} < {confidence_threshold})")
    web_chunks = await get_web_context(query)
    
    # Merge and re-sort
    combined = local_chunks + web_chunks
    combined.sort(key=lambda x: x.get("cross_encoder_score", 0.0), reverse=True)
    return combined
