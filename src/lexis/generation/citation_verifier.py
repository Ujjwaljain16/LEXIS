"""
WHAT: Moat 2: Citation Verification and Grounding.
WHY: Prevents citation hallucination. Extracts supporting spans.
HOW: Stage 1 Fast Similarity. Stage 2 Batched LLM. Enforces strict Latency Budgets.
"""
import re
import time
import json
import asyncio
import logging
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel
from litellm import acompletion

logger = logging.getLogger(__name__)

MAX_VERIFICATION_LATENCY_MS = 1500

class CitationEvidence(BaseModel):
    citation_id: str
    supporting_span: Optional[str]
    support_score: float
    is_valid: bool

class ClaimCitationPair(BaseModel):
    claim_text: str
    citation_id: str
    cited_text: str

class GroundedAnswer(BaseModel):
    answer: str
    citations: List[CitationEvidence]
    confidence: float
    is_grounded: bool

# Dummy Stage 1 Similarity
def semantic_similarity(text1: str, text2: str) -> float:
    """Mock similarity until embeddings are hooked up."""
    t1 = set(text1.lower().split())
    t2 = set(text2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1.intersection(t2)) / float(min(len(t1), len(t2)))

JUDGE_PROMPT = """You are a strict legal citation auditor.
Verify if the cited text semantically supports the generated claim.

Claim: {claim}

Cited Text: {text}

Output JSON:
{{
    "supports": true/false,
    "supporting_span": "exact quote from text that supports the claim",
    "score": 0.0 to 1.0
}}
"""

class CitationSupportJudge:
    async def _verify_single(self, pair: ClaimCitationPair, model: str) -> CitationEvidence:
        # Stage 1: Fast similarity
        sim_score = semantic_similarity(pair.claim_text, pair.cited_text)
        if sim_score > 0.8:
            # So similar we don't need LLM
            return CitationEvidence(
                citation_id=pair.citation_id,
                supporting_span=pair.cited_text[:100],  # Mock span
                support_score=sim_score,
                is_valid=True
            )
            
        # Stage 2: LLM
        try:
            response = await acompletion(
                model=model,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(claim=pair.claim_text, text=pair.cited_text)}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            res = json.loads(response.choices[0].message.content.strip())
            supports = res.get("supports", False)
            return CitationEvidence(
                citation_id=pair.citation_id,
                supporting_span=res.get("supporting_span"),
                support_score=res.get("score", 0.0),
                is_valid=supports
            )
        except Exception as e:
            logger.error(f"Citation judge failed: {e}")
            # Fallback invalid
            return CitationEvidence(
                citation_id=pair.citation_id,
                supporting_span=None,
                support_score=0.0,
                is_valid=False
            )

    async def verify_batch(self, claims: List[ClaimCitationPair], model: str = "gemini/gemini-2.5-flash") -> List[CitationEvidence]:
        start_time = time.time() * 1000
        
        # We fire them concurrently
        tasks = [self._verify_single(c, model) for c in claims]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for i, res in enumerate(results):
            elapsed = (time.time() * 1000) - start_time
            if elapsed > MAX_VERIFICATION_LATENCY_MS:
                logger.warning(f"Verification latency exceeded {MAX_VERIFICATION_LATENCY_MS}ms. Aborting remaining batch.")
                # Force fallback valid based on similarity only to prevent tail latency explosions
                for j in range(i, len(claims)):
                    sim = semantic_similarity(claims[j].claim_text, claims[j].cited_text)
                    valid_results.append(CitationEvidence(
                        citation_id=claims[j].citation_id,
                        supporting_span=None,
                        support_score=sim,
                        is_valid=sim > 0.5
                    ))
                break
                
            if isinstance(res, Exception):
                logger.error(f"Batch element failed: {res}")
                valid_results.append(CitationEvidence(
                    citation_id=claims[i].citation_id,
                    supporting_span=None,
                    support_score=0.0,
                    is_valid=False
                ))
            else:
                valid_results.append(res)
                
        return valid_results
