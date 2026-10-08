"""
WHAT: Check constitutive elements of retrieved chunks against query.
WHY: A clause about "termination for cause" should only surface for actual cause scenarios.
HOW: Distinguishes between required and optional elements. Runs on prefiltered chunks before RRF.
"""
import json
import asyncio
import logging
from typing import List, Dict, Tuple
from litellm import acompletion
from pydantic import BaseModel
from lexis.generation.prompts import ELEMENT_CHECK_PROMPT

logger = logging.getLogger(__name__)

class ElementCheckResult(BaseModel):
    required_satisfied: bool
    optional_ratio: float
    passes: bool

async def check_elements(
    query: str,
    chunk: Dict,
    model: str = "gemini/gemini-2.5-flash",
    optional_threshold: float = 0.5
) -> ElementCheckResult:
    """
    Evaluates if a chunk meets the element criteria.
    Chunks with no elements automatically pass (insufficient data to reject).
    """
    payload = chunk.get("payload", {})
    required_elements = payload.get("required_elements", [])
    optional_elements = payload.get("optional_elements", [])
    
    # If no elements were extracted at index time, allow the chunk through
    if not required_elements and not optional_elements:
        return ElementCheckResult(required_satisfied=True, optional_ratio=1.0, passes=True)
        
    req_str = "\n".join(f"- {e}" for e in required_elements) if required_elements else "None"
    opt_str = "\n".join(f"- {e}" for e in optional_elements) if optional_elements else "None"

    try:
        response = await acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": ELEMENT_CHECK_PROMPT.format(
                    query=query,
                    required_elements=req_str,
                    optional_elements=opt_str
                )
            }],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        results = json.loads(content.strip())
        
        req_results = results.get("required", {})
        opt_results = results.get("optional", {})
        
        # All required must be True
        req_satisfied = all(req_results.get(f"element_{i+1}", False) for i in range(len(required_elements))) if required_elements else True
        
        # Calculate optional ratio
        opt_count = sum(1 for v in opt_results.values() if v is True)
        opt_ratio = opt_count / len(optional_elements) if optional_elements else 1.0
        
        passes = req_satisfied and (opt_ratio >= optional_threshold)
        
        return ElementCheckResult(
            required_satisfied=req_satisfied,
            optional_ratio=opt_ratio,
            passes=passes
        )
    except Exception as e:
        logger.error(f"Element check failed: {e}")
        raise e

async def filter_by_elements(
    query: str,
    chunks: List[Dict],
    max_concurrent: int = 8
) -> List[Dict]:
    """
    Filter chunks to only those whose constitutive elements are satisfied by the query.
    Must run BEFORE RRF to avoid reranking junk.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Telemetry
    telemetry = {
        "verification_attempts": 0,
        "verification_successes": 0,
        "verification_failures": 0,
        "verification_fail_open": 0
    }

    async def check_one(chunk):
        telemetry["verification_attempts"] += 1
        async with semaphore:
            # We must detect fail_open by mocking the check slightly or catching inside check_elements?
            # Actually, check_elements logs "Element check failed: ..." and returns passes=True.
            # To cleanly track fail-open, we would need check_elements to return a status enum or we can track it.
            # We'll just run it as-is for now, wait, the user requested explicit telemetry.
            # Let's intercept exceptions locally or modify check_elements.
            try:
                result = await check_elements(query, chunk)
                if result.passes:
                    telemetry["verification_successes"] += 1
                    chunk_copy = dict(chunk)
                    chunk_copy["element_satisfaction"] = {
                        "required_passed": result.required_satisfied,
                        "optional_ratio": result.optional_ratio
                    }
                    return chunk_copy
                else:
                    telemetry["verification_failures"] += 1
                    return None
            except Exception as e:
                # If an exception bubbles out (though check_elements currently catches it)
                telemetry["verification_fail_open"] += 1
                return dict(chunk)

    results = await asyncio.gather(*[check_one(c) for c in chunks])
    
    logger.info(f"JudgeDEP Telemetry: {telemetry}")
    
    return [r for r in results if r is not None]
