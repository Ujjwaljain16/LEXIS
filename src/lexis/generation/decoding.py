"""
WHAT: Dynamic decoding policy for generation.
WHY: High confidence retrieval -> deterministic generation. Low confidence -> slightly more exploratory.
HOW: Returns kwargs for litellm.acompletion based on retrieval confidence.
"""
from typing import Dict

def get_decoding_policy(calibrated_confidence: float) -> Dict[str, float]:
    """
    Returns LLM sampling parameters based on the confidence of the retrieved context.
    For Lexis, we want deterministic answers (temp=0.0) when we have strong evidence.
    If evidence is weaker, we allow slight temperature (0.1) but cap top_p tightly.
    """
    if calibrated_confidence > 0.8:
        return {
            "temperature": 0.0,
            "top_p": 0.1
        }
    elif calibrated_confidence > 0.5:
        return {
            "temperature": 0.1,
            "top_p": 0.3
        }
    else:
        # Very low confidence (possibly web fallback or poor context)
        # Keep temperature low to prevent hallucination when compensating for bad context
        return {
            "temperature": 0.0,
            "top_p": 0.1
        }
