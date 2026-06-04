"""
WHAT: Computes raw retrieval heuristics (Top-1, Score Gap, Path Agreement).
WHY: Necessary as inputs for the Platt Scaling confidence calibration.
"""
from typing import List, Dict
from pydantic import BaseModel

class ConfidenceHeuristics(BaseModel):
    top1_score: float
    score_spread: float
    path_agreement: float
    reranker_confidence: float

def compute_heuristics(
    reranked_chunks: List[Dict],
    fusion_scores: Dict[str, float]
) -> ConfidenceHeuristics:
    """
    Computes raw heuristics from a set of reranked candidate chunks.
    """
    if not reranked_chunks:
        return ConfidenceHeuristics(
            top1_score=0.0,
            score_spread=0.0,
            path_agreement=0.0,
            reranker_confidence=0.0
        )
        
    top1 = reranked_chunks[0]
    top1_score = top1.get("cross_encoder_score", 0.0)
    
    # Calculate score spread (gap between #1 and #5)
    score_spread = 0.0
    if len(reranked_chunks) >= 5:
        top5_score = reranked_chunks[4].get("cross_encoder_score", 0.0)
        score_spread = top1_score - top5_score
        
    # Calculate path agreement (how many paths found the top chunks)
    # E.g., if BM25 and Dense both found it, agreement is high.
    agreement_count = 0
    total_top_paths = 0
    for chunk in reranked_chunks[:5]:
        paths = chunk.get("paths", [])
        if len(paths) > 1:
            agreement_count += 1
        total_top_paths += 1
        
    path_agreement = agreement_count / max(1, total_top_paths)
    
    # Use the cross_encoder score directly as reranker confidence
    reranker_confidence = top1_score
    
    return ConfidenceHeuristics(
        top1_score=top1_score,
        score_spread=score_spread,
        path_agreement=path_agreement,
        reranker_confidence=reranker_confidence
    )
