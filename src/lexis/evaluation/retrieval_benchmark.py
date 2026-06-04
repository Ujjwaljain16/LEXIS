import logging
from typing import List
from lexis.retrieval.interfaces import Query, Candidate

logger = logging.getLogger(__name__)

def assert_path_recall(path_candidates: List[Candidate], expected_chunk_ids: List[str], min_recall: float = 0.8):
    """
    CI Gate: Ensures a specific retrieval path hits a minimum recall threshold 
    for a known dataset of gold standard queries.
    """
    retrieved_ids = {c.chunk_id for c in path_candidates}
    expected_set = set(expected_chunk_ids)
    
    hits = len(retrieved_ids.intersection(expected_set))
    recall = hits / len(expected_set) if expected_set else 1.0
    
    assert recall >= min_recall, f"Recall {recall:.2f} failed to meet threshold {min_recall:.2f}"
    logger.info(f"Path recall check passed: {recall:.2f} >= {min_recall:.2f}")

def assert_rrf_improvement(
    single_path_candidates: List[List[Candidate]], 
    fused_candidates: List[Candidate], 
    expected_chunk_ids: List[str]
):
    """
    CI Gate: Ensures that Reciprocal Rank Fusion (RRF) actually improves Mean Reciprocal Rank (MRR)
    over the best individual path. If it doesn't, RRF is degrading performance.
    """
    def compute_mrr(candidates: List[Candidate], expected: List[str]) -> float:
        expected_set = set(expected)
        for i, c in enumerate(candidates):
            if c.chunk_id in expected_set:
                return 1.0 / (i + 1)
        return 0.0

    best_single_mrr = max(compute_mrr(path, expected_chunk_ids) for path in single_path_candidates)
    fused_mrr = compute_mrr(fused_candidates, expected_chunk_ids)

    assert fused_mrr >= best_single_mrr, (
        f"RRF degraded performance! Fused MRR: {fused_mrr:.3f} < Best Single MRR: {best_single_mrr:.3f}"
    )
    logger.info(f"RRF improvement check passed: Fused ({fused_mrr:.3f}) >= Best Single ({best_single_mrr:.3f})")
