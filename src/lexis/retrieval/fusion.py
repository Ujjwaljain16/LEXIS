from typing import List, Dict
from collections import defaultdict
from lexis.retrieval.interfaces import Candidate

def apply_rrf(candidate_lists: List[List[Candidate]], k: int = 60) -> List[Candidate]:
    """
    Applies Reciprocal Rank Fusion (RRF) across multiple candidate lists.
    RRF score = sum(1 / (k + rank_in_path)) for each path where candidate appears.
    
    Args:
        candidate_lists: A list where each element is a list of Candidates from a specific retrieval path.
        k: The RRF constant (default 60 is standard in IR literature).
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    candidate_map: Dict[str, Candidate] = {}
    
    # Compute RRF score for each chunk
    for path_idx, candidates in enumerate(candidate_lists):
        for rank, candidate in enumerate(candidates):
            # Rank is 0-indexed, standard RRF uses 1-indexed rank
            rrf_score = 1.0 / (k + (rank + 1))
            rrf_scores[candidate.chunk_id] += rrf_score
            
            if candidate.chunk_id not in candidate_map:
                candidate_map[candidate.chunk_id] = candidate
            else:
                # Merge metadata/sources to show it was found in multiple paths
                existing = candidate_map[candidate.chunk_id]
                if candidate.source_path not in existing.source_path:
                    existing.source_path += f",{candidate.source_path}"
                    
    # Re-score and sort based on fused score
    fused_candidates = []
    for chunk_id, fused_score in rrf_scores.items():
        base_candidate = candidate_map[chunk_id]
        fused_candidates.append(
            Candidate(
                chunk_id=base_candidate.chunk_id,
                score=fused_score,
                source_path=base_candidate.source_path,
                metadata=base_candidate.metadata,
                content=base_candidate.content
            )
        )
        
    fused_candidates.sort(key=lambda c: c.score, reverse=True)
    return fused_candidates
