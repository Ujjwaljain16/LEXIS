"""
Document-level routing signals (plan R6), pure and benchmark-agnostic.

vote_document(): pick ONE document from an already-ranked chunk list by
summing reciprocal-rank weight per document -- "which document do the
top-ranked chunks collectively point to?". It uses only the ranked list
(query-derived retrieval output); it never sees ground truth. The RRF
constant is an input (the same one fusion uses), not a literal here.
"""
from collections import defaultdict
from typing import Dict, Mapping, Optional, Sequence


def document_scores(ranked: Sequence[Mapping], k: int) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for rank, item in enumerate(ranked, start=1):
        doc_id = item.get("doc_id")
        if doc_id is not None:
            scores[doc_id] += 1.0 / (k + rank)
    return dict(scores)


def vote_document(ranked: Sequence[Mapping], k: int) -> Optional[str]:
    """Highest-scoring document; ties break by document id (deterministic).
    None if the list has no document ids."""
    scores = document_scores(ranked, k)
    if not scores:
        return None
    return min(scores, key=lambda d: (-scores[d], d))
