"""
Case-level oracle analysis (plan P0): how high can MRR go on our strict
chunk-level labels, and WHY does it stop short?

Given a ranked list produced for one case (by any query, including an
oracle query built from the answer text), this module computes case-level
Recall/MRR with the project's existing metric functions and, when the first
relevant chunk is not ranked first, classifies what sits above it:

  different_document           the top chunk is from another document
                               (a document-routing problem)
  same_doc_high_answer_overlap the top chunk is from the right document and
                               contains nearly all of the answer's tokens
                               but is not labelled relevant -- evidence the
                               label is stricter than the answer's meaning
                               (answer split across chunk boundaries,
                               near-duplicate clauses)
  same_doc_low_answer_overlap  right document, little answer overlap: an
                               ordinary ranking error

The overlap threshold is an input (config), not a constant here. Nothing in
this module knows about any benchmark.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from lexis.evaluation.hard_miss_diagnostics import tokenize
from lexis.evaluation.metrics import recall_at_k, reciprocal_rank

DIFFERENT_DOCUMENT = "different_document"
HIGH_OVERLAP = "same_doc_high_answer_overlap"
LOW_OVERLAP = "same_doc_low_answer_overlap"


def answer_token_coverage(answer_texts: Iterable[str], chunk_text: str) -> float:
    """Fraction of the answer's distinct tokens (bm25s tokenization, the
    project's existing convention) that also appear in the chunk."""
    answer_tokens = set()
    for text in answer_texts:
        answer_tokens.update(tokenize(text))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & set(tokenize(chunk_text))) / len(answer_tokens)


def categorize_blocker(top_doc_id: Optional[str], relevant_doc_ids: Iterable[str],
                       coverage: float, overlap_threshold: float) -> str:
    if top_doc_id not in set(relevant_doc_ids):
        return DIFFERENT_DOCUMENT
    return HIGH_OVERLAP if coverage >= overlap_threshold else LOW_OVERLAP


@dataclass(frozen=True)
class CaseOracleAnalysis:
    recall: float
    reciprocal_rank: float
    first_relevant_rank: Optional[int]
    top1_chunk_id: Optional[str]
    top1_is_relevant: bool
    blocker: Optional[str]              # set only when the top-1 chunk is not relevant
    top1_answer_coverage: Optional[float]


def analyze_ranked_case(
    ranked: Sequence[Mapping],           # ordered: {"id", "doc_id", "text"}
    relevant_chunk_ids: Iterable[str],
    relevant_doc_ids: Iterable[str],
    answer_texts: Iterable[str],
    overlap_threshold: float,
    k: int,
) -> CaseOracleAnalysis:
    relevant = list(relevant_chunk_ids)
    ids = [r["id"] for r in ranked]
    first_rank = next((i + 1 for i, cid in enumerate(ids) if cid in relevant), None)
    recall = recall_at_k(relevant, ids, k=k)
    rr = reciprocal_rank(relevant, ids)
    if not ranked:
        return CaseOracleAnalysis(recall, rr, first_rank, None, False, None, None)

    top = ranked[0]
    if top["id"] in relevant:
        return CaseOracleAnalysis(recall, rr, first_rank, top["id"], True, None, None)
    coverage = answer_token_coverage(answer_texts, top.get("text", ""))
    blocker = categorize_blocker(top.get("doc_id"), relevant_doc_ids, coverage, overlap_threshold)
    return CaseOracleAnalysis(recall, rr, first_rank, top["id"], False, blocker, coverage)


def summarize_blockers(analyses: Iterable[CaseOracleAnalysis]) -> Dict[str, int]:
    """Counts over cases whose top-1 chunk is not relevant, plus totals."""
    analyses = list(analyses)
    counts = Counter(a.blocker for a in analyses if a.blocker)
    return {
        "cases": len(analyses),
        "top1_relevant": sum(1 for a in analyses if a.top1_is_relevant),
        "top1_not_relevant": sum(1 for a in analyses if not a.top1_is_relevant),
        DIFFERENT_DOCUMENT: counts.get(DIFFERENT_DOCUMENT, 0),
        HIGH_OVERLAP: counts.get(HIGH_OVERLAP, 0),
        LOW_OVERLAP: counts.get(LOW_OVERLAP, 0),
        "no_relevant_in_list": sum(1 for a in analyses if a.first_relevant_rank is None),
    }


@dataclass(frozen=True)
class CallOutcome:
    result: object
    attempts: int
    healthy: bool


async def call_until_healthy(make_call, is_healthy, max_retries: int) -> CallOutcome:
    """Run an async call; if its result is unhealthy (e.g. a retrieval path
    silently timed out and returned nothing) retry up to max_retries more
    times. Measurement runs use this so a transient timeout cannot silently
    turn a hybrid result into a single-path one; the caller records how many
    attempts were needed and whether the final result was still unhealthy."""
    attempts = 0
    result = None
    for _ in range(max_retries + 1):
        attempts += 1
        result = await make_call()
        if is_healthy(result):
            return CallOutcome(result, attempts, True)
    return CallOutcome(result, attempts, False)


def best_of(analyses: Sequence[CaseOracleAnalysis]) -> CaseOracleAnalysis:
    """Upper bound over several queries for the same case: the analysis with
    the highest reciprocal rank (ties: higher recall, then earliest given)."""
    if not analyses:
        raise ValueError("best_of needs at least one analysis")
    return max(analyses, key=lambda a: (a.reciprocal_rank, a.recall))
