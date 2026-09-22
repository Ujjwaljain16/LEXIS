"""
Generic, document-agnostic diagnostics for the "hard miss" set identified in
Step 9's depth-sensitivity experiment: relevant chunks absent from BOTH the
dense and BM25 retrieval paths even at the largest diagnostic depth tested.

This module answers "what does the evidence show about these specific
misses", never "why do they happen". It has no knowledge of CUAD, legal
documents, clauses, or any other benchmark-specific concept -- it operates
purely on a query string, a chunk's content, an optional list of matched
ground-truth answer texts (explicitly labeled and never used as a retrieval
feature), and optional pre-fetched dense/BM25 evidence.

Everything here is a pure function/dataclass -- no embedding, no
vector-store I/O, no BM25 indexing. All of that I/O lives in
evaluation/run_hard_miss_diagnostics.py, which fills in DenseEvidence/
BM25Evidence and calls build_hard_miss_record(). This split is what makes
the module fully unit-testable without live infrastructure.

Tokenization reuses bm25s's own tokenize() with the exact arguments
(stopwords="en", default lower=True and token_pattern) already used to
build and query the production BM25 index (see indexing/bm25_index.py) --
not a new convention invented for this diagnostic, and not a
domain-specific stopword list.
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import bm25s

from lexis.evaluation.depth_sensitivity import IndexIntegrityStatus


def tokenize(text: str) -> List[str]:
    """Token list for one string, via the exact tokenization bm25s applies
    during indexing/search (indexing/bm25_index.py: stopwords="en", default
    lower=True/token_pattern). Returns [] for empty/whitespace text."""
    if not text or not text.strip():
        return []
    return bm25s.tokenize([text], stopwords="en", return_ids=False, show_progress=False)[0]


def _normalize_whitespace(text: str) -> str:
    """Same whitespace normalization as
    evaluation/dataset/mapping.py::_normalize_whitespace, reused here (not
    reimplemented independently) so an answer span's reported length
    matches what the ground-truth mapping actually matched against."""
    return re.sub(r"\s+", " ", text).strip()


# --- 1. Query/chunk lexical relationship -----------------------------------

@dataclass
class LexicalDiagnostic:
    query_token_count: int
    chunk_token_count: int
    shared_token_count: int
    query_token_coverage: Optional[float]   # shared / distinct query tokens; None if query has no tokens
    chunk_token_coverage: Optional[float]   # shared / distinct chunk tokens; None if chunk has no tokens
    overlap_ratio_jaccard: Optional[float]  # shared / union of distinct tokens; None if both are empty


def compute_lexical_diagnostic(query: str, chunk_content: str) -> LexicalDiagnostic:
    query_tokens = tokenize(query)
    chunk_tokens = tokenize(chunk_content)
    query_set, chunk_set = set(query_tokens), set(chunk_tokens)
    shared = query_set & chunk_set
    union = query_set | chunk_set
    return LexicalDiagnostic(
        query_token_count=len(query_tokens),
        chunk_token_count=len(chunk_tokens),
        shared_token_count=len(shared),
        query_token_coverage=(len(shared) / len(query_set)) if query_set else None,
        chunk_token_coverage=(len(shared) / len(chunk_set)) if chunk_set else None,
        overlap_ratio_jaccard=(len(shared) / len(union)) if union else None,
    )


# --- 2. Chunk characteristics ------------------------------------------------

@dataclass
class ChunkCharacteristics:
    chunk_char_length: int
    chunk_token_count: int
    matched_answer_texts: List[str]           # ground truth, explicitly separate from the query
    answer_span_char_length: Optional[int]    # max whitespace-normalized length among matched_answer_texts
    answer_share_of_chunk_chars: Optional[float]


def compute_chunk_characteristics(chunk_content: str, matched_answer_texts: List[str]) -> ChunkCharacteristics:
    chunk_chars = len(chunk_content)
    chunk_tokens = len(tokenize(chunk_content))
    normalized_answers = [_normalize_whitespace(a) for a in matched_answer_texts if a and a.strip()]
    answer_span_char_length = max((len(a) for a in normalized_answers), default=None)
    answer_share = (
        answer_span_char_length / chunk_chars
        if (answer_span_char_length is not None and chunk_chars > 0)
        else None
    )
    return ChunkCharacteristics(
        chunk_char_length=chunk_chars,
        chunk_token_count=chunk_tokens,
        matched_answer_texts=list(matched_answer_texts),
        answer_span_char_length=answer_span_char_length,
        answer_share_of_chunk_chars=answer_share,
    )


# --- 3 & 4. Dense-side / BM25-side evidence ---------------------------------
# These carry pre-computed evidence gathered by the caller (which has access
# to the live vector store / BM25 index); this module never fetches them
# itself. A None field means "not computed" -- never fabricated as 0.

class VectorStatus(str, Enum):
    FOUND = "found"
    MISSING_FROM_VECTOR_STORE = "missing_from_vector_store"
    LOOKUP_UNAVAILABLE = "lookup_unavailable"  # a per-chunk fallback lookup could not be performed (see caller)


@dataclass
class DenseEvidence:
    vector_status: str
    query_chunk_similarity: Optional[float]  # cosine similarity of normalized embeddings; None if vector missing
    population_size: Optional[int]           # number of indexed points the exact rank was computed over
    exact_rank: Optional[int]                # 1-indexed rank among population_size by the same similarity; None if unavailable
    rank_unavailable_reason: Optional[str]   # e.g. "population exceeds configured max for a full scan"


class BM25TextStatus(str, Enum):
    FOUND = "found"
    MISSING_FROM_BM25_CORPUS = "missing_from_bm25_corpus"


@dataclass
class BM25Evidence:
    text_status: str
    score: Optional[float]                 # bm25s score for this query/chunk pair; None if text missing from corpus
    corpus_size: Optional[int]              # number of documents the exact rank was computed over
    exact_rank: Optional[int]               # 1-indexed rank among corpus_size by that score; None if unavailable
    rank_unavailable_reason: Optional[str]


# --- Assembly ----------------------------------------------------------------

@dataclass
class HardMissRecord:
    case_id: str
    chunk_id: str
    doc_id: Optional[str]
    query: str
    index_integrity: str
    lexical: LexicalDiagnostic
    chunk_characteristics: ChunkCharacteristics
    dense_evidence: Optional[DenseEvidence]
    bm25_evidence: Optional[BM25Evidence]


def build_hard_miss_record(
    case_id: str,
    chunk_id: str,
    doc_id: Optional[str],
    query: str,
    chunk_content: str,
    matched_answer_texts: List[str],
    index_integrity: str,
    dense_evidence: Optional[DenseEvidence],
    bm25_evidence: Optional[BM25Evidence],
) -> HardMissRecord:
    return HardMissRecord(
        case_id=case_id,
        chunk_id=chunk_id,
        doc_id=doc_id,
        query=query,
        index_integrity=index_integrity,
        lexical=compute_lexical_diagnostic(query, chunk_content),
        chunk_characteristics=compute_chunk_characteristics(chunk_content, matched_answer_texts),
        dense_evidence=dense_evidence,
        bm25_evidence=bm25_evidence,
    )


def _distribution_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Continuous summary stats (count/min/max/mean/median) rather than an
    arbitrary bucketing scheme -- see Step 10's "prefer continuous
    measurements over arbitrary buckets" constraint."""
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    median = sorted_vals[mid] if n % 2 == 1 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return {
        "count": n,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": sum(sorted_vals) / n,
        "median": median,
    }


@dataclass
class HardMissReport:
    benchmark: str
    run_config: dict
    num_hard_misses: int
    records: List[HardMissRecord]
    lexical_overlap_ratio_distribution: Dict[str, Optional[float]]
    chunk_char_length_distribution: Dict[str, Optional[float]]
    dense_similarity_distribution: Dict[str, Optional[float]]
    dense_rank_distribution: Dict[str, Optional[float]]
    bm25_score_distribution: Dict[str, Optional[float]]
    bm25_rank_distribution: Dict[str, Optional[float]]
    num_with_nonzero_lexical_overlap: int
    num_with_valid_dense_vector: int
    num_dense_evidence_available: int
    num_with_nonzero_bm25_score: int
    num_bm25_evidence_available: int
    num_index_integrity_anomalies: int


def build_hard_miss_report(records: List[HardMissRecord], benchmark: str, run_config: dict) -> HardMissReport:
    overlap_values = [r.lexical.overlap_ratio_jaccard for r in records if r.lexical.overlap_ratio_jaccard is not None]
    length_values = [float(r.chunk_characteristics.chunk_char_length) for r in records]

    dense_available = [r.dense_evidence for r in records if r.dense_evidence is not None]
    similarity_values = [d.query_chunk_similarity for d in dense_available if d.query_chunk_similarity is not None]
    dense_rank_values = [float(d.exact_rank) for d in dense_available if d.exact_rank is not None]
    num_valid_vector = sum(1 for d in dense_available if d.vector_status == VectorStatus.FOUND.value)

    bm25_available = [r.bm25_evidence for r in records if r.bm25_evidence is not None]
    bm25_score_values = [b.score for b in bm25_available if b.score is not None]
    bm25_rank_values = [float(b.exact_rank) for b in bm25_available if b.exact_rank is not None]
    num_nonzero_bm25 = sum(1 for b in bm25_available if b.score is not None and b.score > 0)

    num_integrity_anomalies = sum(
        1 for r in records if r.index_integrity != IndexIntegrityStatus.INDEXED_NORMALLY.value
    )
    num_nonzero_overlap = sum(1 for r in records if r.lexical.shared_token_count > 0)

    return HardMissReport(
        benchmark=benchmark,
        run_config=run_config,
        num_hard_misses=len(records),
        records=records,
        lexical_overlap_ratio_distribution=_distribution_stats(overlap_values),
        chunk_char_length_distribution=_distribution_stats(length_values),
        dense_similarity_distribution=_distribution_stats(similarity_values),
        dense_rank_distribution=_distribution_stats(dense_rank_values),
        bm25_score_distribution=_distribution_stats(bm25_score_values),
        bm25_rank_distribution=_distribution_stats(bm25_rank_values),
        num_with_nonzero_lexical_overlap=num_nonzero_overlap,
        num_with_valid_dense_vector=num_valid_vector,
        num_dense_evidence_available=len(dense_available),
        num_with_nonzero_bm25_score=num_nonzero_bm25,
        num_bm25_evidence_available=len(bm25_available),
        num_index_integrity_anomalies=num_integrity_anomalies,
    )


def format_hard_miss_summary(report: HardMissReport) -> str:
    """Concise human-readable summary. Reports measurements only -- deciding
    which observations are evidence-supported vs. which remain hypotheses
    is left to whoever reads the numbers, not baked in here as a canned
    interpretation that could stop matching the actual data."""
    lines = [
        f"=== Hard-Miss Diagnostic Summary ({report.benchmark}) ===",
        f"Hard misses analyzed: {report.num_hard_misses}",
        "",
        f"Index integrity anomalies (not indexed_normally): {report.num_index_integrity_anomalies} / {report.num_hard_misses}",
        "",
        f"Non-zero lexical overlap with query: {report.num_with_nonzero_lexical_overlap} / {report.num_hard_misses}",
        f"Lexical overlap ratio (Jaccard) distribution: {report.lexical_overlap_ratio_distribution}",
        f"Chunk character-length distribution: {report.chunk_char_length_distribution}",
        "",
        f"Valid indexed dense vector found: {report.num_with_valid_dense_vector} / {report.num_dense_evidence_available} "
        f"(dense evidence available for {report.num_dense_evidence_available} / {report.num_hard_misses})",
        f"Dense query-chunk cosine similarity distribution: {report.dense_similarity_distribution}",
        f"Dense exact-rank distribution (among evaluated population): {report.dense_rank_distribution}",
        "",
        f"Non-zero BM25 score: {report.num_with_nonzero_bm25_score} / {report.num_bm25_evidence_available} "
        f"(BM25 evidence available for {report.num_bm25_evidence_available} / {report.num_hard_misses})",
        f"BM25 score distribution: {report.bm25_score_distribution}",
        f"BM25 exact-rank distribution (among evaluated corpus): {report.bm25_rank_distribution}",
    ]
    return "\n".join(lines)
