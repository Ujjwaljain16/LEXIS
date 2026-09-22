"""
Generic, document-agnostic core for the offline query-representation
diagnostic (Step 11): `oracle_query_diagnostic`.

Purpose: hold the retrieval mechanism (dense/BM25/RRF, same indexed corpus,
same embedding model, same collection, same distance metric, same RRF
implementation, same top_k/depth) completely fixed, and vary only the QUERY
TEXT for a given target chunk, to measure how much query representation
affects retrievability. This is an ORACLE diagnostic: one representation is
built from ground-truth answer text, which must never reach production
query handling.

This module has no knowledge of CUAD, legal documents, answer-span
extraction, or any benchmark-specific concept -- it receives plain query
strings and chunk ids. Constructing the actual answer-derived query text
(CUAD's "answer_texts" metadata + chunk matching) is a benchmark-adapter
concern that lives in evaluation/dataset/mapping.py (generic) and
evaluation/run_oracle_query_diagnostic.py's CUAD-specific glue -- never
here.

Deliberately thin: everything about coverage/rank/Recall/MRR computation is
reused UNCHANGED from evaluation/depth_sensitivity.py (ChunkDepthProfile,
DepthCoverage, check_index_integrity, build_chunk_profile,
compute_depth_coverage, build_rank_distribution, set_rrf_hit_depths) --
this module adds only the two things that module does not already provide:
(1) constructing a representation-tagged, single-chunk-target diagnostic
BenchmarkCase, and (2) classifying per-chunk recovery between two
representations' RRF outcomes. This module never imports
lexis.config.settings and never calls RetrievalEngine.retrieve() (only its
diagnostic sibling retrieve_with_trace(), and only via the caller) --
verified by tests/unit/test_oracle_query_diagnostic.py.
"""
from dataclasses import dataclass

from lexis.evaluation.types import BenchmarkCase


def build_representation_case(
    base_case: BenchmarkCase,
    chunk_id: str,
    query_text: str,
    representation_label: str,
) -> BenchmarkCase:
    """Generic transform: a new diagnostic case targeting exactly ONE
    relevant chunk, with a caller-supplied query string and representation
    label. Has no knowledge of where query_text came from -- the existing
    benchmark query, an answer-derived string, or anything else -- that
    decision belongs entirely to the caller. relevant_chunk_ids is always a
    singleton {chunk_id}, so pooled coverage and case-averaged Recall/MRR
    computed over a set of these coincide by construction (each unit
    contributes exactly 0 or 1) -- this is intentional: see
    evaluation/run_oracle_query_diagnostic.py for how the "existing query"
    representation is additionally evaluated case-level (matching the
    production baseline's own aggregation) for comparison.

    case_id is intentionally kept IDENTICAL to base_case.case_id -- NOT
    suffixed by chunk_id -- so that (case_id, chunk_id) remains a stable
    join key for comparing two representations' outcomes on the exact same
    pair (e.g. depth_sensitivity.py::build_chunk_profile/
    set_rrf_hit_depths tag their output with case.case_id; a suffixed id
    would silently break that join). Uniqueness of the diagnostic unit
    itself is guaranteed by the (case_id, chunk_id) pair, since chunk_id is
    already distinct across one case's own relevant_chunk_ids."""
    return BenchmarkCase(
        case_id=base_case.case_id,
        query=query_text,
        relevant_chunk_ids=frozenset({chunk_id}),
        relevant_doc_ids=base_case.relevant_doc_ids,
        source_benchmark=base_case.source_benchmark,
        metadata={
            "representation": representation_label,
            "source_case_id": base_case.case_id,
            "chunk_id": chunk_id,
        },
    )


@dataclass
class RepresentationRecovery:
    """Per-(case, relevant_chunk) comparison between an ORIGINAL
    representation's RRF outcome and an ALTERNATE representation's RRF
    outcome, at one fixed comparison depth. "hit" means the chunk appeared
    in retrieve_with_trace()'s fused top-`depth` for that representation's
    own query. Derived properties give exactly the five observations Step
    11 asks for: "original misses" = not original_hit, "alternate
    retrieves" = alternate_hit, "both miss" = both_miss, "original
    succeeds" = original_hit, "alternate succeeds" = alternate_hit."""
    case_id: str
    chunk_id: str
    depth: int
    original_hit: bool
    alternate_hit: bool

    @property
    def both_miss(self) -> bool:
        return not self.original_hit and not self.alternate_hit

    @property
    def became_retrievable(self) -> bool:
        """Original representation missed; alternate representation hit --
        this specific miss is representation-sensitive at this depth. This
        is a measurement of sensitivity to query representation, not a
        claim about why the original query missed."""
        return (not self.original_hit) and self.alternate_hit

    @property
    def remains_missing(self) -> bool:
        """Both representations missed -- the miss persists even with an
        answer-like query, at this depth."""
        return self.both_miss

    @property
    def regressed(self) -> bool:
        """Original representation hit; alternate representation missed --
        this specific hit is NOT preserved under the alternate
        representation, at this depth. Used by Step 12's error analysis to
        report regressions, not just gains."""
        return self.original_hit and not self.alternate_hit


def classify_recovery(case_id: str, chunk_id: str, depth: int, original_hit: bool, alternate_hit: bool) -> RepresentationRecovery:
    return RepresentationRecovery(
        case_id=case_id, chunk_id=chunk_id, depth=depth,
        original_hit=original_hit, alternate_hit=alternate_hit,
    )
