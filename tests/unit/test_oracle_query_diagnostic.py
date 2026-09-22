"""
Tests for the generic offline query-representation diagnostic core
(evaluation/oracle_query_diagnostic.py). Content is generic (reports,
memos), never CUAD/legal-specific. No Qdrant/BM25/embedding model/network
is used -- retrieval-stage data is synthetic, exactly like
test_depth_sensitivity.py.
"""
import inspect
from dataclasses import dataclass
from typing import List

import pytest

from lexis.evaluation.depth_sensitivity import build_chunk_profile, compute_depth_coverage
from lexis.evaluation.oracle_query_diagnostic import (
    build_representation_case,
    classify_recovery,
)
from lexis.evaluation.types import BenchmarkCase
from lexis.retrieval.interfaces import Candidate


def make_candidate(chunk_id: str, doc_id: str = "doc-x") -> Candidate:
    return Candidate(chunk_id=chunk_id, score=1.0, source_path="test", metadata={"doc_id": doc_id}, content="text")


class FakeChunkWithId:
    def __init__(self, chunk_id, raw_content):
        self.chunk_id = chunk_id
        self.raw_content = raw_content


@dataclass
class FakeTrace:
    dense_candidates: List[Candidate]
    bm25_candidates: List[Candidate]
    fused_candidates: List[Candidate]
    final_chunks: List[dict]


def make_base_case(case_id="case-1", relevant_chunk_ids=("chunk-a", "chunk-b"), doc_ids=("doc-x",)):
    return BenchmarkCase(
        case_id=case_id,
        query="what is the notice period",
        relevant_chunk_ids=frozenset(relevant_chunk_ids),
        relevant_doc_ids=frozenset(doc_ids),
        source_benchmark="synthetic",
        metadata={"some_adapter_field": "opaque"},
    )


# --- build_representation_case ---

def test_existing_and_alternate_representations_remain_distinct_inputs():
    base = make_base_case()
    existing = build_representation_case(base, "chunk-a", "what is the notice period", "existing_query")
    alternate = build_representation_case(base, "chunk-a", "either party may terminate with thirty days notice", "answer_derived")

    assert existing.query != alternate.query
    assert existing.query == "what is the notice period"
    assert alternate.query == "either party may terminate with thirty days notice"
    # both target the SAME single chunk, only the query text differs
    assert existing.relevant_chunk_ids == alternate.relevant_chunk_ids == frozenset({"chunk-a"})


def test_representation_case_targets_a_single_chunk_even_when_base_case_has_many():
    base = make_base_case(relevant_chunk_ids=("chunk-a", "chunk-b", "chunk-c"))
    case_for_b = build_representation_case(base, "chunk-b", "answer text for b", "answer_derived")
    assert case_for_b.relevant_chunk_ids == frozenset({"chunk-b"})
    assert "chunk-a" not in case_for_b.relevant_chunk_ids
    assert "chunk-c" not in case_for_b.relevant_chunk_ids


def test_representation_case_preserves_doc_ids_and_benchmark_label():
    base = make_base_case(doc_ids=("doc-x", "doc-y"))
    case = build_representation_case(base, "chunk-a", "q", "existing_query")
    assert case.relevant_doc_ids == frozenset({"doc-x", "doc-y"})
    assert case.source_benchmark == "synthetic"


def test_representation_case_metadata_tags_representation_and_source():
    base = make_base_case(case_id="case-42")
    case = build_representation_case(base, "chunk-a", "answer text", "answer_derived")
    assert case.metadata["representation"] == "answer_derived"
    assert case.metadata["source_case_id"] == "case-42"
    assert case.metadata["chunk_id"] == "chunk-a"


def test_representation_case_construction_is_deterministic():
    base = make_base_case()
    a = build_representation_case(base, "chunk-a", "answer text", "answer_derived")
    b = build_representation_case(base, "chunk-a", "answer text", "answer_derived")
    assert a == b  # frozen dataclass equality


def test_representation_case_id_stays_equal_to_source_case_for_stable_joining():
    """case_id is intentionally NOT suffixed by chunk_id -- (case_id,
    chunk_id) is the stable join key run_oracle_query_diagnostic.py uses to
    compare an existing_query profile against an answer_derived profile for
    the same pair. A suffixed case_id would silently break that join (this
    is a regression test for exactly that bug)."""
    base = make_base_case(case_id="case-42", relevant_chunk_ids=("chunk-a", "chunk-b"))
    case_a = build_representation_case(base, "chunk-a", "qa", "answer_derived")
    case_b = build_representation_case(base, "chunk-b", "qb", "answer_derived")
    assert case_a.case_id == case_b.case_id == "case-42"
    assert case_a != case_b  # still distinct values: different target chunk and query
    assert (case_a.case_id, next(iter(case_a.relevant_chunk_ids))) != (case_b.case_id, next(iter(case_b.relevant_chunk_ids)))


# --- classify_recovery / RepresentationRecovery ---

def test_recovery_became_retrievable_when_original_misses_and_alternate_hits():
    r = classify_recovery("c1", "chunk-a", depth=30, original_hit=False, alternate_hit=True)
    assert r.became_retrievable is True
    assert r.both_miss is False
    assert r.remains_missing is False


def test_recovery_remains_missing_when_both_miss():
    r = classify_recovery("c1", "chunk-a", depth=30, original_hit=False, alternate_hit=False)
    assert r.both_miss is True
    assert r.remains_missing is True
    assert r.became_retrievable is False


def test_recovery_not_became_retrievable_when_original_already_hits():
    r = classify_recovery("c1", "chunk-a", depth=30, original_hit=True, alternate_hit=True)
    assert r.became_retrievable is False  # only counts as "recovered" if original MISSED
    assert r.both_miss is False


def test_recovery_original_hit_alternate_miss_is_not_both_miss_or_recovered():
    r = classify_recovery("c1", "chunk-a", depth=30, original_hit=True, alternate_hit=False)
    assert r.both_miss is False
    assert r.became_retrievable is False
    assert r.regressed is True


def test_recovery_regressed_false_when_original_missed_too():
    r = classify_recovery("c1", "chunk-a", depth=30, original_hit=False, alternate_hit=False)
    assert r.regressed is False  # can't "regress" from a miss


def test_recovery_regressed_false_when_both_hit():
    r = classify_recovery("c1", "chunk-a", depth=30, original_hit=True, alternate_hit=True)
    assert r.regressed is False


# --- no dependency on production Settings ---

def test_module_never_imports_or_touches_settings():
    import lexis.evaluation.oracle_query_diagnostic as module
    source = inspect.getsource(module)
    assert "from lexis.config import settings" not in source
    assert "import settings" not in source


def test_module_has_no_retrieval_engine_or_io_dependency():
    """This module must only ever be composed with retrieve_with_trace()
    results BY THE CALLER -- it must not import RetrievalEngine or perform
    any retrieval/embedding/network I/O itself. Checked structurally (import
    statements), not via a docstring text match, since this module's own
    documentation legitimately mentions retrieve()/retrieve_with_trace() by
    name."""
    import lexis.evaluation.oracle_query_diagnostic as module
    source = inspect.getsource(module)
    import_lines = [line for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
    assert not any("hybrid_retriever" in line or "RetrievalEngine" in line for line in import_lines)


# --- end-to-end composition with depth_sensitivity.py: pooled coverage vs
# case-averaged Recall/MRR stay distinct aggregations, exactly as in Step 9 ---

def test_composes_with_depth_sensitivity_pooled_coverage_stays_separate_from_case_averaged():
    base = make_base_case(relevant_chunk_ids=("chunk-a", "chunk-b"))
    chunks_by_doc_id = {"doc-x": [FakeChunkWithId("chunk-a", "text a"), FakeChunkWithId("chunk-b", "text b")]}
    depths = [10, 30]

    existing_case = build_representation_case(base, "chunk-a", "original query text", "existing_query")
    alt_case = build_representation_case(base, "chunk-a", "answer derived text", "answer_derived")
    assert existing_case.query != alt_case.query

    # chunk-a hit via dense at rank 3 under BOTH representations' own traces (synthetic)
    trace = FakeTrace(dense_candidates=[make_candidate("chunk-a")], bm25_candidates=[], fused_candidates=[], final_chunks=[])
    profile = build_chunk_profile(existing_case, "chunk-a", trace, depths, chunks_by_doc_id)
    coverage = compute_depth_coverage(30, [profile], rrf_hit_pairs_at_depth=set(), per_case_metrics_at_depth=[(1.0, 1.0)])

    assert coverage.dense_coverage == 1.0          # pooled chunk-level coverage
    assert coverage.mean_recall_at_depth == 1.0     # case-averaged Recall@depth -- a separate aggregation
    # they happen to agree here only because there is exactly one unit; the
    # two numbers are computed by entirely different code paths (dense rank
    # membership vs evaluation/metrics.py::recall_at_k), not conflated.
