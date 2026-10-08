"""
Tests for the generic evaluation harness (evaluation/harness.py). Retrieval
is mocked throughout -- no Qdrant, Elasticsearch, embedding model, or LLM is
required. Benchmark cases are constructed directly (not via the CUAD
adapter) to prove the harness has no dependency on any specific benchmark.
"""
import pytest

from lexis.evaluation.harness import EvalHarness
from lexis.evaluation.types import BenchmarkCase


def make_case(case_id, query, relevant_chunk_ids):
    return BenchmarkCase(
        case_id=case_id,
        query=query,
        relevant_chunk_ids=frozenset(relevant_chunk_ids),
        relevant_doc_ids=frozenset({f"doc-for-{case_id}"}),
        source_benchmark="synthetic-test-benchmark",
    )


class ScriptedRetriever:
    """Returns a pre-programmed ranked result list per query, and records
    every call made to it."""

    def __init__(self, results_by_query: dict):
        self._results_by_query = results_by_query
        self.calls = []

    async def retrieve(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return self._results_by_query.get(query, [])[:top_k]


@pytest.mark.asyncio
async def test_harness_scores_recall_and_mrr_per_case_and_aggregates():
    retriever = ScriptedRetriever({
        "q1": ["relevant-1", "noise-a", "noise-b"],       # relevant at rank 1 -> recall 1.0, mrr 1.0
        "q2": ["noise-a", "noise-b", "relevant-2"],        # relevant at rank 3 -> recall 1.0, mrr 1/3
    })
    cases = [
        make_case("case-1", "q1", {"relevant-1"}),
        make_case("case-2", "q2", {"relevant-2"}),
    ]
    harness = EvalHarness(retrieve_fn=retriever.retrieve, top_k=30)

    report = await harness.run(cases, benchmark_name="synthetic-test-benchmark")

    assert report.num_cases_scored == 2
    assert report.num_cases_excluded == 0
    by_id = {c.case_id: c for c in report.per_case}
    assert by_id["case-1"].recall_at_k == 1.0
    assert by_id["case-1"].reciprocal_rank == 1.0
    assert by_id["case-2"].recall_at_k == 1.0
    assert by_id["case-2"].reciprocal_rank == pytest.approx(1 / 3)
    assert report.mean_recall_at_k == pytest.approx(1.0)
    assert report.mean_reciprocal_rank == pytest.approx((1.0 + 1 / 3) / 2)
    assert report.benchmark == "synthetic-test-benchmark"
    assert report.top_k == 30


@pytest.mark.asyncio
async def test_harness_scores_a_miss_correctly():
    retriever = ScriptedRetriever({"q1": ["noise-a", "noise-b", "noise-c"]})
    cases = [make_case("case-1", "q1", {"relevant-1"})]
    harness = EvalHarness(retrieve_fn=retriever.retrieve, top_k=30)

    report = await harness.run(cases)

    assert report.per_case[0].recall_at_k == 0.0
    assert report.per_case[0].reciprocal_rank == 0.0


@pytest.mark.asyncio
async def test_harness_respects_the_configured_top_k_not_a_hardcoded_one():
    retriever = ScriptedRetriever({"q1": [f"item-{i}" for i in range(10)] + ["relevant-1"]})
    cases = [make_case("case-1", "q1", {"relevant-1"})]

    harness_small = EvalHarness(retrieve_fn=retriever.retrieve, top_k=5)
    report_small = await harness_small.run(cases)
    assert report_small.per_case[0].recall_at_k == 0.0  # relevant item is at rank 11, past top_k=5

    harness_large = EvalHarness(retrieve_fn=retriever.retrieve, top_k=30)
    report_large = await harness_large.run(cases)
    assert report_large.per_case[0].recall_at_k == 1.0

    assert retriever.calls == [("q1", 5), ("q1", 30)]


@pytest.mark.asyncio
async def test_cases_without_chunk_level_ground_truth_are_excluded_not_scored_or_retrieved():
    retriever = ScriptedRetriever({"q1": ["a", "b"]})
    case_no_gt = make_case("case-no-gt", "q1", set())  # no chunk-level ground truth
    harness = EvalHarness(retrieve_fn=retriever.retrieve, top_k=30)

    report = await harness.run([case_no_gt])

    assert report.num_cases_scored == 0
    assert report.num_cases_excluded == 1
    assert report.excluded_case_ids == ["case-no-gt"]
    assert retriever.calls == []  # never even attempted retrieval for an unscoreable case
    assert report.mean_recall_at_k == 0.0
    assert report.mean_reciprocal_rank == 0.0


@pytest.mark.asyncio
async def test_empty_case_list_does_not_crash():
    retriever = ScriptedRetriever({})
    harness = EvalHarness(retrieve_fn=retriever.retrieve, top_k=30)
    report = await harness.run([])
    assert report.num_cases_scored == 0
    assert report.mean_recall_at_k == 0.0
    assert report.mean_reciprocal_rank == 0.0


def make_scoped_case(case_id, query, relevant_chunk_ids, relevant_doc_ids):
    return BenchmarkCase(
        case_id=case_id, query=query, relevant_chunk_ids=frozenset(relevant_chunk_ids),
        relevant_doc_ids=frozenset(relevant_doc_ids), source_benchmark="synthetic-test-benchmark",
    )


class ScriptedScopedRetriever:
    """Records the exact (query, top_k, doc_ids) it was called with, and
    returns results that depend on which documents were passed in -- so a
    test can prove the scope actually reached the retrieval call."""

    def __init__(self, results_by_query_and_scope: dict):
        self._results = results_by_query_and_scope
        self.calls = []

    async def retrieve(self, query, top_k, doc_ids):
        self.calls.append((query, top_k, doc_ids))
        return self._results.get((query, frozenset(doc_ids)), [])[:top_k]


# --- protocol validation ---

def test_pooled_protocol_requires_retrieve_fn():
    with pytest.raises(ValueError, match="pooled"):
        EvalHarness(protocol="pooled")


def test_doc_scoped_protocol_requires_scoped_retrieve_fn():
    with pytest.raises(ValueError, match="doc_scoped"):
        EvalHarness(protocol="doc_scoped")


def test_unknown_protocol_rejected():
    with pytest.raises(ValueError, match="unknown protocol"):
        EvalHarness(retrieve_fn=lambda q, k: None, protocol="not-a-real-protocol")


def test_default_protocol_is_pooled_for_backward_compatibility():
    assert EvalHarness(retrieve_fn=lambda q, k: None).protocol == "pooled"


# --- doc_scoped protocol behavior ---

@pytest.mark.asyncio
async def test_doc_scoped_passes_the_case_document_scope_to_the_retriever():
    retriever = ScriptedScopedRetriever({("q1", frozenset({"doc-a"})): ["relevant-1"]})
    cases = [make_scoped_case("case-1", "q1", {"relevant-1"}, {"doc-a"})]
    harness = EvalHarness(scoped_retrieve_fn=retriever.retrieve, top_k=30, protocol="doc_scoped")

    report = await harness.run(cases)

    assert report.protocol == "doc_scoped"
    assert report.per_case[0].recall_at_k == 1.0
    assert retriever.calls == [("q1", 30, frozenset({"doc-a"}))]


@pytest.mark.asyncio
async def test_doc_scoped_uses_only_that_cases_own_document_scope_not_others():
    retriever = ScriptedScopedRetriever({
        ("q", frozenset({"doc-a"})): ["relevant-1"],
        ("q", frozenset({"doc-b"})): [],  # correct chunk is not visible when scoped to the wrong document
    })
    cases = [
        make_scoped_case("case-a", "q", {"relevant-1"}, {"doc-a"}),
        make_scoped_case("case-b", "q", {"relevant-1"}, {"doc-b"}),
    ]
    harness = EvalHarness(scoped_retrieve_fn=retriever.retrieve, top_k=30, protocol="doc_scoped")

    report = await harness.run(cases)

    by_id = {c.case_id: c for c in report.per_case}
    assert by_id["case-a"].recall_at_k == 1.0
    assert by_id["case-b"].recall_at_k == 0.0


@pytest.mark.asyncio
async def test_doc_scoped_excludes_a_case_with_no_known_document_scope():
    retriever = ScriptedScopedRetriever({})
    case_no_scope = make_scoped_case("case-1", "q1", {"relevant-1"}, set())  # ground truth chunk but no doc scope
    harness = EvalHarness(scoped_retrieve_fn=retriever.retrieve, top_k=30, protocol="doc_scoped")

    report = await harness.run([case_no_scope])

    assert report.num_cases_scored == 0 and report.num_cases_excluded == 1
    assert report.excluded_case_ids == ["case-1"]
    assert retriever.calls == []


@pytest.mark.asyncio
async def test_doc_scoped_still_excludes_cases_with_no_chunk_level_ground_truth():
    retriever = ScriptedScopedRetriever({})
    case = make_scoped_case("case-1", "q1", set(), {"doc-a"})  # doc scope present, but no chunk ground truth
    harness = EvalHarness(scoped_retrieve_fn=retriever.retrieve, top_k=30, protocol="doc_scoped")

    report = await harness.run([case])

    assert report.num_cases_excluded == 1 and retriever.calls == []


@pytest.mark.asyncio
async def test_pooled_report_labels_itself_pooled():
    retriever = ScriptedRetriever({"q1": ["relevant-1"]})
    harness = EvalHarness(retrieve_fn=retriever.retrieve, top_k=30)
    report = await harness.run([make_case("case-1", "q1", {"relevant-1"})])
    assert report.protocol == "pooled"


@pytest.mark.asyncio
async def test_harness_has_no_cuad_specific_behavior():
    """Construct cases that look nothing like CUAD (a report and a memo,
    not a contract) to prove the harness has no benchmark-specific logic."""
    retriever = ScriptedRetriever({
        "What was the Q3 incident count?": ["chunk-report-1"],
        "Who approved the memo?": ["chunk-memo-9"],
    })
    cases = [
        make_case("report-case", "What was the Q3 incident count?", {"chunk-report-1"}),
        make_case("memo-case", "Who approved the memo?", {"chunk-memo-9"}),
    ]
    harness = EvalHarness(retrieve_fn=retriever.retrieve, top_k=30)
    report = await harness.run(cases, benchmark_name="a-different-benchmark")
    assert report.mean_recall_at_k == 1.0
    assert report.benchmark == "a-different-benchmark"
