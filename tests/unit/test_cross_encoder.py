"""
Tests for reranking/cross_encoder.py::BAAICrossEncoder.

This class existed but was never imported/used anywhere in the codebase
(R4 of the retrieval-improvement ladder was unwired) -- these tests cover
it directly with an injected fake model (no real ~600MB+ model download).
"""
import pytest

from lexis.reranking.cross_encoder import BAAICrossEncoder
from lexis.retrieval.interfaces import Candidate, Query


class FakeModel:
    """Scores each (query, doc) pair by how many words they share --
    deterministic and cheap, no real model needed."""
    def predict(self, pairs):
        scores = []
        for query, doc in pairs:
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            scores.append(float(len(q_words & d_words)))
        return scores


def make_candidate(chunk_id, content, score=0.0, source_path="path_b_global"):
    return Candidate(chunk_id=chunk_id, score=score, source_path=source_path, content=content)


@pytest.mark.asyncio
async def test_empty_candidates_returns_empty():
    reranker = BAAICrossEncoder(model=FakeModel())
    result = await reranker.transform(Query(text="q"), [])
    assert result == []


@pytest.mark.asyncio
async def test_reorders_candidates_by_the_model_score():
    reranker = BAAICrossEncoder(model=FakeModel())
    candidates = [
        make_candidate("low", "irrelevant text about weather"),
        make_candidate("high", "termination notice period clause"),
    ]

    result = await reranker.transform(Query(text="termination notice period"), candidates)

    assert [c.chunk_id for c in result] == ["high", "low"]


@pytest.mark.asyncio
async def test_updates_score_to_the_new_model_score():
    reranker = BAAICrossEncoder(model=FakeModel())
    candidates = [make_candidate("c1", "termination clause", score=0.001)]

    result = await reranker.transform(Query(text="termination"), candidates)

    assert result[0].score == 1.0  # one shared word ("termination"), not the original 0.001


@pytest.mark.asyncio
async def test_does_not_mutate_the_input_list_or_its_candidate_objects():
    """Regression: the input may be a slice sharing Candidate object
    references with a caller's own list kept for diagnostics (e.g.
    RetrievalTrace.fused_candidates) -- mutating scores/order in place
    would silently corrupt that already-returned state."""
    original = [
        make_candidate("a", "termination clause", score=0.5),
        make_candidate("b", "unrelated text", score=0.9),
    ]
    original_scores = [c.score for c in original]
    original_order = [c.chunk_id for c in original]

    reranker = BAAICrossEncoder(model=FakeModel())
    result = await reranker.transform(Query(text="termination"), original)

    assert [c.chunk_id for c in original] == original_order
    assert [c.score for c in original] == original_scores
    assert result is not original
