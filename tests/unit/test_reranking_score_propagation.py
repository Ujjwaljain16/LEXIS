"""
Regression tests for the reranking-score propagation bug discovered during
the Step 4 E2E trace.

Root cause: generation/context_assembler.py::rerank_only() (the reranker
actually invoked by the live /query/fast handler) writes the CrossEncoder
score under the key "_relevance_score" -- the same key it uses internally
to sort chunks descending before returning them. serving/routes/query.py
then built each Candidate with `score=c.get("cross_encoder_score", 0.0)`, a
key that ContextAssembler never produces, so every live Candidate.score was
silently 0.0 regardless of the real reranking result.

Canonical field chosen: "_relevance_score" -- it is what the producer
(ContextAssembler) actually writes and already uses as its own source of
truth for ranking. No second/compatibility field was introduced.

Missing-score fallback: ContextAssembler's own sort already treats a
missing "_relevance_score" as -999.0 (used when its CrossEncoder failed to
load and no scoring happened at all). query.py's Candidate construction now
uses that same sentinel instead of 0.0, so an absent score cannot be
confused with a real (possibly negative, since CrossEncoder outputs raw
logits, not a bounded [0,1] probability) low relevance score.

These tests use a fake CrossEncoder model (no real model load) plugged
into the REAL ContextAssembler, and drive the REAL /query/fast route
handler, so the propagation is proven through the actual runtime path, not
just re-asserted against a stub of query.py's own logic. Content is
generic (a report, a memo, a policy) to keep this domain-agnostic.
"""
from types import SimpleNamespace

import pytest

from lexis.config import settings
from lexis.serving.routes import query as query_module
from lexis.serving.models import DeepModeEnqueueRequest
from lexis.generation import synthesizer as synthesizer_module
from lexis.generation import context_assembler as context_assembler_module
from lexis.generation.synthesizer import LexisSynthesizer
from lexis.generation.context_assembler import ContextAssembler


class FakeStreamChunk:
    def __init__(self, content):
        self.choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]


class FakeStream:
    def __init__(self, tokens):
        self._tokens = tokens

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for t in self._tokens:
            yield FakeStreamChunk(t)


async def noop_acompletion(**kwargs):
    return FakeStream(["ok"])


class FakeCrossEncoderModel:
    """Stands in for sentence_transformers.CrossEncoder -- deterministic,
    no real model download/load. Looks scores up by the exact chunk text."""

    def __init__(self, score_by_text: dict):
        self._score_by_text = score_by_text

    def predict(self, pairs):
        return [self._score_by_text[text] for _, text in pairs]


class RecordingSentenceExpander:
    """Stands in for the live sentence_expander -- records the Candidate
    objects query.py actually built, then passes them through unchanged, so
    the test can inspect Candidate.score without needing it to survive any
    further (it currently does not survive into final_chunks/synthesis --
    see the final report)."""

    def __init__(self):
        self.captured_candidates = None

    async def transform(self, query, candidates):
        self.captured_candidates = candidates
        return candidates


class FakeRetrievalEngine:
    def __init__(self, chunks):
        self._chunks = chunks

    async def retrieve(self, query_text, top_k_per_path=5, top_n_rrf=15):
        return self._chunks


def make_chunk(chunk_id: str, doc_id: str, text: str) -> dict:
    return {
        "id": chunk_id,
        "score": 1.0,
        "rrf_score": 1.0,  # high enough that CRAG's web-fallback branch never triggers
        "source_path": "path_b_global",
        "payload": {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "content": text,
            "source_file": f"{doc_id}.txt",
        },
        "text": text,
    }


def make_real_context_assembler(monkeypatch, score_by_text: dict) -> ContextAssembler:
    monkeypatch.setattr(context_assembler_module, "CrossEncoder", lambda model_name: FakeCrossEncoderModel(score_by_text))
    return ContextAssembler()


async def drive_query_fast(monkeypatch, engine, assembler, expander):
    monkeypatch.setattr(synthesizer_module, "acompletion", noop_acompletion)
    monkeypatch.setattr(query_module, "sentence_expander", expander)
    monkeypatch.setattr(query_module, "get_fast_components", lambda: (engine, assembler, LexisSynthesizer()))

    req = DeepModeEnqueueRequest(query="irrelevant query text", metadata_filters={})
    response = await query_module.query_fast(req, request=None)
    async for _ in response.body_iterator:
        pass  # drain the SSE stream to completion


# --- 1 & 2. Score propagation, multiple candidates, no cross-candidate mixup ---

@pytest.mark.asyncio
async def test_distinct_crossencoder_scores_propagate_to_the_correct_candidates(monkeypatch):
    text_a = "A passage about renewal timelines from a generic report."
    text_b = "A passage about staffing levels from an unrelated memo."
    text_c = "A passage about routing procedures from a policy document."

    score_by_text = {text_a: 0.7319, text_b: 0.1842, text_c: 0.9427}
    chunks = [
        make_chunk("pqac-a", "doc-a", text_a),
        make_chunk("pqac-b", "doc-b", text_b),
        make_chunk("pqac-c", "doc-c", text_c),
    ]

    engine = FakeRetrievalEngine(chunks)
    assembler = make_real_context_assembler(monkeypatch, score_by_text)
    expander = RecordingSentenceExpander()

    await drive_query_fast(monkeypatch, engine, assembler, expander)

    candidates = expander.captured_candidates
    assert candidates is not None
    assert len(candidates) == 3

    score_by_content = {c.content: c.score for c in candidates}
    assert score_by_content[text_a] == pytest.approx(0.7319)
    assert score_by_content[text_b] == pytest.approx(0.1842)
    assert score_by_content[text_c] == pytest.approx(0.9427)

    # No candidate silently defaulted to 0.0 (the original bug).
    assert all(c.score != 0.0 for c in candidates)


# --- 4. Ranking semantics preserved (real rerank_only sort, not reinvented here) ---

@pytest.mark.asyncio
async def test_candidate_order_reflects_the_real_reranking_order(monkeypatch):
    text_high = "Highest relevance passage."
    text_mid = "Medium relevance passage."
    text_low = "Lowest relevance passage."

    score_by_text = {text_high: 0.9427, text_mid: 0.7319, text_low: 0.1842}
    # Deliberately fed in an order that does NOT match score order, so a
    # passing test proves rerank_only's own sort -- not input order -- won.
    chunks = [
        make_chunk("pqac-mid", "doc-mid", text_mid),
        make_chunk("pqac-low", "doc-low", text_low),
        make_chunk("pqac-high", "doc-high", text_high),
    ]

    engine = FakeRetrievalEngine(chunks)
    assembler = make_real_context_assembler(monkeypatch, score_by_text)
    expander = RecordingSentenceExpander()

    await drive_query_fast(monkeypatch, engine, assembler, expander)

    candidates = expander.captured_candidates
    assert [c.content for c in candidates] == [text_high, text_mid, text_low]
    assert [c.score for c in candidates] == [0.9427, 0.7319, 0.1842]


# --- 3. Missing-score behaviour (ContextAssembler's own existing contract) ---

@pytest.mark.asyncio
async def test_missing_score_uses_the_established_sentinel_not_a_manufactured_value(monkeypatch):
    """When ContextAssembler's reranker is unavailable, rerank_only's own
    documented fallback returns chunks with NO _relevance_score key at all
    (see generation/context_assembler.py::rerank_only). This must not be
    disguised as a real score of 0.0 -- it should carry the same "no score"
    sentinel (-999.0) ContextAssembler itself already uses for this exact
    condition in its own sort fallback."""
    text = "A passage from a document with no reranker available."
    chunks = [make_chunk("pqac-no-rerank", "doc-x", text)]

    engine = FakeRetrievalEngine(chunks)
    assembler = ContextAssembler.__new__(ContextAssembler)
    assembler.reranker = None  # simulates a failed model load, per rerank_only's own guard
    assembler.max_tokens = 6000
    import tiktoken
    assembler.tokenizer = tiktoken.get_encoding("cl100k_base")

    expander = RecordingSentenceExpander()
    await drive_query_fast(monkeypatch, engine, assembler, expander)

    candidates = expander.captured_candidates
    assert len(candidates) == 1
    assert candidates[0].score == -999.0
    assert candidates[0].score != 0.0
    assert candidates[0].content == text


# --- Producer-side contract, independent of query.py ---

def test_rerank_only_writes_relevance_score_key_directly(monkeypatch):
    """Documents the actual producer contract this fix depends on: real
    ContextAssembler.rerank_only() writes "_relevance_score", never
    "cross_encoder_score"."""
    text = "Some generic passage."
    assembler = make_real_context_assembler(monkeypatch, {text: 0.4242})
    chunks = [{"text": text}]

    result = assembler.rerank_only("q", chunks, top_k=5)

    assert result[0]["_relevance_score"] == pytest.approx(0.4242)
    assert "cross_encoder_score" not in result[0]
