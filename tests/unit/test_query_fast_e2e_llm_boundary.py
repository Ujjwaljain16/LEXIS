"""
Mocked end-to-end check: drives an actual /query/fast request through the
real route handler, real SentenceWindowExpansion, and the real
LexisSynthesizer.stream_answer(), to confirm the configured synthesis model
and API key genuinely reach the final litellm.acompletion() boundary -- not
just that stream_answer() does the right thing in isolation (covered by
test_synthesizer_api_key_config.py).

Retrieval and reranking are replaced with lightweight fakes so this test
needs no live Qdrant/Elasticsearch and never loads the real CrossEncoder
reranker or bge-m3 embedding model. Only the litellm call itself is mocked.

Content is a generic, non-legal passage to keep the synthesizer's document
handling untested-but-unconstrained by any particular document schema.
"""
from types import SimpleNamespace

import pytest

from lexis.config import settings
from lexis.serving.routes import query as query_module
from lexis.serving.models import DeepModeEnqueueRequest
from lexis.generation import synthesizer as synthesizer_module
from lexis.generation.synthesizer import LexisSynthesizer

SENTINEL_KEY = "sk-test-sentinel-e2e-not-a-real-credential"


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


class FakeRetrievalEngine:
    """Stands in for hybrid_retriever.RetrievalEngine -- returns one generic,
    non-legal candidate with a high enough rrf_score that CRAG's web-fallback
    branch is skipped (no network needed)."""

    async def retrieve(self, query_text, top_k_per_path=5, top_n_rrf=15):
        return [{
            "id": "pqac-e2e-generic-1",
            "score": 1.0,
            "rrf_score": 1.0,
            "source_path": "path_b_global",
            "payload": {
                "chunk_id": "pqac-e2e-generic-1",
                "doc_id": "generic-doc-e2e",
                "content": "An arbitrary passage from a generic report.",
                "source_file": "generic_report.txt",
                # deliberately no chunk_index -- exercises sentence_window's
                # graceful passthrough for candidates without positional metadata.
            },
            "text": "An arbitrary passage from a generic report.",
        }]


class FakeContextAssembler:
    """Stands in for generation.context_assembler.ContextAssembler -- avoids
    loading the real CrossEncoder model. Passes candidates through unchanged."""

    def rerank_only(self, query_text, candidates, top_k=5):
        return candidates[:top_k]


@pytest.mark.asyncio
async def test_fast_query_reaches_litellm_boundary_with_configured_model_and_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SENTINEL_KEY)

    captured_llm_call = {}

    async def fake_acompletion(**kwargs):
        captured_llm_call.update(kwargs)
        return FakeStream(["The report ", "says X."])

    monkeypatch.setattr(synthesizer_module, "acompletion", fake_acompletion)

    fake_engine = FakeRetrievalEngine()
    fake_assembler = FakeContextAssembler()
    real_synthesizer = LexisSynthesizer()  # exercise the real synthesis code path

    monkeypatch.setattr(
        query_module,
        "get_fast_components",
        lambda: (fake_engine, fake_assembler, real_synthesizer),
    )

    req = DeepModeEnqueueRequest(query="What does the report say?", metadata_filters={})
    response = await query_module.query_fast(req, request=None)

    events = []
    async for chunk in response.body_iterator:
        events.append(chunk)
    full_stream = "".join(events)

    # The LLM boundary was actually reached with the configured, non-hardcoded values.
    assert captured_llm_call["model"] == settings.gemini_model_synthesis
    assert captured_llm_call["api_key"] == SENTINEL_KEY
    assert captured_llm_call["stream"] is True

    # The SSE stream carries the synthesized tokens and completes normally.
    assert "The report " in full_stream
    assert "says X." in full_stream
    assert "event: completed" in full_stream

    # The credential never appears anywhere in what was sent back to the client.
    assert SENTINEL_KEY not in full_stream
