"""serving/service.py + the /v2/query/fast route, end to end over HTTP with fakes at the two external
boundaries only: retrieval (no Qdrant/BM25/embedder) and the litellm call (no network). The real
AnswerService, grounding, synthesizer.stream_grounded, security and SSE encoding all run.

Content is deliberately generic and non-legal: nothing in the answer path depends on a document
schema."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lexis.config import settings
from lexis.generation import synthesizer as synthesizer_module
from lexis.generation.synthesizer import LexisSynthesizer
from lexis.serving.app import create_app
from lexis.serving.routes import query as query_module
from lexis.serving.service import AnswerService

SENTINEL_KEY = "sk-test-sentinel-e2e-not-a-real-credential"


def chunk(i, text, doc="doc-a"):
    return {"payload": {"chunk_id": f"c{i}", "doc_id": doc, "content": text, "page_num": i, "chunk_index": i, "doc_type": "report"},
            "text": text}


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    async def retrieve(self, query, document_ids):
        self.calls.append((query, document_ids))
        return self.chunks


class FakeGenerator:
    def __init__(self, tokens=None, error=None):
        self.tokens, self.error, self.seen_sources = tokens or [], error, None

    async def stream_grounded(self, query, sources):
        self.seen_sources = sources
        if self.error:
            raise self.error
        for t in self.tokens:
            yield t


class FakeVerifier:
    def check_entailment(self, premise, hypothesis, threshold=None):
        return "five years" in premise


async def collect(service, query="q", document_ids=None):
    return [e async for e in service.stream(query, document_ids)]


def types(events):
    return [e.type for e in events]


def by_type(events, t):
    return [e for e in events if e.type == t]


CHUNKS = [chunk(1, "The report covers a five years period."), chunk(2, "Costs were reviewed annually.")]


@pytest.mark.asyncio
async def test_happy_path_streams_validates_and_cites():
    gen = FakeGenerator(["The period is five", " years [1]. Costs are annual [2]", " and secret [9]."])
    events = await collect(AnswerService(FakeRetriever(CHUNKS), gen))

    # The first ~20 chars are held back (abstention-marker check), so the first two tokens arrive merged.
    assert types(events) == ["status", "status", "token", "token", "answer", "citations", "completed"]
    answer = by_type(events, "answer")[0].data
    assert answer["text"] == "The period is five years [1]. Costs are annual [2] and secret."
    assert answer["invalid_citation_indices"] == [9]
    cited = by_type(events, "citations")[0].data["citations"]
    assert [c["index"] for c in cited] == [1, 2]
    assert cited[0]["chunk_id"] == "c1" and cited[0]["text"] == "The report covers a five years period."
    assert [s.index for s in gen.seen_sources] == [1, 2]


@pytest.mark.asyncio
async def test_no_context_abstains_without_calling_the_model():
    gen = FakeGenerator(["should not be used"])
    events = await collect(AnswerService(FakeRetriever([]), gen))
    assert types(events) == ["status", "abstained", "completed"]
    assert by_type(events, "abstained")[0].data["reason"] == "no_context"
    assert gen.seen_sources is None


@pytest.mark.asyncio
async def test_model_abstention_marker_is_not_streamed_to_the_user():
    gen = FakeGenerator(["INSUFFICIENT", "_CONTEXT"])
    events = await collect(AnswerService(FakeRetriever(CHUNKS), gen))
    assert "token" not in types(events)
    assert by_type(events, "abstained")[0].data["reason"] == "model_declined"
    assert "answer" not in types(events) and "citations" not in types(events)


@pytest.mark.asyncio
async def test_short_answer_below_marker_length_is_still_delivered():
    events = await collect(AnswerService(FakeRetriever(CHUNKS), FakeGenerator(["Yes."])))
    assert "".join(e.data["text"] for e in by_type(events, "token")) == "Yes."
    assert by_type(events, "answer")[0].data["text"] == "Yes."


@pytest.mark.asyncio
async def test_generator_failure_yields_generic_failed_event_not_the_raw_error():
    gen = FakeGenerator(error=RuntimeError(f"provider said key {SENTINEL_KEY} is invalid"))
    events = await collect(AnswerService(FakeRetriever(CHUNKS), gen))
    assert types(events)[-1] == "failed"
    payload = json.dumps(events[-1].data)
    assert "RuntimeError" in payload and SENTINEL_KEY not in payload
    assert "completed" not in types(events)


@pytest.mark.asyncio
async def test_document_ids_are_passed_to_the_retriever():
    retr = FakeRetriever(CHUNKS)
    await collect(AnswerService(retr, FakeGenerator(["ok [1]."])), "what?", ["doc-a"])
    assert retr.calls == [("what?", ["doc-a"])]


@pytest.mark.asyncio
async def test_context_is_capped_at_answer_context_chunks(monkeypatch):
    monkeypatch.setattr(settings, "answer_context_chunks", 1)
    gen = FakeGenerator(["fine [1]."])
    await collect(AnswerService(FakeRetriever(CHUNKS), gen))
    assert len(gen.seen_sources) == 1


@pytest.mark.asyncio
async def test_verification_runs_only_when_a_verifier_is_configured():
    gen_tokens = ["It spans five years [1]. Costs are annual [2]."]
    without = await collect(AnswerService(FakeRetriever(CHUNKS), FakeGenerator(gen_tokens)))
    assert "verification" not in types(without)

    with_v = await collect(AnswerService(FakeRetriever(CHUNKS), FakeGenerator(gen_tokens), FakeVerifier()))
    report = by_type(with_v, "verification")[0].data
    assert [c["status"] for c in report["claims"]] == ["supported", "unsupported"]
    assert report["support_rate"] == 0.5
    assert types(with_v)[-1] == "completed"


# ---- HTTP level -----------------------------------------------------------------------------------

class FakeStreamChunk:
    def __init__(self, content):
        self.choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]


class FakeLLMStream:
    def __init__(self, tokens):
        self._tokens = tokens

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for t in self._tokens:
            yield FakeStreamChunk(t)


def parse_sse(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SENTINEL_KEY)
    monkeypatch.setattr(settings, "lexis_api_keys", "good:acme")
    monkeypatch.setattr(settings, "auth_disabled", False)
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return FakeLLMStream(["The period is five years [1]."])

    monkeypatch.setattr(synthesizer_module, "acompletion", fake_acompletion)
    retriever = FakeRetriever(CHUNKS)
    app = create_app()
    app.dependency_overrides[query_module.get_answer_service] = lambda: AnswerService(retriever, LexisSynthesizer())
    query_module._fast_limiter._hits.clear()
    return TestClient(app), captured, retriever


def test_http_requires_a_key(http):
    client, _, _ = http
    assert client.post("/v2/query/fast", json={"query": "q"}).status_code == 401
    assert client.get("/v2/health").status_code == 200   # health stays open


def test_http_stream_reaches_llm_with_configured_model_and_key_and_never_leaks_it(http):
    client, captured, retriever = http
    r = client.post("/v2/query/fast", json={"query": "How long?", "document_ids": ["doc-a"]}, headers={"X-API-Key": "good"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")

    assert captured["model"] == settings.gemini_model_synthesis
    assert captured["api_key"] == SENTINEL_KEY and captured["stream"] is True
    assert "[1] The report covers a five years period." in captured["messages"][1]["content"]
    assert retriever.calls == [("How long?", ["doc-a"])]

    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "trace" and names[-1] == "completed"
    citations = dict(events)["citations"]["citations"]
    assert citations[0]["chunk_id"] == "c1"
    assert SENTINEL_KEY not in r.text


def test_http_validates_request_body(http):
    client, _, _ = http
    h = {"X-API-Key": "good"}
    assert client.post("/v2/query/fast", json={"query": ""}, headers=h).status_code == 422
    assert client.post("/v2/query/fast", json={"query": "x" * (settings.api_max_query_chars + 1)}, headers=h).status_code == 422
    assert client.post("/v2/query/fast", json={"query": "ok", "document_ids": ["d"] * (settings.api_max_document_ids + 1)}, headers=h).status_code == 422


def test_http_rate_limit(http, monkeypatch):
    client, _, _ = http
    monkeypatch.setattr(query_module._fast_limiter, "limit", 1)
    h = {"X-API-Key": "good"}
    assert client.post("/v2/query/fast", json={"query": "q"}, headers=h).status_code == 200
    assert client.post("/v2/query/fast", json={"query": "q"}, headers=h).status_code == 429
