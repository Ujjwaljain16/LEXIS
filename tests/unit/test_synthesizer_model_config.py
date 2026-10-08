"""
Regression tests for generation/synthesizer.py::LexisSynthesizer.stream_answer.

Bug fixed: stream_answer() called acompletion(model=settings.llm_model, ...),
but Settings never defined an `llm_model` field (config.py only defines
gemini_model_synthesis/gemini_model_feature/gemini_model_vision). Since this
method is on the live /query/fast path, every real call raised AttributeError
before ever reaching the LLM. The synthesizer performs answer synthesis, so it
now uses the existing settings.gemini_model_synthesis field -- no new config
field was introduced, and no model name is hardcoded in the synthesizer.

The external LLM call is mocked throughout; these tests only verify what
model name is requested, not real generation behaviour.
"""
from types import SimpleNamespace

import pytest

from lexis.config import settings
from lexis.generation import synthesizer as synthesizer_module
from lexis.generation.synthesizer import LexisSynthesizer


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


def make_fake_acompletion(captured: dict, tokens=("Answer", " text.")):
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return FakeStream(list(tokens))
    return fake_acompletion


# Generic, non-legal packed chunks -- proves nothing here depends on a
# contract/clause document structure.
GENERIC_CHUNKS = [
    {"chunk_id": "pqac-generic-1", "text": "A passage from an arbitrary report.", "_source_path": "path_b_global"},
    {"chunk_id": "pqac-generic-2", "text": "A passage from an unrelated memo.", "_source_path": "path_d_bm25"},
]


@pytest.mark.asyncio
async def test_stream_answer_uses_configured_synthesis_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(synthesizer_module, "acompletion", make_fake_acompletion(captured))

    synth = LexisSynthesizer()
    result = "".join([tok async for tok in synth.stream_answer("What does the report say?", GENERIC_CHUNKS)])

    assert captured["model"] == settings.gemini_model_synthesis
    assert result == "Answer text."


@pytest.mark.asyncio
async def test_stream_answer_does_not_reference_nonexistent_llm_model_setting():
    """Documents the actual bug: Settings never had this field. If something
    reintroduces settings.llm_model for an unrelated reason later, this test
    forces a conscious review of whether the synthesizer should keep using
    gemini_model_synthesis instead."""
    assert not hasattr(settings, "llm_model")


@pytest.mark.asyncio
async def test_changing_configured_synthesis_model_changes_the_llm_call(monkeypatch):
    monkeypatch.setattr(settings, "gemini_model_synthesis", "sentinel/synthesis-model-v9")
    captured = {}
    monkeypatch.setattr(synthesizer_module, "acompletion", make_fake_acompletion(captured))

    synth = LexisSynthesizer()
    _ = [tok async for tok in synth.stream_answer("A generic question.", GENERIC_CHUNKS)]

    assert captured["model"] == "sentinel/synthesis-model-v9"


@pytest.mark.asyncio
async def test_no_hardcoded_model_name_survives_a_config_change(monkeypatch):
    """If any real model string were hardcoded as a fallback/workaround inside
    stream_answer, changing config would not fully control what's sent."""
    monkeypatch.setattr(settings, "gemini_model_synthesis", "another-sentinel-model")
    captured = {}
    monkeypatch.setattr(synthesizer_module, "acompletion", make_fake_acompletion(captured))

    synth = LexisSynthesizer()
    _ = [tok async for tok in synth.stream_answer("Another generic question.", GENERIC_CHUNKS)]

    assert captured["model"] == "another-sentinel-model"
    assert captured["model"] not in ("gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro")
