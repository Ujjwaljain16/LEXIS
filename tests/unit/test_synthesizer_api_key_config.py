"""
Regression tests for generation/synthesizer.py::LexisSynthesizer.stream_answer
authentication.

Bug fixed: stream_answer() passed `model=` to litellm's acompletion() but never
passed `api_key=`, and nothing in the app exports .env values into
os.environ (pydantic-settings' env_file only populates Settings' own
fields). evaluation/generate_dataset.py already establishes the correct
project pattern -- pass api_key=settings.gemini_api_key explicitly -- and
stream_answer() now does the same.

The external LLM call is mocked throughout. A clearly-fake sentinel string
stands in for the API key; it is never printed, logged, or asserted via a
substring in any human-readable failure message, per the security
requirements for this change.
"""
import logging
from types import SimpleNamespace

import pytest

from lexis.config import settings
from lexis.generation import synthesizer as synthesizer_module
from lexis.generation.synthesizer import LexisSynthesizer

SENTINEL_KEY = "sk-test-sentinel-not-a-real-credential"  # noqa: S105 -- test-only placeholder, never a live key


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


GENERIC_CHUNKS = [
    {"chunk_id": "pqac-generic-1", "text": "A passage from an arbitrary report.", "_source_path": "path_b_global"},
]


# --- 1. Configuration forwarding ---

@pytest.mark.asyncio
async def test_stream_answer_forwards_configured_model_and_api_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SENTINEL_KEY)
    captured = {}
    monkeypatch.setattr(synthesizer_module, "acompletion", make_fake_acompletion(captured))

    synth = LexisSynthesizer()
    result = "".join([tok async for tok in synth.stream_answer("A generic question.", GENERIC_CHUNKS)])

    assert captured["model"] == settings.gemini_model_synthesis
    assert captured["api_key"] == SENTINEL_KEY
    assert result == "Answer text."


# --- 2. Configuration independence (not hardcoded, not os.environ, not stale) ---

@pytest.mark.asyncio
async def test_changing_configured_api_key_changes_the_forwarded_value(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "first-sentinel-key")
    captured_first = {}
    monkeypatch.setattr(synthesizer_module, "acompletion", make_fake_acompletion(captured_first))
    synth = LexisSynthesizer()
    _ = [tok async for tok in synth.stream_answer("q1", GENERIC_CHUNKS)]
    assert captured_first["api_key"] == "first-sentinel-key"

    monkeypatch.setattr(settings, "gemini_api_key", "second-different-sentinel-key")
    captured_second = {}
    monkeypatch.setattr(synthesizer_module, "acompletion", make_fake_acompletion(captured_second))
    _ = [tok async for tok in synth.stream_answer("q2", GENERIC_CHUNKS)]
    assert captured_second["api_key"] == "second-different-sentinel-key"
    assert captured_second["api_key"] != captured_first["api_key"]


@pytest.mark.asyncio
async def test_forwarded_api_key_is_not_read_from_os_environ(monkeypatch):
    """Guards against a regression to os.environ["GEMINI_API_KEY"] instead of
    the Settings object -- set a decoy in os.environ that must NOT be used."""
    monkeypatch.setenv("GEMINI_API_KEY", "decoy-value-from-os-environ-must-not-be-used")
    monkeypatch.setattr(settings, "gemini_api_key", SENTINEL_KEY)
    captured = {}
    monkeypatch.setattr(synthesizer_module, "acompletion", make_fake_acompletion(captured))

    synth = LexisSynthesizer()
    _ = [tok async for tok in synth.stream_answer("q", GENERIC_CHUNKS)]

    assert captured["api_key"] == SENTINEL_KEY
    assert captured["api_key"] != "decoy-value-from-os-environ-must-not-be-used"


@pytest.mark.asyncio
async def test_changing_configured_model_and_key_together_both_forward_correctly(monkeypatch):
    monkeypatch.setattr(settings, "gemini_model_synthesis", "sentinel/model-9")
    monkeypatch.setattr(settings, "gemini_api_key", "sentinel-key-9")
    captured = {}
    monkeypatch.setattr(synthesizer_module, "acompletion", make_fake_acompletion(captured))

    synth = LexisSynthesizer()
    _ = [tok async for tok in synth.stream_answer("q", GENERIC_CHUNKS)]

    assert captured["model"] == "sentinel/model-9"
    assert captured["api_key"] == "sentinel-key-9"


# --- 3. Secret safety ---

@pytest.mark.asyncio
async def test_api_key_does_not_appear_in_the_yielded_response(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SENTINEL_KEY)
    captured = {}
    monkeypatch.setattr(synthesizer_module, "acompletion", make_fake_acompletion(captured, tokens=("Some", " answer", " text.")))

    synth = LexisSynthesizer()
    result = "".join([tok async for tok in synth.stream_answer("q", GENERIC_CHUNKS)])

    assert SENTINEL_KEY not in result


@pytest.mark.asyncio
async def test_api_key_does_not_appear_in_logs_during_a_call(monkeypatch, caplog):
    monkeypatch.setattr(settings, "gemini_api_key", SENTINEL_KEY)
    captured = {}
    monkeypatch.setattr(synthesizer_module, "acompletion", make_fake_acompletion(captured))

    with caplog.at_level(logging.DEBUG):
        synth = LexisSynthesizer()
        _ = [tok async for tok in synth.stream_answer("q", GENERIC_CHUNKS)]

    for record in caplog.records:
        assert SENTINEL_KEY not in record.getMessage()


@pytest.mark.asyncio
async def test_api_key_does_not_leak_into_the_synthesis_error_path(monkeypatch):
    """stream_answer() catches acompletion failures and yields the exception
    text directly to the caller. Our own code must never construct that
    exception (or any wrapping text) using the key itself -- simulate a
    realistic auth failure (as litellm raises, without embedding the key,
    matching how real auth-failure exceptions from LLM SDKs behave) and
    confirm the sentinel is absent from what reaches the client."""
    monkeypatch.setattr(settings, "gemini_api_key", SENTINEL_KEY)

    async def failing_acompletion(**kwargs):
        raise RuntimeError("AuthenticationError: invalid API key provided")

    monkeypatch.setattr(synthesizer_module, "acompletion", failing_acompletion)

    synth = LexisSynthesizer()
    result = "".join([tok async for tok in synth.stream_answer("q", GENERIC_CHUNKS)])

    assert SENTINEL_KEY not in result
    assert "AuthenticationError" in result  # the (key-free) failure reason is still surfaced


# --- 4. Streaming behaviour preserved ---

@pytest.mark.asyncio
async def test_stream_true_is_still_passed_and_response_remains_a_token_stream(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SENTINEL_KEY)
    captured = {}
    monkeypatch.setattr(synthesizer_module, "acompletion", make_fake_acompletion(captured, tokens=("tok1", "tok2", "tok3")))

    synth = LexisSynthesizer()
    yielded = [tok async for tok in synth.stream_answer("q", GENERIC_CHUNKS)]

    assert captured["stream"] is True
    assert yielded == ["tok1", "tok2", "tok3"]
