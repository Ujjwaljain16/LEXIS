"""
Regression tests for ingestion/hype_generator.py::HyPEGenerator.

Bug fixed: generate_questions() called acompletion(model=settings.llm_model,
...), but Settings never defined an llm_model field (same class of bug as
generation/synthesizer.py, see test_synthesizer_model_config.py). Every
real call raised AttributeError, which a bare `except Exception: return []`
silently swallowed -- so this entire module produced empty results forever
without ever surfacing an error. Now uses settings.gemini_model_feature
(the same field ingestion/feature_extractor.py already uses for other
per-chunk LLM enrichment), and failures are logged instead of swallowed
silently.
"""
import asyncio
import json
from types import SimpleNamespace

import litellm
import pytest

from lexis.config import settings
from lexis.ingestion import hype_generator as hype_generator_module
from lexis.ingestion.hype_generator import HyPEGenerator


def rate_limit_error(message="quota exceeded, quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"):
    return litellm.RateLimitError(message=message, llm_provider="gemini", model="gemini-2.5-flash")


def daily_quota_error():
    # Matches the real shape of a live Gemini free-tier daily-cap error (confirmed in
    # production use): the quotaId embedded in the message names which quota was hit.
    return rate_limit_error(
        message="quota exceeded, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: 20"
    )


def make_fake_acompletion(captured: dict, questions=("What is the term?", "Who are the parties?", "Is it exclusive?")):
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        content = json.dumps({"questions": list(questions)})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return fake_acompletion


@pytest.mark.asyncio
async def test_generate_questions_uses_the_configured_feature_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(hype_generator_module, "acompletion", make_fake_acompletion(captured))

    gen = HyPEGenerator()
    questions = await gen.generate_questions("The term of this agreement is five years.")

    assert captured["model"] == settings.gemini_model_feature
    assert questions == ["What is the term?", "Who are the parties?", "Is it exclusive?"]


@pytest.mark.asyncio
async def test_generate_questions_does_not_reference_nonexistent_llm_model_setting():
    """Documents the actual bug: Settings never had this field."""
    assert not hasattr(settings, "llm_model")


@pytest.mark.asyncio
async def test_changing_configured_feature_model_changes_the_llm_call(monkeypatch):
    monkeypatch.setattr(settings, "gemini_model_feature", "sentinel/feature-model-v9")
    captured = {}
    monkeypatch.setattr(hype_generator_module, "acompletion", make_fake_acompletion(captured))

    gen = HyPEGenerator()
    await gen.generate_questions("Some chunk text.")

    assert captured["model"] == "sentinel/feature-model-v9"


@pytest.mark.asyncio
async def test_requests_the_configured_number_of_questions(monkeypatch):
    monkeypatch.setattr(settings, "hype_questions_per_chunk", 5)
    captured = {}
    monkeypatch.setattr(hype_generator_module, "acompletion", make_fake_acompletion(captured))

    gen = HyPEGenerator()
    await gen.generate_questions("Some chunk text.")

    system_msg = next(m["content"] for m in captured["messages"] if m["role"] == "system")
    assert "exactly 5" in system_msg


@pytest.mark.asyncio
async def test_a_failed_call_returns_an_empty_list_instead_of_raising(monkeypatch):
    async def failing_acompletion(**kwargs):
        raise RuntimeError("simulated provider outage")
    monkeypatch.setattr(hype_generator_module, "acompletion", failing_acompletion)

    gen = HyPEGenerator()
    questions = await gen.generate_questions("Some chunk text.")

    assert questions == []


@pytest.mark.asyncio
async def test_a_slow_call_times_out_instead_of_hanging_forever(monkeypatch):
    """Regression: pipeline.py fans this call out concurrently across every
    chunk of a document via asyncio.gather -- without a per-call timeout,
    one stuck request would block the whole document's ingestion
    indefinitely, not just its own 3 questions."""
    monkeypatch.setattr(settings, "hype_generation_timeout_s", 0.05)

    async def hanging_acompletion(**kwargs):
        await asyncio.sleep(10)
        raise AssertionError("should have been cancelled by the timeout long before this")

    monkeypatch.setattr(hype_generator_module, "acompletion", hanging_acompletion)

    gen = HyPEGenerator()
    questions = await asyncio.wait_for(gen.generate_questions("Some chunk text."), timeout=2.0)

    assert questions == []


@pytest.mark.asyncio
async def test_rate_limit_error_is_retried_and_succeeds_on_a_later_attempt(monkeypatch):
    monkeypatch.setattr(settings, "hype_max_retries", 3)
    monkeypatch.setattr(settings, "hype_retry_backoff_s", 0.001)  # keep the test fast

    calls = {"n": 0}

    async def flaky_acompletion(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise rate_limit_error()
        content = json.dumps({"questions": ["Recovered question?"]})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(hype_generator_module, "acompletion", flaky_acompletion)

    gen = HyPEGenerator()
    questions = await gen.generate_questions("Some chunk text.")

    assert questions == ["Recovered question?"]
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_rate_limit_error_gives_up_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(settings, "hype_max_retries", 2)
    monkeypatch.setattr(settings, "hype_retry_backoff_s", 0.001)

    calls = {"n": 0}

    async def always_rate_limited(**kwargs):
        calls["n"] += 1
        raise rate_limit_error()

    monkeypatch.setattr(hype_generator_module, "acompletion", always_rate_limited)

    gen = HyPEGenerator()
    questions = await gen.generate_questions("Some chunk text.")

    assert questions == []
    assert calls["n"] == 3  # initial attempt + 2 retries, then give up


@pytest.mark.asyncio
async def test_daily_quota_exhaustion_is_not_retried(monkeypatch):
    """A per-DAY quota will not reset within this run no matter how long a per-minute-style
    backoff waits -- confirmed live (a real Colab run burned minutes of backoff-and-retry
    against a 20-requests/day cap before this fix). Must fail on the first attempt, not after
    exhausting hype_max_retries."""
    monkeypatch.setattr(settings, "hype_max_retries", 3)
    monkeypatch.setattr(settings, "hype_retry_backoff_s", 0.001)
    calls = {"n": 0}

    async def always_daily_exhausted(**kwargs):
        calls["n"] += 1
        raise daily_quota_error()

    monkeypatch.setattr(hype_generator_module, "acompletion", always_daily_exhausted)

    gen = HyPEGenerator()
    questions = await gen.generate_questions("Some chunk text.")

    assert questions == []
    assert calls["n"] == 1  # no retries burned on a quota that can't reset mid-run


@pytest.mark.asyncio
async def test_daily_quota_exhaustion_short_circuits_every_later_call_on_the_same_instance(monkeypatch):
    """Once confirmed once, every other chunk's call this run would fail identically --
    pipeline.py's _build_hype_points calls generate_questions once per chunk (potentially
    hundreds) via the SAME HyPEGenerator instance, so this must stop making real API calls
    at all after the first daily-exhaustion, not just stop retrying that one call."""
    monkeypatch.setattr(settings, "hype_retry_backoff_s", 0.001)
    calls = {"n": 0}

    async def always_daily_exhausted(**kwargs):
        calls["n"] += 1
        raise daily_quota_error()

    monkeypatch.setattr(hype_generator_module, "acompletion", always_daily_exhausted)

    gen = HyPEGenerator()
    first = await gen.generate_questions("Chunk one.")
    second = await gen.generate_questions("Chunk two.")
    third = await gen.generate_questions("Chunk three.")

    assert first == second == third == []
    assert calls["n"] == 1  # only the first call ever reached acompletion


@pytest.mark.asyncio
async def test_per_minute_rate_limit_is_still_retried_normally_even_after_checking_for_daily(monkeypatch):
    """Regression: the daily-quota check must not accidentally swallow the existing
    per-minute retry behavior for a plain rate-limit message."""
    monkeypatch.setattr(settings, "hype_max_retries", 3)
    monkeypatch.setattr(settings, "hype_retry_backoff_s", 0.001)
    calls = {"n": 0}

    async def flaky_acompletion(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise rate_limit_error()  # plain per-minute message, no "PerDay"
        content = json.dumps({"questions": ["Recovered?"]})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(hype_generator_module, "acompletion", flaky_acompletion)

    gen = HyPEGenerator()
    questions = await gen.generate_questions("Some chunk text.")

    assert questions == ["Recovered?"]
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_a_non_rate_limit_error_is_not_retried(monkeypatch):
    """Retrying only makes sense for a transient rate limit -- a genuine
    error (bad request, auth failure, etc.) should fail fast, not burn
    hype_max_retries attempts waiting for something that will never
    resolve by itself."""
    monkeypatch.setattr(settings, "hype_max_retries", 3)
    calls = {"n": 0}

    async def failing_acompletion(**kwargs):
        calls["n"] += 1
        raise RuntimeError("not a rate limit")

    monkeypatch.setattr(hype_generator_module, "acompletion", failing_acompletion)

    gen = HyPEGenerator()
    questions = await gen.generate_questions("Some chunk text.")

    assert questions == []
    assert calls["n"] == 1
