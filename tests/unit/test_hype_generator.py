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

import pytest

from lexis.config import settings
from lexis.ingestion import hype_generator as hype_generator_module
from lexis.ingestion.hype_generator import HyPEGenerator


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
