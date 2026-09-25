"""
Regression tests for evaluation/nli_checker.py::NLIChecker.

Bug fixed: check_entailment() unconditionally returned True -- every
generated claim was reported as faithful to its source regardless of
whether the retrieved context said anything of the kind. Now runs a real
NLI cross-encoder (injected as a fake here, so no model download).
"""
from types import SimpleNamespace

import pytest

from lexis.evaluation.nli_checker import NLIChecker, _default_model_name
from lexis.registry.models import ModelRegistry
from lexis.packs.loader import load_pack
from pathlib import Path


def make_fake_model(id2label, row):
    """row: the softmax-probability row predict() should return for any single pair."""
    model = SimpleNamespace()
    model.config = SimpleNamespace(id2label=id2label)
    model.predict = lambda pairs, apply_softmax=True: [row for _ in pairs]
    return model


def test_resolves_entailment_index_from_a_non_standard_label_order():
    """Proves the index genuinely comes from config.id2label, not a hardcoded
    position -- entailment is deliberately NOT at index 0 or 1 here."""
    model = make_fake_model({0: "neutral", 1: "contradiction", 2: "entailment"}, row=[0.1, 0.1, 0.8])
    checker = NLIChecker(model=model)
    assert checker._entailment_idx == 2
    assert checker.entailment_score("premise", "hypothesis") == pytest.approx(0.8)


def test_resolves_entailment_index_case_and_whitespace_insensitively():
    model = make_fake_model({0: " Entailment ", 1: "contradiction", 2: "neutral"}, row=[0.9, 0.05, 0.05])
    checker = NLIChecker(model=model)
    assert checker._entailment_idx == 0


def test_raises_if_model_has_no_id2label():
    model = SimpleNamespace(config=SimpleNamespace())
    with pytest.raises(ValueError, match="id2label"):
        NLIChecker(model=model)


def test_raises_if_id2label_has_no_entailment_label():
    model = make_fake_model({0: "positive", 1: "negative"}, row=[0.5, 0.5])
    with pytest.raises(ValueError, match="entailment"):
        NLIChecker(model=model)


def test_check_entailment_true_above_threshold():
    model = make_fake_model({0: "entailment", 1: "neutral", 2: "contradiction"}, row=[0.9, 0.05, 0.05])
    checker = NLIChecker(model=model)
    assert checker.check_entailment("The term is five years.", "The contract lasts five years.") is True


def test_check_entailment_false_below_threshold():
    model = make_fake_model({0: "entailment", 1: "neutral", 2: "contradiction"}, row=[0.1, 0.8, 0.1])
    checker = NLIChecker(model=model)
    assert checker.check_entailment("The term is five years.", "The contract lasts ten years.") is False


def test_check_entailment_respects_a_custom_threshold():
    model = make_fake_model({0: "entailment", 1: "neutral", 2: "contradiction"}, row=[0.6, 0.3, 0.1])
    checker = NLIChecker(model=model)
    assert checker.check_entailment("p", "h", threshold=0.5) is True
    assert checker.check_entailment("p", "h", threshold=0.7) is False


def test_does_not_unconditionally_return_true():
    """Documents the actual bug: the old implementation ignored its inputs
    entirely. A checker built on a model that scores everything as
    contradiction must be able to say so."""
    model = make_fake_model({0: "entailment", 1: "neutral", 2: "contradiction"}, row=[0.0, 0.0, 1.0])
    checker = NLIChecker(model=model)
    assert checker.check_entailment("The sky is blue.", "The contract is void.") is False


def test_default_model_name_resolves_the_us_contracts_pack_nli_model():
    """No fakes here -- checks the real registry/pack resolution logic
    (config/models.yaml + config/packs/us_contracts.yaml) without loading
    an actual CrossEncoder."""
    pack = load_pack(Path("config/packs/us_contracts.yaml"))
    registry = ModelRegistry.from_file(Path("config/models.yaml"))
    expected = registry.resolve(pack.models.nli, kind="nli").name
    assert _default_model_name() == expected
