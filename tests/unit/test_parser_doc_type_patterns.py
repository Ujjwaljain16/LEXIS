"""
Regression tests for ingestion/parser.py::LexisParser._detect_doc_type.

Bug fixed: doc-type classification used a module-level DOC_TYPE_PATTERNS
constant -- hardcoded English vocabulary ("agreement", "contract",
"whereas", ...) baked into the base parser, applied to every ingested
document regardless of jurisdiction/language. It's now sourced from the
active jurisdiction pack's structure.doc_type_patterns
(config/packs/*.yaml), with LexisParser() defaulting to the us_contracts
pack so existing behavior and frozen baselines are unchanged.
"""
from lexis.ingestion.parser import LexisParser, _default_doc_type_patterns


def test_default_parser_uses_the_us_contracts_pack_patterns():
    parser = LexisParser()
    assert parser.doc_type_patterns == _default_doc_type_patterns()
    assert "contract" in parser.doc_type_patterns


def test_default_parser_classifies_contract_text():
    parser = LexisParser()
    assert parser._detect_doc_type("This Agreement is entered into by the parties.") == "contract"


def test_default_parser_classifies_10k_text():
    parser = LexisParser()
    assert parser._detect_doc_type("FORM 10-K Annual Report pursuant to Section 13") == "10-k"


def test_default_parser_falls_back_to_unknown():
    parser = LexisParser()
    assert parser._detect_doc_type("A short note about lunch plans.") == "unknown"


def test_custom_doc_type_patterns_override_the_default_entirely():
    """Proves classification is genuinely config-driven: a pack with
    completely different (non-English) vocabulary changes what gets
    detected, with zero code changes."""
    parser = LexisParser(doc_type_patterns={"karar": [r"yüksek mahkeme", r"karar"]})
    assert parser._detect_doc_type("Bu bir mahkeme kararıdır.") == "karar"
    # The same text would never match the default (English) pattern set.
    assert LexisParser()._detect_doc_type("Bu bir mahkeme kararıdır.") == "unknown"


def test_empty_doc_type_patterns_always_classifies_unknown():
    """Matches the india/uk_eu_statutes packs today: doc_type_patterns is
    an explicit empty dict (not yet built), so every document is
    "unknown" -- honest, not silently misclassified with US-English
    vocabulary from a different pack."""
    parser = LexisParser(doc_type_patterns={})
    assert parser._detect_doc_type("This Agreement is entered into by the parties.") == "unknown"
