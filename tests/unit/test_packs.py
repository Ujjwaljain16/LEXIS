import pytest
import yaml
from pydantic import ValidationError

from lexis.packs.citations import extract_citations
from lexis.packs.loader import PackError, discover_packs, load_pack, validate_against_registry
from lexis.packs.schema import CitationGrammar, JurisdictionPack
from lexis.registry.models import ModelRegistry

PROMPTS = {"system": "s", "refusal": "r", "insufficient_marker": "M"}


def pack_dict(**kw):
    base = {"name": "test_pack", "languages": ["EN"], "prompts": PROMPTS}
    base.update(kw)
    return base


def grammar(**kw):
    base = {"name": "g", "pattern": r"\d{3}-[A-Z]{2}", "examples": ["123-AB"]}
    base.update(kw)
    return base


# --- grammar self-test ---

def test_grammar_accepts_matching_examples():
    assert CitationGrammar(**grammar()).name == "g"


def test_grammar_rejects_example_that_does_not_fully_match():
    with pytest.raises(ValidationError, match="does not fully match"):
        CitationGrammar(**grammar(examples=["123-ABC"]))


def test_grammar_rejects_non_example_that_matches():
    with pytest.raises(ValidationError, match="unexpectedly matches"):
        CitationGrammar(**grammar(non_examples=["see 999-ZZ here"]))


def test_grammar_rejects_bad_regex():
    with pytest.raises(ValidationError, match="does not compile"):
        CitationGrammar(**grammar(pattern="(unclosed", examples=["x"]))


def test_grammar_requires_at_least_one_example():
    with pytest.raises(ValidationError):
        CitationGrammar(**grammar(examples=[]))


# --- pack schema ---

def test_minimal_pack_is_valid_and_lowercases_languages():
    assert JurisdictionPack(**pack_dict()).languages == ["en"]


@pytest.mark.parametrize("name", ["Bad-Name", "1abc", "", "has space"])
def test_pack_name_must_be_a_slug(name):
    with pytest.raises(ValidationError):
        JurisdictionPack(**pack_dict(name=name))


def test_pack_requires_a_language_and_prompts():
    with pytest.raises(ValidationError):
        JurisdictionPack(**pack_dict(languages=[]))
    with pytest.raises(ValidationError):
        JurisdictionPack(name="p", languages=["en"])


def test_trusted_source_weights_must_be_in_unit_interval():
    with pytest.raises(ValidationError, match="within"):
        JurisdictionPack(**pack_dict(trusted_sources={"a.org": 1.5}))
    assert JurisdictionPack(**pack_dict(trusted_sources={"a.org": 0.0, "b.org": 1.0})).trusted_sources


def test_duplicate_grammar_names_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        JurisdictionPack(**pack_dict(citation_grammars=[grammar(), grammar()]))


def test_bm25_token_pattern_must_compile():
    with pytest.raises(ValidationError):
        JurisdictionPack(**pack_dict(bm25={"token_pattern": "(bad"}))


def test_max_chunk_tokens_must_be_positive():
    with pytest.raises(ValidationError):
        JurisdictionPack(**pack_dict(structure={"max_chunk_tokens": 0}))


def test_doc_type_patterns_default_to_empty():
    pack = JurisdictionPack(**pack_dict())
    assert pack.structure.doc_type_patterns == {}


def test_doc_type_patterns_round_trip():
    pack = JurisdictionPack(**pack_dict(structure={"doc_type_patterns": {"contract": ["agreement", "covenant"]}}))
    assert pack.structure.doc_type_patterns == {"contract": ["agreement", "covenant"]}


def test_doc_type_patterns_rejects_a_pattern_that_does_not_compile():
    with pytest.raises(ValidationError, match="does not compile"):
        JurisdictionPack(**pack_dict(structure={"doc_type_patterns": {"contract": ["(bad"]}}))


# --- loading ---

def write_pack(tmp_path, name="test_pack", **kw):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(pack_dict(name=name, **kw)))
    return path


def test_load_pack_name_must_match_filename(tmp_path):
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump(pack_dict(name="test_pack")))
    with pytest.raises(PackError, match="file name"):
        load_pack(path)


def test_discover_packs_loads_every_yaml(tmp_path):
    write_pack(tmp_path, "alpha")
    write_pack(tmp_path, "beta")
    assert sorted(discover_packs(tmp_path)) == ["alpha", "beta"]


def test_adding_a_pack_needs_no_code_change(tmp_path):
    write_pack(tmp_path, "brand_new_jurisdiction", citation_grammars=[grammar()])
    pack = discover_packs(tmp_path)["brand_new_jurisdiction"]
    assert [m.text for m in extract_citations("cite 123-AB now", pack)] == ["123-AB"]


# --- registry cross-validation ---

def reg():
    base = dict(license="MIT", license_verified=True, commercial_ok=True, hardware="cpu")
    return ModelRegistry.from_mapping({"models": {
        "emb": dict(name="o/e", kind="embedder", dim=4, **base),
        "rr": dict(name="o/r", kind="reranker", **base),
    }})


def test_validate_passes_for_known_correct_kinds():
    validate_against_registry(JurisdictionPack(**pack_dict(models={"embedder": "emb", "reranker": "rr"})), reg())


def test_validate_reports_all_problems_at_once():
    pack = JurisdictionPack(**pack_dict(models={"embedder": "missing", "reranker": "emb"}))
    with pytest.raises(PackError) as e:
        validate_against_registry(pack, reg())
    msg = str(e.value)
    assert "embedder" in msg and "reranker" in msg


def test_unset_model_refs_are_ignored():
    validate_against_registry(JurisdictionPack(**pack_dict()), reg())


# --- citation extraction ---

def two_grammar_pack():
    return JurisdictionPack(**pack_dict(citation_grammars=[
        grammar(name="short", pattern=r"\d{3}-[A-Z]{2}", examples=["123-AB"]),
        grammar(name="long", pattern=r"\d{3}-[A-Z]{2}-\d{2}", examples=["123-AB-45"]),
    ]))


def test_extract_prefers_longer_span_on_same_start():
    hits = extract_citations("see 123-AB-45 today", two_grammar_pack())
    assert [(h.grammar, h.text) for h in hits] == [("long", "123-AB-45")]


def test_extract_finds_multiple_in_order_with_offsets():
    text = "a 111-AA and b 222-BB"
    hits = extract_citations(text, two_grammar_pack())
    assert [h.text for h in hits] == ["111-AA", "222-BB"]
    assert all(text[h.start:h.end] == h.text for h in hits)


def test_extract_empty_and_no_match():
    assert extract_citations("", two_grammar_pack()) == []
    assert extract_citations("nothing here", two_grammar_pack()) == []


def test_extract_is_deterministic():
    p = two_grammar_pack()
    assert extract_citations("x 111-AA y 222-BB-33", p) == extract_citations("x 111-AA y 222-BB-33", p)


def test_pack_without_grammars_extracts_nothing():
    assert extract_citations("123-AB", JurisdictionPack(**pack_dict())) == []
