"""
Jurisdiction Pack schema (plan section 3.1, extension point 1).

A pack is DATA describing everything about a legal system / corpus family
that the core pipeline must not hardcode: languages and BM25 tokenization,
citation grammars, document structure and chunking policy, prompts and
refusal text, trusted web sources, court hierarchy, temporal/territorial
rules, validity-signal phrases, model choices (by registry key), evaluation
datasets, and licence/compliance obligations.

Validation is strict and self-testing: every citation grammar must compile,
fully match each of its own `examples`, and not match any `non_examples`,
so a mistyped pattern fails at load time rather than silently mis-parsing
citations in production.
"""
import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

_SLUG = r"^[a-z][a-z0-9_]*$"


class CitationGrammar(BaseModel):
    name: str = Field(min_length=1)
    pattern: str = Field(min_length=1)
    examples: List[str] = Field(min_length=1)
    non_examples: List[str] = Field(default_factory=list)
    description: str = ""

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"pattern does not compile: {e}") from e
        return v

    @model_validator(mode="after")
    def _self_test(self) -> "CitationGrammar":
        rx = re.compile(self.pattern)
        for ex in self.examples:
            if not rx.fullmatch(ex):
                raise ValueError(f"grammar {self.name!r}: example {ex!r} does not fully match its own pattern")
        for bad in self.non_examples:
            if rx.search(bad):
                raise ValueError(f"grammar {self.name!r}: non-example {bad!r} unexpectedly matches")
        return self


class BM25Config(BaseModel):
    lowercase: bool = True
    stopwords: Optional[str] = None      # stopword language code understood by the BM25 backend, or none
    stemmer: Optional[str] = None
    token_pattern: Optional[str] = None
    segmenter: Optional[str] = None      # e.g. a CJK segmenter name, resolved by the tokenizer factory

    @field_validator("token_pattern")
    @classmethod
    def _compiles(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"token_pattern does not compile: {e}") from e
        return v


class StructureSpec(BaseModel):
    levels: List[str] = Field(default_factory=list)   # e.g. hierarchy from largest to smallest unit
    chunk_strategy: Literal["semantic", "clause", "paragraph", "section"] = "semantic"
    max_chunk_tokens: Optional[int] = Field(default=None, gt=0)
    parent_context_template: str = ""                  # e.g. "{title} > {path}" injected before embedding


class PromptSet(BaseModel):
    system: str = Field(min_length=1)
    refusal: str = Field(min_length=1)
    insufficient_marker: str = Field(min_length=1)     # sentinel the model emits when context is insufficient


class TemporalSpec(BaseModel):
    as_of_supported: bool = False
    territories: List[str] = Field(default_factory=list)


class ModelRefs(BaseModel):
    """Registry KEYS (see registry/models.py), never hub identifiers."""
    embedder: Optional[str] = None
    reranker: Optional[str] = None
    nli: Optional[str] = None
    llm: Optional[str] = None


class Compliance(BaseModel):
    source: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    blocks_redistribution: bool = False
    blocks_commercial_use: bool = False


class JurisdictionPack(BaseModel):
    name: str = Field(pattern=_SLUG)
    description: str = ""
    languages: List[str] = Field(min_length=1)
    bm25: BM25Config = Field(default_factory=BM25Config)
    citation_grammars: List[CitationGrammar] = Field(default_factory=list)
    structure: StructureSpec = Field(default_factory=StructureSpec)
    feature_schema_ref: Optional[str] = None
    prompts: PromptSet
    trusted_sources: Dict[str, float] = Field(default_factory=dict)
    court_hierarchy: List[str] = Field(default_factory=list)
    temporal: TemporalSpec = Field(default_factory=TemporalSpec)
    validity_phrases: Dict[str, List[str]] = Field(default_factory=dict)
    models: ModelRefs = Field(default_factory=ModelRefs)
    corpus_adapters: List[str] = Field(default_factory=list)
    eval_datasets: List[str] = Field(default_factory=list)
    compliance: List[Compliance] = Field(default_factory=list)

    @field_validator("languages")
    @classmethod
    def _lower_languages(cls, v: List[str]) -> List[str]:
        return [x.lower() for x in v]

    @field_validator("trusted_sources")
    @classmethod
    def _weights_in_unit_interval(cls, v: Dict[str, float]) -> Dict[str, float]:
        bad = {k: w for k, w in v.items() if not 0.0 <= w <= 1.0}
        if bad:
            raise ValueError(f"trusted source weights must be within [0, 1]: {bad}")
        return v

    @model_validator(mode="after")
    def _unique_grammar_names(self) -> "JurisdictionPack":
        names = [g.name for g in self.citation_grammars]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate citation grammar names: {dupes}")
        return self
