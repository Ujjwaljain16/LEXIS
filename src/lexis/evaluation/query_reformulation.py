"""
Generic, deterministic query-representation transformations for Step 12's
offline query-reformulation diagnostic.

Every transformation here takes ONLY query text as input -- never answer
text, ground-truth chunk/document text, relevance labels, evaluation
metadata, or retrieved results (verified structurally by
tests/unit/test_query_reformulation.py: this module imports nothing from
evaluation/dataset/mapping.py's ground-truth functions, nothing from
retrieval/hybrid_retriever.py, and nothing from lexis.config). No LLM, no
external service, no randomness -- every function is a pure, deterministic
function of its input text.

Two representations are provided:

  normalize_query()   -- Representation B. A thin wrapper around
                          dataset/mapping.py::normalize_whitespace, the
                          project's one existing generic text-normalization
                          rule. Not a new normalization.

  reduce_boilerplate() -- Representation C. Removes sentences that recur,
                          near-verbatim, across a large fraction of the
                          QUERY SET being evaluated (fit_boilerplate_model
                          learns this purely from the queries' own text --
                          never from answers, chunks, or labels). The
                          reasoning: a sentence that appears in most queries
                          in a batch carries no case-specific signal (it is
                          template/instructional filler shared by the whole
                          set), while a sentence that varies per query is
                          more likely to carry the actual, case-specific
                          request. This requires no domain knowledge and no
                          hardcoded phrase list -- it would identify
                          whatever boilerplate exists in ANY templated query
                          set, CUAD or otherwise. Falls back to the original
                          query whenever removal would leave nothing, so it
                          can never delete the only content a query has.

Sentence segmentation reuses the exact regex ingestion/chunker.py already
uses to split contract text into sentences (`re.split(r'(?<=[.!?])\\s+',
...)`) -- not a new segmentation rule invented for this module.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from lexis.evaluation.dataset.mapping import normalize_whitespace

# Identical to ingestion/chunker.py's own sentence-splitting regex (see
# SemanticChunker.chunk's "Step 1: Sentence splitting") -- reused, not
# reinvented, so this module doesn't introduce a second segmentation rule.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_query(query: str) -> str:
    """Representation B: the project's one existing generic whitespace
    normalization, reused verbatim."""
    return normalize_whitespace(query)


def split_sentences(text: str) -> List[str]:
    """Deterministic, punctuation-based sentence segmentation -- the same
    rule ingestion/chunker.py already applies to contract text. Returns []
    for empty/whitespace-only input."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


@dataclass
class BoilerplateModel:
    """Learned ONLY from the text of the queries being evaluated -- never
    from answers, chunks, documents, or relevance labels. sentence_frequency
    counts, for each whitespace-normalized sentence, how many DISTINCT
    queries in the fitted set contained it at least once."""
    sentence_frequency: Dict[str, int]
    num_queries: int
    min_frequency_ratio: float

    def is_boilerplate(self, normalized_sentence: str) -> bool:
        if self.num_queries == 0:
            return False
        frequency = self.sentence_frequency.get(normalized_sentence, 0)
        return (frequency / self.num_queries) >= self.min_frequency_ratio


def fit_boilerplate_model(queries: List[str], min_frequency_ratio: float = 0.5) -> BoilerplateModel:
    """Counts, across the given queries, how many distinct queries contain
    each exact (whitespace-normalized) sentence. A sentence is later
    classified as boilerplate if its frequency ratio meets or exceeds
    min_frequency_ratio (a configurable, non-hardcoded threshold -- not
    baked into this function). Input is exclusively the queries' own text."""
    sentence_frequency: Dict[str, int] = {}
    for query in queries:
        # A set, not a list: a sentence repeated twice WITHIN one query
        # must still only count once towards its cross-query frequency.
        distinct_sentences = {normalize_whitespace(s) for s in split_sentences(query)}
        for sentence in distinct_sentences:
            sentence_frequency[sentence] = sentence_frequency.get(sentence, 0) + 1
    return BoilerplateModel(
        sentence_frequency=sentence_frequency,
        num_queries=len(queries),
        min_frequency_ratio=min_frequency_ratio,
    )


def reduce_boilerplate(query: str, model: BoilerplateModel) -> str:
    """Representation C: removes sentences the given model classifies as
    boilerplate, keeping the rest in original order and rejoining with a
    single space. Falls back to the ORIGINAL query text whenever a query has
    no sentences, or every one of its sentences is classified as
    boilerplate -- this transformation must never return an empty string or
    silently discard the only content a query has."""
    sentences = split_sentences(query)
    if not sentences:
        return query
    kept = [s for s in sentences if not model.is_boilerplate(normalize_whitespace(s))]
    if not kept:
        return query
    return " ".join(kept)


def _word_tokenize(text: str) -> List[str]:
    """Simple, generic whitespace tokenization that keeps punctuation
    attached to its word -- used only to define n-gram WINDOWS for the
    span-level boilerplate detector below. Not a scoring tokenizer (bm25s's
    own tokenizer is used for that elsewhere) and not a new normalization
    rule; this exists solely so a recurring word span can be located and
    removed by position."""
    return text.split()


@dataclass
class NgramBoilerplateModel:
    """A finer-grained sibling of BoilerplateModel: flags a fixed-length
    word n-gram (not a whole sentence) as boilerplate if it recurs,
    verbatim, across at least min_frequency_ratio of the QUERY SET's own
    text. This matters because a template's fixed wording is not always a
    whole sentence -- e.g. "...related to \"X\" that should..." has a fixed
    span both BEFORE and AFTER a varying quoted term in the middle of one
    sentence, which whole-sentence matching can never detect since the
    varying term breaks exact sentence equality. Sliding word windows catch
    a fixed span regardless of what precedes or follows it. Learned
    exclusively from the queries' own text -- never from answers, chunks,
    or labels."""
    ngram_frequency: Dict[Tuple[str, ...], int]
    num_queries: int
    min_frequency_ratio: float
    ngram_length: int

    def is_boilerplate(self, ngram: Tuple[str, ...]) -> bool:
        if self.num_queries == 0:
            return False
        frequency = self.ngram_frequency.get(ngram, 0)
        return (frequency / self.num_queries) >= self.min_frequency_ratio


def fit_ngram_boilerplate_model(queries: List[str], ngram_length: int = 6, min_frequency_ratio: float = 0.5) -> NgramBoilerplateModel:
    """Counts, across the given queries, how many distinct queries contain
    each exact word n-gram of ngram_length (both configurable, not
    hardcoded). Input is exclusively the queries' own text."""
    ngram_frequency: Dict[Tuple[str, ...], int] = {}
    for query in queries:
        words = _word_tokenize(query)
        if len(words) < ngram_length:
            continue
        distinct_ngrams = {tuple(words[i:i + ngram_length]) for i in range(len(words) - ngram_length + 1)}
        for ngram in distinct_ngrams:
            ngram_frequency[ngram] = ngram_frequency.get(ngram, 0) + 1
    return NgramBoilerplateModel(
        ngram_frequency=ngram_frequency, num_queries=len(queries),
        min_frequency_ratio=min_frequency_ratio, ngram_length=ngram_length,
    )


def reduce_ngram_boilerplate(query: str, model: NgramBoilerplateModel) -> str:
    """Removes every word covered by at least one boilerplate-flagged
    n-gram window (windows are merged by simple positional OR-ing, so
    adjacent/overlapping boilerplate windows collapse into one contiguous
    removed span), keeping the rest in original order. Falls back to the
    ORIGINAL query if it is shorter than the model's n-gram length, or if
    removal would leave nothing -- never returns an empty string or
    silently discards the only content a query has."""
    words = _word_tokenize(query)
    n = model.ngram_length
    if len(words) < n:
        return query
    is_boilerplate_word = [False] * len(words)
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        if model.is_boilerplate(ngram):
            for j in range(i, i + n):
                is_boilerplate_word[j] = True
    kept_words = [w for w, flagged in zip(words, is_boilerplate_word) if not flagged]
    if not kept_words:
        return query
    return " ".join(kept_words)


def content_token_ratio(sentence: str) -> float:
    """Analysis-only helper (NOT used to build a retrieval candidate in
    Step 12): the fraction of a sentence's tokens that survive bm25s's
    existing English-stopword tokenization (indexing/bm25_index.py already
    uses stopwords="en" -- reused here, not a new stopword list). A single
    per-sentence signal some single-query-only boilerplate heuristics rely
    on; see run_query_reformulation.py for why this alone did not reliably
    separate boilerplate from substance on the real query set and was
    therefore not used as an active candidate."""
    import bm25s

    if not sentence or not sentence.strip():
        return 0.0
    raw_tokens = bm25s.tokenize([sentence], stopwords=None, return_ids=False, show_progress=False)[0]
    if not raw_tokens:
        return 0.0
    content_tokens = bm25s.tokenize([sentence], stopwords="en", return_ids=False, show_progress=False)[0]
    return len(content_tokens) / len(raw_tokens)
