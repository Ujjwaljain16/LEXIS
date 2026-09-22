"""
Generic ground-truth mapping utilities shared by benchmark adapters.

These functions know nothing about any specific benchmark's field names or
document structure. They solve two dataset-agnostic problems any
answer-span-style benchmark (CUAD today, possibly others later) needs:

1. Turning an arbitrary external document identifier (a title, a filename,
   a dataset row id -- whatever the benchmark provides) into a stable LEXIS
   doc_id, deterministically and without a hand-maintained per-document
   lookup table.
2. Determining which of LEXIS's own indexed chunks actually contain a given
   ground-truth answer text, so retrieval can be evaluated at chunk
   granularity instead of only at document granularity.
"""
import hashlib
import re
from typing import Iterable, List, Optional, Set

from lexis.indexing.schema import Chunk


def deterministic_document_id(source_identifier: str, prefix: str = "doc") -> str:
    """
    Derives a stable LEXIS-compatible doc_id from an arbitrary benchmark
    document identifier (e.g. a CUAD contract title). Pure function of the
    identifier's content -- the same source_identifier always yields the
    same doc_id, and no benchmark-specific document needs to be listed by
    name anywhere in code.
    """
    digest = hashlib.sha256(source_identifier.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _normalize_whitespace(text: str) -> str:
    """Collapses all whitespace runs to a single space. Chunking re-joins
    sentences with `" ".join(...)` (see ingestion/chunker.py), which does not
    preserve a source document's original whitespace/line-break layout even
    though the actual words are unchanged -- so exact substring matching
    against raw chunk text would spuriously fail without this."""
    return re.sub(r"\s+", " ", text).strip()


def map_answer_texts_to_chunk_ids(answer_texts: Iterable[str], chunks: List[Chunk]) -> Set[str]:
    """
    Determines which of the given chunks contain at least one of the given
    ground-truth answer texts, via whitespace-normalized substring
    containment. This is deterministic and makes no use of any model or
    heuristic ranking -- a chunk either does or does not contain the
    (whitespace-normalized) answer text.

    Returns the set of chunk_ids that contain at least one answer text. An
    empty result means none of the provided chunks could be confirmed to
    contain any of the answer texts -- callers must treat that as "ground
    truth could not be mapped for this case", not as "not relevant".
    """
    normalized_chunks = [(c.chunk_id, _normalize_whitespace(c.raw_content)) for c in chunks]
    matched_chunk_ids: Set[str] = set()

    for answer_text in answer_texts:
        normalized_answer = _normalize_whitespace(answer_text)
        if not normalized_answer:
            continue
        for chunk_id, normalized_content in normalized_chunks:
            if normalized_answer in normalized_content:
                matched_chunk_ids.add(chunk_id)

    return matched_chunk_ids


def normalize_whitespace(text: str) -> str:
    """Public alias for the same whitespace normalization
    map_answer_texts_to_chunk_ids already applies to build ground truth --
    exposed so callers building an offline diagnostic query representation
    from ground-truth text (see evaluation/oracle_query_diagnostic.py) reuse
    the exact same rule instead of inventing a second one."""
    return _normalize_whitespace(text)


def texts_matching_chunk(candidate_texts: Iterable[str], chunk: Chunk) -> List[str]:
    """Of the given candidate ground-truth texts (in their original order),
    returns those that are contained (whitespace-normalized substring match)
    within THIS SPECIFIC chunk's raw_content. A thin, single-chunk view over
    the exact same matching rule map_answer_texts_to_chunk_ids uses -- no
    new matching logic -- so callers can determine which of possibly
    several ground-truth texts is responsible for a given chunk being
    relevant."""
    normalized_chunk = _normalize_whitespace(chunk.raw_content)
    matched = []
    for text in candidate_texts:
        normalized_text = _normalize_whitespace(text)
        if normalized_text and normalized_text in normalized_chunk:
            matched.append(text)
    return matched


def build_answer_derived_query(candidate_texts: Iterable[str], chunk: Chunk) -> Optional[str]:
    """Deterministically joins (in input order, single-space separated) the
    candidate ground-truth texts that match this chunk, for use as an
    offline diagnostic query representation (see
    evaluation/oracle_query_diagnostic.py). Returns None if none match --
    callers must treat that as "no answer-derived query available for this
    chunk" and skip it, never fabricate one."""
    matched = texts_matching_chunk(candidate_texts, chunk)
    if not matched:
        return None
    return " ".join(matched)
