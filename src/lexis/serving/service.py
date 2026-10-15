"""
The fast answer path as one injectable unit: retrieve -> number sources -> stream a cited answer ->
validate citations -> (optionally) verify claims.

Every collaborator is passed in, so the whole path is unit-testable with fakes and the route stays a
thin SSE adapter. Event contract (what a client can rely on):

  status        {"stage": "RETRIEVAL" | "SYNTHESIS" | "VERIFYING"}
  context       {"chunk_ids": [...]}                 the chunks actually shown to the model, in order
                                                    (lets an evaluation ask "was the evidence in the prompt?")
  token         {"text": ...}                       live generator output (may contain unresolved
                                                    [n] markers -- display only)
  answer        {"text", "invalid_citation_indices"} authoritative final text, citations validated
  citations     {"citations": [SourceRef...]}        only sources the answer actually cited
  abstained     {"reason": "no_context" | "model_declined"}   instead of answer/citations
  verification  {"support_rate", "claims": [...]}    only when a verifier is configured
  failed        {"error": <exception class name>}    generic on purpose: provider errors can echo
                                                    request details; the full error is logged
  completed     {}                                   always last on success

Retrieval is the evaluated hybrid+RRF engine; passing `document_ids` switches to the document-scoped
variant that measured Recall@30 0.558 -> 0.881 on the dev baseline (README, "Findings").
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, Sequence

from lexis.config import settings
from lexis.generation.grounding import (
    ABSTENTION_MARKER,
    EntailmentChecker,
    SourceRef,
    build_sources,
    is_abstention,
    resolve_citations,
    verify_claims,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerEvent:
    type: str
    data: Dict[str, Any]


class Retriever(Protocol):
    async def retrieve(self, query: str, document_ids: Optional[Sequence[str]]) -> List[dict]: ...


class Generator(Protocol):
    def stream_grounded(self, query: str, sources: List[SourceRef]) -> AsyncIterator[str]: ...


class EngineRetriever:
    """Adapts RetrievalEngine (and its document-scoped variant) to the Retriever protocol."""

    def __init__(self, engine):
        self.engine = engine

    async def retrieve(self, query: str, document_ids: Optional[Sequence[str]]) -> List[dict]:
        k, n = settings.answer_top_k_per_path, settings.answer_top_n_fused
        if document_ids:
            from lexis.retrieval.scoped import retrieve_scoped
            trace = await retrieve_scoped(self.engine, query, list(document_ids), k, n)
            return trace.final_chunks
        return await self.engine.retrieve(query, top_k_per_path=k, top_n_rrf=n)


class AnswerService:
    def __init__(self, retriever: Retriever, generator: Generator, verifier: Optional[EntailmentChecker] = None):
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier

    async def stream(self, query: str, document_ids: Optional[Sequence[str]] = None) -> AsyncIterator[AnswerEvent]:
        try:
            yield AnswerEvent("status", {"stage": "RETRIEVAL"})
            chunks = await self.retriever.retrieve(query, document_ids)
            sources = build_sources(chunks[: settings.answer_context_chunks])
            if not sources:
                yield AnswerEvent("abstained", {"reason": "no_context"})
                yield AnswerEvent("completed", {})
                return

            yield AnswerEvent("context", {"chunk_ids": [s.chunk_id for s in sources]})
            yield AnswerEvent("status", {"stage": "SYNTHESIS"})
            raw_parts: List[str] = []
            pending = ""
            decided = False   # have we seen enough of the stream to know it isn't the abstention marker?
            async for token in self.generator.stream_grounded(query, sources):
                raw_parts.append(token)
                if decided:
                    yield AnswerEvent("token", {"text": token})
                    continue
                # Hold back the first few characters so the abstention marker is never streamed
                # to the user as if it were answer text.
                pending += token
                if len(pending.lstrip()) >= len(ABSTENTION_MARKER):
                    decided = True
                    if not pending.lstrip().startswith(ABSTENTION_MARKER):
                        yield AnswerEvent("token", {"text": pending})
            if not decided and pending and not pending.lstrip().startswith(ABSTENTION_MARKER):
                yield AnswerEvent("token", {"text": pending})

            raw = "".join(raw_parts)
            if is_abstention(raw):
                yield AnswerEvent("abstained", {"reason": "model_declined"})
                yield AnswerEvent("completed", {})
                return

            cleaned, cited, invalid = resolve_citations(raw, sources)
            yield AnswerEvent("answer", {"text": cleaned, "invalid_citation_indices": invalid})
            yield AnswerEvent("citations", {"citations": [s.to_public() for s in cited]})

            if self.verifier is not None:
                yield AnswerEvent("status", {"stage": "VERIFYING"})
                report = await asyncio.to_thread(verify_claims, cleaned, sources, self.verifier)
                yield AnswerEvent("verification", report.to_public())

            yield AnswerEvent("completed", {})
        except Exception as e:  # noqa: BLE001 - boundary: the client gets a generic failure, the log gets the detail
            logger.error("Answer stream failed: %s", e, exc_info=True)
            yield AnswerEvent("failed", {"error": type(e).__name__})
