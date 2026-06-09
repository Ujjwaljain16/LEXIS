from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator, Optional
import uuid

from lexis.retrieval.engine import RetrievalEngine
from lexis.retrieval.reranker import ContextAssembler
from lexis.retrieval.synthesizer import LexisSynthesizer
from lexis.config import settings

# Deep Mode Imports
from lexis.verification.judge_dep import filter_by_elements
from lexis.reranking.sentence_window import SentenceWindowExpansion
from lexis.reranking.map_reduce_filter import map_reduce_deep_mode, ResearchBudget
from lexis.retrieval.adapters import dict_to_candidate, candidate_to_dict, flatten_research_graph
from lexis.retrieval.interfaces import Candidate

router = APIRouter()
engine = RetrievalEngine()
assembler = ContextAssembler()
synthesizer = LexisSynthesizer()

# Initialize SentenceWindowExpansion
# For Deep Mode, fetch_chunk_fn needs to fetch from Qdrant/ES.
# For MVP, we'll just mock it or provide a dummy fetch_chunk_fn if we don't have a direct fetch API yet.
# Wait, SentenceWindow requires a fetch_chunk_fn.
async def dummy_fetch_chunk(chunk_id: str) -> Optional[Candidate]:
    # Placeholder: In a real system, this would query Qdrant by ID.
    return None

sentence_window = SentenceWindowExpansion(fetch_chunk_fn=dummy_fetch_chunk, window_size=1)

class QueryRequest(BaseModel):
    query: str
    strategy: str = "fast"  # "fast" or "deep"

async def generate_sse(query: str, strategy: str) -> AsyncGenerator[str, None]:
    if strategy == "deep":
        # 1. Retrieve & RRF Trim to configuration limit
        raw_chunks = await engine.retrieve(query, top_n_rrf=settings.deep_mode_rrf_candidates)
        
        # 2. JudgeDEP Verification
        verified_chunks = await filter_by_elements(query, raw_chunks)
        
        # 3. CrossEncoder Trim to Top K
        top_k_chunks = assembler.rerank_only(query, verified_chunks, top_k=settings.deep_mode_top_k)
        
        # 4. SentenceWindowExpansion
        candidates = [dict_to_candidate(c) for c in top_k_chunks]
        expanded_candidates = await sentence_window.transform(query, candidates)
        expanded_chunks = [candidate_to_dict(c) for c in expanded_candidates]
        
        # 5. MapReduce Filter
        session_id = str(uuid.uuid4())
        budget = ResearchBudget(max_tokens=8000, max_duration_sec=30)
        research_session = await map_reduce_deep_mode(
            session_id=session_id,
            query=query,
            subqueries=[],  # Can be expanded later
            chunks=expanded_chunks,
            budget=budget
        )
        
        # 6. Flatten ResearchGraph
        packed_chunks = flatten_research_graph(research_session)
        
    else:
        # Fast Mode (Unchanged)
        # 1. Retrieve across 4 concurrent paths
        raw_chunks = await engine.retrieve(query)
        
        # 2. Rerank & Pack safely to 6k tokens
        packed_chunks = assembler.rerank_and_pack(query, raw_chunks)
        
    # Stream Answer via Server-Sent Events
    async for token in synthesizer.stream_answer(query, packed_chunks):
        safe_token = token.replace("\n", "\\n")
        yield f"data: {safe_token}\n\n"

@router.post("/v1/chat/completions")
async def chat(request: QueryRequest):
    """
    Exposes a streaming endpoint for the LEXIS 4-Path RAG.
    Supports strategy="fast" and strategy="deep".
    """
    return StreamingResponse(
        generate_sse(request.query, request.strategy),
        media_type="text/event-stream"
    )
