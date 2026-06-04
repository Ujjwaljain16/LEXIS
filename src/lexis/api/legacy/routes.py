from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator

from lexis.retrieval.engine import RetrievalEngine
from lexis.retrieval.reranker import ContextAssembler
from lexis.retrieval.synthesizer import LexisSynthesizer

router = APIRouter()
engine = RetrievalEngine()
assembler = ContextAssembler()
synthesizer = LexisSynthesizer()

class QueryRequest(BaseModel):
    query: str

async def generate_sse(query: str) -> AsyncGenerator[str, None]:
    # 1. Retrieve across 4 concurrent paths
    raw_chunks = await engine.retrieve(query)
    
    # 2. Rerank & Pack safely to 6k tokens
    packed_chunks = assembler.rerank_and_pack(query, raw_chunks)
    
    # 3. Stream Answer via Server-Sent Events
    async for token in synthesizer.stream_answer(query, packed_chunks):
        # SSE format specification requires "data: payload\n\n"
        # We replace newlines in the token to ensure we don't break the SSE protocol prematurely
        safe_token = token.replace("\n", "\\n")
        yield f"data: {safe_token}\n\n"

@router.post("/v1/chat/completions")
async def chat(request: QueryRequest):
    """
    Exposes a streaming endpoint for the LEXIS 4-Path RAG.
    """
    return StreamingResponse(
        generate_sse(request.query),
        media_type="text/event-stream"
    )
