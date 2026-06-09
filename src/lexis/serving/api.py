import uuid
import json
import asyncio
import logging
from fastapi import APIRouter, Request, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from lexis.serving.models import BaseLexisResponse, DeepModeEnqueueRequest, JobState
from lexis.serving.telemetry import LexisTracer, get_trace_id
from lexis.serving.redis_manager import RedisManager
from lexis.indexing.pg_client import PostgresClient, CitationReference, DocumentMetadata

from lexis.retrieval.engine import RetrievalEngine
from lexis.retrieval.reranker import ContextAssembler
from lexis.retrieval.synthesizer import LexisSynthesizer

engine_fast = None
assembler_fast = None
synthesizer_fast = None

def get_fast_components():
    global engine_fast, assembler_fast, synthesizer_fast
    if engine_fast is None:
        engine_fast = RetrievalEngine()
        assembler_fast = ContextAssembler()
        synthesizer_fast = LexisSynthesizer()
    return engine_fast, assembler_fast, synthesizer_fast

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2", tags=["Query", "Citations"])
pg_client = PostgresClient()

# Initialize Redis Manager globally for the router, connecting to local by default
# In production, this would use FastAPI lifespan events to manage the connection pool
redis_manager = None
def get_redis():
    global redis_manager
    if redis_manager is None:
        redis_manager = RedisManager()
    return redis_manager

# Security & Rate Limiting Mock Layer
async def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-Key")
    # if not api_key: raise HTTPException(status_code=401, detail="Unauthorized")
    return "tenant_123"

async def rate_limit_fast():
    # token bucket check for 10/min
    pass

async def rate_limit_deep():
    # concurrency check max 2 active deep jobs
    pass

@router.post("/query/fast", dependencies=[Depends(verify_api_key), Depends(rate_limit_fast)])
async def query_fast(req: DeepModeEnqueueRequest, request: Request):
    """Executes fast-mode retrieval, streaming SSE responses via real retrieval stack."""
    trace_id = get_trace_id()
    eng, asm, syn = get_fast_components()

    async def event_generator():
        try:
            with LexisTracer.start_span("fast_mode_query"):
                yield f"event: status\ndata: {json.dumps({'status': 'RETRIEVAL', 'trace_id': trace_id})}\n\n"
                # Retrieve & RRF
                candidates = await eng.retrieve(req.query, top_k_per_path=5, top_n_rrf=15)
                
                yield f"event: status\ndata: {json.dumps({'status': 'RERANK_PHASE', 'trace_id': trace_id})}\n\n"
                # Rerank
                reranked_chunks = asm.rerank_only(req.query, candidates, top_k=5)
                
                yield f"event: status\ndata: {json.dumps({'status': 'SYNTHESIS', 'trace_id': trace_id})}\n\n"
                # Flatten simple context
                context_parts = []
                for chunk in reranked_chunks:
                    p = chunk.get("payload", {})
                    context_parts.append(f"[{p.get('pqac_key', 'unknown')}] {p.get('content', '')}")
                context_str = "\n\n".join(context_parts)
                
                # Stream Synthesis
                async for token in syn.stream_answer(req.query, context_str):
                    yield f"event: progress\ndata: {json.dumps({'token': token})}\n\n"
                
                yield "event: completed\ndata: {}\n\n"
        except Exception as e:
            logger.error(f"Fast Mode Error: {e}", exc_info=True)
            yield f"event: failed\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/query/deep", response_model=BaseLexisResponse, dependencies=[Depends(verify_api_key), Depends(rate_limit_deep)])
async def query_deep_enqueue(req: DeepModeEnqueueRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex}"
    with LexisTracer.start_span("deep_mode_enqueue"):
        payload = {
            "query": req.query,
            "metadata_filters": req.metadata_filters,
        }
        await get_redis().enqueue_job(job_id, payload)
        
    return BaseLexisResponse(
        request_id=str(uuid.uuid4()),
        trace_id=get_trace_id(),
        job_id=job_id,
        data={"message": "Job enqueued successfully."}
    )

@router.get("/query/events/{job_id}")
async def query_deep_events(job_id: str, request: Request):
    """SSE Endpoint for Deep Mode progress updates streaming from Redis PubSub."""
    
    async def progress_generator():
        pubsub = await get_redis().subscribe(job_id)
        logger.info(f"SSE Client subscribed to job_events:{job_id}")
        
        try:
            while True:
                if await request.is_disconnected():
                    logger.info(f"SSE Client disconnected for job {job_id}")
                    break

                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    data_str = message.get("data")
                    if data_str:
                        data = json.loads(data_str)
                        msg_type = data.get("type")
                        
                        if msg_type == "state_change":
                            state = data.get("state")
                            event_type = "status"
                            if state == JobState.COMPLETED.value:
                                event_type = "completed"
                            elif state in [JobState.FAILED.value, JobState.CANCELLED.value, JobState.BUDGET_EXCEEDED.value]:
                                event_type = "failed"
                                
                            yield f"event: {event_type}\ndata: {json.dumps({'job_id': job_id, 'state': state})}\n\n"
                            
                            # Terminal states
                            if state in [JobState.COMPLETED.value, JobState.FAILED.value, JobState.CANCELLED.value, JobState.BUDGET_EXCEEDED.value]:
                                break
                        
                        elif msg_type == "token":
                            token = data.get("content")
                            yield f"event: progress\ndata: {json.dumps({'token': token})}\n\n"
                            
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in SSE stream for job {job_id}: {e}")
            yield f"event: failed\ndata: {json.dumps({'error': 'SSE_STREAM_ERROR'})}\n\n"
        finally:
            await pubsub.unsubscribe(f"job_events:{job_id}")
            # Do NOT close the redis_manager.client here since it's a global pool.

    return StreamingResponse(progress_generator(), media_type="text/event-stream")

@router.delete("/query/{job_id}", response_model=BaseLexisResponse)
async def cancel_job(job_id: str):
    with LexisTracer.start_span("cancel_job"):
        await get_redis().cancel_job(job_id)
    return BaseLexisResponse(
        request_id=str(uuid.uuid4()), trace_id=get_trace_id(), job_id=job_id,
        data={"message": "Job cancellation requested."}
    )

# --- PHASE C1: Citation Projection API ---

@router.get("/citations/{pqac_id}", response_model=CitationReference)
async def get_citation(pqac_id: str):
    citation = await pg_client.get_citation(pqac_id)
    if not citation:
        raise HTTPException(status_code=404, detail="Citation not found")
    return citation

@router.get("/documents/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id: str):
    doc = await pg_client.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


