import uuid
import json
import asyncio
from fastapi import APIRouter, Request, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from lexis.serving.models import BaseLexisResponse, DeepModeEnqueueRequest
from lexis.serving.telemetry import LexisTracer, get_trace_id
from lexis.indexing.pg_client import PostgresClient, CitationReference, DocumentMetadata

router = APIRouter(prefix="/v2", tags=["Query", "Citations"])
pg_client = PostgresClient()

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
    """Executes fast-mode retrieval, streaming SSE responses within 2s SLA."""
    req_id = str(uuid.uuid4())
    trace_id = get_trace_id()

    async def event_generator():
        try:
            with LexisTracer.start_span("fast_mode_query"):
                yield f"event: status\ndata: {json.dumps({'status': 'retrieving', 'trace_id': trace_id})}\n\n"
                await asyncio.sleep(0.5) 
                
                yield f"event: status\ndata: {json.dumps({'status': 'synthesizing', 'trace_id': trace_id})}\n\n"
                
                for token in ["This ", "is ", "a ", "fast ", "response."]:
                    yield f"event: progress\ndata: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0.1)
                
                yield "event: completed\ndata: {}\n\n"
        except asyncio.TimeoutError:
            yield f"event: failed\ndata: {json.dumps({'error': 'FAST_MODE_TIMEOUT'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/query/deep", response_model=BaseLexisResponse, dependencies=[Depends(verify_api_key), Depends(rate_limit_deep)])
async def query_deep_enqueue(req: DeepModeEnqueueRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex}"
    with LexisTracer.start_span("deep_mode_enqueue"):
        pass # enqueue to Redis Streams
        
    return BaseLexisResponse(
        request_id=str(uuid.uuid4()),
        trace_id=get_trace_id(),
        job_id=job_id,
        data={"message": "Job enqueued successfully."}
    )

@router.get("/query/events/{job_id}")
async def query_deep_events(job_id: str, request: Request):
    """SSE Endpoint for Deep Mode progress updates."""
    async def progress_generator():
        states = ["QUEUED", "RUNNING", "RETRIEVAL", "MAP_PHASE", "REDUCE_PHASE", "VERIFYING", "COMPLETED"]
        for state in states:
            if await request.is_disconnected():
                break
            
            event_type = "completed" if state == "COMPLETED" else "progress"
            if state in ["FAILED", "CANCELLED"]:
                event_type = state.lower()
                
            yield f"event: {event_type}\ndata: {json.dumps({'job_id': job_id, 'state': state, 'percent': 100/len(states)})}\n\n"
            await asyncio.sleep(1) 

    return StreamingResponse(progress_generator(), media_type="text/event-stream")

@router.delete("/query/{job_id}", response_model=BaseLexisResponse)
async def cancel_job(job_id: str):
    with LexisTracer.start_span("cancel_job"):
        pass # mark cancelled in Redis
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
