import uuid
import json
import asyncio
import logging
from fastapi import APIRouter, Request, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from lexis.config import settings
from lexis.serving.models import AnswerRequest, BaseLexisResponse, DeepModeEnqueueRequest, JobState
from lexis.serving.telemetry import LexisTracer, get_trace_id
from lexis.serving.redis_manager import RedisManager

from lexis.retrieval.hybrid_retriever import RetrievalEngine
from lexis.generation.synthesizer import LexisSynthesizer
from lexis.serving.security import SlidingWindowLimiter, rate_limit_dependency
from lexis.serving.service import AnswerEvent, AnswerService, EngineRetriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])

_answer_service = None


def get_answer_service() -> AnswerService:
    """Process-wide service. Built lazily (the embedder and, if enabled, reranker load on first
    use); tests and alternative deployments replace it with app.dependency_overrides."""
    global _answer_service
    if _answer_service is None:
        _answer_service = AnswerService(
            retriever=EngineRetriever(RetrievalEngine()),
            generator=LexisSynthesizer(),
            verifier=_build_verifier(),
        )
    return _answer_service


def _build_verifier():
    if not settings.answer_verify_claims:
        return None
    from lexis.evaluation.nli_checker import NLIChecker
    return NLIChecker()


redis_manager = None
def get_redis():
    global redis_manager
    if redis_manager is None:
        redis_manager = RedisManager()
    return redis_manager


_fast_limiter = SlidingWindowLimiter(settings.rate_limit_fast_per_window, settings.rate_limit_window_s)
_deep_limiter = SlidingWindowLimiter(settings.rate_limit_deep_per_window, settings.rate_limit_window_s)
rate_limit_fast = rate_limit_dependency(_fast_limiter)
rate_limit_deep = rate_limit_dependency(_deep_limiter)


def _sse(event: AnswerEvent) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"


@router.post("/fast", dependencies=[Depends(rate_limit_fast)])
async def query_fast(req: AnswerRequest, service: AnswerService = Depends(get_answer_service)):
    """Cited, streamed answer (SSE). Event contract: see lexis/serving/service.py."""
    trace_id = get_trace_id()

    async def event_generator():
        with LexisTracer.start_span("fast_mode_query"):
            yield f"event: trace\ndata: {json.dumps({'trace_id': trace_id})}\n\n"
            async for event in service.stream(req.query, req.document_ids):
                yield _sse(event)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/deep", response_model=BaseLexisResponse, dependencies=[Depends(rate_limit_deep)])
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

@router.get("/events/{job_id}")
async def query_deep_events(job_id: str, request: Request):
    """SSE Endpoint for Deep Mode progress updates streaming from Redis PubSub."""
    async def progress_generator():
        pubsub = await get_redis().subscribe(job_id)
        logger.info(f"SSE Client subscribed to job_events:{job_id}")
        try:
            while True:
                if await request.is_disconnected():
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

    return StreamingResponse(progress_generator(), media_type="text/event-stream")

@router.delete("/{job_id}", response_model=BaseLexisResponse)
async def cancel_job(job_id: str):
    with LexisTracer.start_span("cancel_job"):
        await get_redis().cancel_job(job_id)
    return BaseLexisResponse(
        request_id=str(uuid.uuid4()), trace_id=get_trace_id(), job_id=job_id,
        data={"message": "Job cancellation requested."}
    )
