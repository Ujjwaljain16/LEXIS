import uuid
import logging
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lexis.serving.models import BaseLexisResponse
from lexis.serving.telemetry import LexisTracer, get_trace_id
from lexis.serving.redis_manager import RedisManager
from lexis.indexing.pg_client import PostgresClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])
pg_client = PostgresClient()

redis_manager = None
def get_redis():
    global redis_manager
    if redis_manager is None:
        redis_manager = RedisManager()
    return redis_manager

class IngestJobRequest(BaseModel):
    file_path: str
    doc_id: str

class BatchIngestRequest(BaseModel):
    files: List[IngestJobRequest]

@router.post("/", response_model=BaseLexisResponse, status_code=202)
async def ingest_document_job(req: IngestJobRequest):
    job_id = f"ingest_{uuid.uuid4().hex}"
    with LexisTracer.start_span("ingest_enqueue"):
        await pg_client.create_ingestion_job(job_id, req.doc_id)
        await get_redis().enqueue_ingest_job(job_id, req.file_path, req.doc_id)
        
    return BaseLexisResponse(
        request_id=str(uuid.uuid4()),
        trace_id=get_trace_id(),
        job_id=job_id,
        data={"message": "Ingestion job enqueued successfully.", "doc_id": req.doc_id}
    )

@router.post("/batch", response_model=BaseLexisResponse, status_code=202)
async def batch_ingest_documents(req: BatchIngestRequest):
    job_ids = []
    with LexisTracer.start_span("batch_ingest_enqueue"):
        for file_req in req.files:
            job_id = f"ingest_{uuid.uuid4().hex}"
            await pg_client.create_ingestion_job(job_id, file_req.doc_id)
            await get_redis().enqueue_ingest_job(job_id, file_req.file_path, file_req.doc_id)
            job_ids.append(job_id)
            
    return BaseLexisResponse(
        request_id=str(uuid.uuid4()),
        trace_id=get_trace_id(),
        job_id="batch",
        data={"message": f"Enqueued {len(job_ids)} ingestion jobs successfully.", "job_ids": job_ids}
    )
