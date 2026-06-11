import asyncio
import logging
import json
import time
from typing import Dict, Any

from lexis.serving.models import JobState
from lexis.serving.telemetry import LexisTracer
from lexis.serving.redis_manager import RedisManager
from lexis.evaluation.cost_ledger import CostLedger, ResearchBudget
from lexis.retrieval.hybrid_retriever import RetrievalEngine
from lexis.generation.context_assembler import ContextAssembler
from lexis.verification.judge_dep import filter_by_elements
from lexis.reranking.map_reduce_filter import map_reduce_deep_mode
from lexis.generation.synthesizer import LexisSynthesizer
from lexis.retrieval.adapters import flatten_research_graph
from lexis.reranking.sentence_window import SentenceWindowExpansion
from lexis.generation.crag_router import route_crag
from lexis.retrieval.interfaces import Candidate
from lexis.indexing.qdrant_client import LexisQdrantClient
from lexis.config import settings

logger = logging.getLogger(__name__)

class DeepModeWorker:
    """
    Standalone worker for consuming Lexis Deep Mode tasks from Redis Streams.
    Provides Consumer Group semantics, DLQ, Telemetry, and Hard Budget Enforcement.
    """
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self.max_retries = 3
        self.engine = RetrievalEngine()
        self.assembler = ContextAssembler()
        self.synthesizer = LexisSynthesizer()
        self.qdrant = LexisQdrantClient()
        self.sentence_expander = SentenceWindowExpansion(
            fetch_chunk_fn=self._fetch_chunk, 
            window_size=1
        )

    async def _fetch_chunk(self, chunk_id: str) -> Candidate:
        records = await self.qdrant.get_points("primary_v2", [chunk_id])
        if records:
            rec = records[0]
            return Candidate(
                chunk_id=rec.id,
                score=1.0,
                source_path=rec.payload.get("source_file", ""),
                metadata=rec.payload,
                content=rec.payload.get("content", "")
            )
        return None

    async def initialize(self):
        """Create the consumer group."""
        await self.redis.init_consumer_group()

    async def run(self):
        """Main polling loop."""
        logger.info("Starting Lexis Deep Mode Worker...")
        await self.initialize()
        
        iteration_count = 0
        while True:
            try:
                iteration_count += 1
                
                # Use XREADGROUP block for 5s
                response = await self.redis.client.xreadgroup(
                    self.redis.group_name, 
                    "worker_1", 
                    {self.redis.stream_key: ">"}, 
                    count=1, 
                    block=5000
                )
                
                if not response:
                    continue
                    
                for stream, msg_list in response:
                    for msg_id, payload in msg_list:
                        job_id = payload.get("job_id")
                        data_str = payload.get("payload")
                        
                        if not job_id or not data_str:
                            await self.redis.ack_message(msg_id)
                            continue
                            
                        data = json.loads(data_str)
                        query = data.get("query", "")
                        
                        # Handle DLQ
                        retries_raw = await self.redis.client.hget(f"job:{job_id}", "retries")
                        retries = int(retries_raw) if retries_raw else 0
                        
                        if retries >= self.max_retries:
                            logger.warning(f"Job {job_id} exceeded retries. Moving to DLQ.")
                            await self.redis.move_to_dlq(job_id, payload)
                            await self.redis.ack_message(msg_id)
                            continue
                            
                        await self.redis.client.hincrby(f"job:{job_id}", "retries", 1)

                        start_t = time.time()
                        
                        # Process Job
                        await self.process_job(job_id, query)
                        
                        # Ack success
                        await self.redis.ack_message(msg_id)
                        
                        job_duration = time.time() - start_t
                        logger.info(f"Job {job_id} completed in {job_duration:.2f}s")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(5)

    async def process_job(self, job_id: str, query: str):
        """Executes the deep mode state machine."""
        
        budget = ResearchBudget(max_cost_usd=0.50, max_tokens=100000)
        ledger = CostLedger(query_id=job_id, budget=budget)
        
        with LexisTracer.start_span("deep_mode_job") as span:
            span.set_attribute("job_id", job_id)
            span.set_attribute("query", query)
            try:
                # QUEUED -> RUNNING
                await self.redis.publish_state(job_id, JobState.RUNNING.value)
                
                # Check cancellation
                if await self.redis.check_cancellation(job_id):
                    await self.redis.publish_state(job_id, JobState.CANCELLED.value)
                    return

                # RUNNING -> MAP_PHASE (Retrieval is bundled into map_phase prep)
                await self.redis.publish_state(job_id, JobState.MAP_PHASE.value)
                
                ledger.start_timer("retrieval")
                # 1. Retrieve & RRF
                candidates = await self.engine.retrieve(
                    query, 
                    top_k_per_path=settings.deep_mode_top_k, 
                    top_n_rrf=settings.deep_mode_rrf_candidates
                )
                ledger.stop_timer("retrieval")
                
                if await self._check_stop_conditions(job_id, ledger): return
                
                # CRAG Routing Fallback
                max_score = candidates[0].get("rrf_score", 0.0) if candidates else 0.0
                if max_score == 0.0 and candidates and "score" in candidates[0]:
                    max_score = candidates[0]["score"]
                candidates = await route_crag(query, candidates, max_score)

                ledger.start_timer("rerank")
                # 2. Rerank Only
                reranked_chunks = self.assembler.rerank_only(query, candidates, top_k=settings.deep_mode_top_k)
                ledger.stop_timer("rerank")
                
                if await self._check_stop_conditions(job_id, ledger): return
                
                ledger.start_timer("verification")
                # 3. JudgeDEP on Top K only
                verified_chunks = await filter_by_elements(query, reranked_chunks)
                ledger.stop_timer("verification")
                
                if await self._check_stop_conditions(job_id, ledger): return

                # 3.5 Sentence Window Expansion
                candidates_to_expand = []
                for c in verified_chunks:
                    c_obj = Candidate(
                        chunk_id=c.get("payload", {}).get("chunk_id", ""),
                        score=c.get("cross_encoder_score", 0.0),
                        source_path=c.get("payload", {}).get("source_file", ""),
                        metadata=c.get("payload", {}),
                        content=c.get("payload", {}).get("content", "")
                    )
                    candidates_to_expand.append(c_obj)
                    
                expanded_candidates = await self.sentence_expander.transform(query, candidates_to_expand)
                
                # Convert back to dict expected by map_reduce
                final_chunks = []
                for ec in expanded_candidates:
                    final_chunks.append({"payload": ec.metadata, "content": ec.content, "chunk_id": ec.chunk_id, "cross_encoder_score": ec.score})

                # MAP_PHASE -> REDUCE_PHASE
                await self.redis.publish_state(job_id, JobState.REDUCE_PHASE.value)
                
                ledger.start_timer("generation")
                # 4. Map Reduce execution
                subqueries = [query] # Simplified for POC
                session = await map_reduce_deep_mode(
                    session_id=job_id,
                    query=query,
                    subqueries=subqueries,
                    chunks=final_chunks,
                    budget=budget,
                    model=settings.llm_model
                )
                ledger.stop_timer("generation")
                
                if await self._check_stop_conditions(job_id, ledger): return
                
                # REDUCE_PHASE -> SYNTHESIS
                await self.redis.publish_state(job_id, JobState.SYNTHESIS.value)
                
                ledger.start_timer("synthesis")
                # 5. Graph Flattening & Final Synthesis
                flattened_context = flatten_research_graph(session)
                
                # We want to yield tokens here to PubSub!
                # LexisSynthesizer stream_answer is an async generator
                async for token in self.synthesizer.stream_answer(query, flattened_context):
                    await self.redis.publish_token(job_id, token)
                ledger.stop_timer("synthesis")
                
                # SYNTHESIS -> COMPLETED
                await self.redis.publish_state(job_id, JobState.COMPLETED.value)
                
                # Final Telemetry
                LexisTracer.record_cost(span, ledger)
                
                # Record to REDIS hash as well
                receipt = ledger.get_receipt()
                await self.redis.client.hset(f"job:{job_id}", mapping={
                    "total_cost_usd": str(receipt.estimated_cost_usd),
                    "retrieval_ms": str(receipt.retrieval_ms),
                    "verification_ms": str(receipt.verification_ms),
                    "generation_ms": str(receipt.generation_ms)
                })

            except Exception as e:
                logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
                LexisTracer.record_error(span, e)
                await self.redis.publish_state(job_id, JobState.FAILED.value)

    async def _check_stop_conditions(self, job_id: str, ledger: CostLedger) -> bool:
        """Checks for cancel or budget exhaustion and transitions state if needed. Returns True if stopped."""
        if await self.redis.check_cancellation(job_id):
            await self.redis.publish_state(job_id, JobState.CANCELLED.value)
            return True
            
        if ledger.is_budget_exceeded():
            logger.warning(f"Job {job_id} hit hard budget stop.")
            await self.redis.publish_state(job_id, JobState.BUDGET_EXCEEDED.value)
            await self.redis.client.hincrby("telemetry:global", "budget_exceeded_count", 1)
            return True
            
        # Optional: 80% warning condition
        if ledger.cost.estimated_cost_usd > (ledger.budget.max_cost_usd * 0.8):
            logger.warning(f"Job {job_id} is at 80% of its budget!")
            # Emitting warning to telemetry or via PubSub if desired
            
        return False

from lexis.ingestion.pipeline import IngestionPipeline
from lexis.indexing.pg_client import PostgresClient

class IngestionWorker:
    """
    Standalone worker for consuming Lexis Ingestion tasks from Redis Streams.
    Provides Consumer Group semantics, Exponential Backoff, DLQ, and Job State tracking.
    """
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self.max_retries = 3
        # Single instance pipeline reused for all jobs (avoiding repeated init costs)
        self.pipeline = IngestionPipeline()
        self.pg = PostgresClient()

    async def initialize(self):
        """Create the consumer group."""
        await self.redis.init_consumer_group()
        await self.pg.initialize_schema()

    async def run(self):
        """Main polling loop."""
        logger.info("Starting Lexis Ingestion Worker...")
        await self.initialize()
        
        while True:
            try:
                # Use XREADGROUP block for 5s
                response = await self.redis.client.xreadgroup(
                    self.redis.group_name, 
                    "ingest_worker_1", 
                    {self.redis.ingest_stream_key: ">"}, 
                    count=1, 
                    block=5000
                )
                
                if not response:
                    continue
                    
                for stream, msg_list in response:
                    for msg_id, payload in msg_list:
                        data_str = payload.get("payload")
                        if not data_str:
                            await self.redis.ack_ingest_message(msg_id)
                            continue
                            
                        data = json.loads(data_str)
                        job_id = data.get("job_id")
                        file_path = data.get("file_path")
                        doc_id = data.get("doc_id")
                        
                        if not job_id or not file_path or not doc_id:
                            await self.redis.ack_ingest_message(msg_id)
                            continue
                            
                        # Handle Retries and DLQ
                        retries_raw = await self.redis.client.hget(f"ingest_job:{job_id}", "retries")
                        retries = int(retries_raw) if retries_raw else 0
                        
                        if retries >= self.max_retries:
                            logger.warning(f"Ingest Job {job_id} exceeded retries. Moving to DLQ.")
                            await self.redis.move_ingest_to_dlq(job_id, payload)
                            await self.pg.update_ingestion_job_state(job_id, "FAILED", "Max retries exceeded.")
                            await self.redis.ack_ingest_message(msg_id)
                            continue
                            
                        if retries > 0:
                            # Exponential backoff (10s, 30s, 90s)
                            backoff = 10 * (3 ** (retries - 1))
                            logger.info(f"Applying backoff of {backoff}s for job {job_id}")
                            await asyncio.sleep(backoff)
                            
                        await self.redis.client.hincrby(f"ingest_job:{job_id}", "retries", 1)

                        start_t = time.time()
                        
                        # Process Job
                        await self.process_job(job_id, file_path, doc_id, is_retry=(retries > 0))
                        
                        # Ack success
                        await self.redis.ack_ingest_message(msg_id)
                        
                        job_duration = time.time() - start_t
                        logger.info(f"Ingest Job {job_id} completed in {job_duration:.2f}s")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ingest worker loop error: {e}")
                await asyncio.sleep(5)

    async def process_job(self, job_id: str, file_path: str, doc_id: str, is_retry: bool):
        """Executes the ingestion state machine via pipeline callback."""
        
        async def progress_callback(state: str):
            logger.info(f"[Job {job_id}] -> {state}")
            await self.pg.update_ingestion_job_state(job_id, state)
            
        try:
            # We are now picking up the job
            await self.pg.update_ingestion_job_state(job_id, "PARSING", increment_retry=is_retry)
            
            await self.pipeline.ingest_document(
                file_path=file_path, 
                doc_id=doc_id, 
                progress_callback=progress_callback
            )
            
        except Exception as e:
            logger.error(f"Error processing ingest job {job_id}: {e}", exc_info=True)
            await self.pg.update_ingestion_job_state(job_id, "FAILED", error_message=str(e))
            # Raise so the outer loop doesn't ACK and it can be retried
            raise
