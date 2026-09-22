import asyncio
import uuid
import logging
from lexis.serving.redis_manager import RedisManager
from lexis.serving.worker import IngestionWorker, DeepModeWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_redis_worker")

async def test_worker_infrastructure():
    redis_manager = RedisManager()
    
    # Init workers (they use Dummy processors by default now)
    ingest_worker = IngestionWorker(redis_manager)
    deep_worker = DeepModeWorker(redis_manager)
    
    await ingest_worker.initialize()
    await deep_worker.initialize()

    # 1. Test Successful Ingestion Job
    job_id_success = str(uuid.uuid4())
    logger.info(f"--- 1. Testing Successful Ingestion Job: {job_id_success} ---")
    await redis_manager.enqueue_ingest_job(job_id_success, "docs/sample.pdf", "doc_1")
    
    # 2. Test Failing Ingestion Job (Triggers DLQ and Retries)
    job_id_fail = str(uuid.uuid4())
    logger.info(f"--- 2. Testing Failing Ingestion Job: {job_id_fail} ---")
    # 'fail_me' in path triggers the intentional failure in DummyIngestionProcessor
    await redis_manager.enqueue_ingest_job(job_id_fail, "docs/fail_me.pdf", "doc_2")

    # 3. Test Successful Deep Mode Job
    job_id_deep = str(uuid.uuid4())
    logger.info(f"--- 3. Testing Deep Mode Job: {job_id_deep} ---")
    await redis_manager.enqueue_job(job_id_deep, {"query": "What are the termination conditions?"})

    # Start workers as background tasks to consume the queue
    ingest_task = asyncio.create_task(ingest_worker.run())
    deep_task = asyncio.create_task(deep_worker.run())
    
    # We will subscribe to the deep mode job pubsub to verify token stream
    pubsub = await redis_manager.subscribe(job_id_deep)
    
    logger.info("Workers started. Waiting for logs and PubSub events...")

    # Wait for pubsub messages for a few seconds
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < 15:
        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
        if message:
            logger.info(f"[PUBSUB] Deep Mode Event for {job_id_deep}: {message['data']}")
        
    logger.info("Test finished. Shutting down workers...")
    
    # Cancel tasks
    ingest_task.cancel()
    deep_task.cancel()
    
    await redis_manager.close()

if __name__ == "__main__":
    asyncio.run(test_worker_infrastructure())
