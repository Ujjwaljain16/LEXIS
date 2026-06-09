import os
import json
import logging
from typing import Dict, Any, Optional, List
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class RedisManager:
    """
    Manages Redis Connections for:
    1. Redis Streams (Task Queues & DLQ)
    2. Redis PubSub (Real-time SSE events)
    3. Redis Hash (Job State)
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = os.getenv("REDIS_URL", redis_url)
        self.client = redis.from_url(self.redis_url, decode_responses=True)
        # Streams
        self.stream_key = "lexis_deep_research_queue"
        self.dlq_key = "lexis_deep_research_dlq"
        self.ingest_stream_key = "lexis:ingest:tasks"
        self.ingest_dlq_key = "lexis:ingest:dlq"
        self.group_name = "lexis_workers"

    async def init_consumer_group(self):
        """Idempotent creation of Consumer Group."""
        try:
            await self.client.xgroup_create(self.stream_key, self.group_name, id="0", mkstream=True)
            logger.info(f"Consumer group '{self.group_name}' initialized for query stream.")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e): raise e

        try:
            await self.client.xgroup_create(self.ingest_stream_key, self.group_name, id="0", mkstream=True)
            logger.info(f"Consumer group '{self.group_name}' initialized for ingest stream.")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e): raise e

    async def enqueue_job(self, job_id: str, payload: Dict[str, Any]) -> str:
        """Push query job to stream and init state."""
        await self.client.hset(f"job:{job_id}", mapping={"status": "QUEUED", "retries": 0, "cancelled": "0"})
        msg_id = await self.client.xadd(self.stream_key, {"job_id": job_id, "payload": json.dumps(payload)})
        return msg_id

    async def enqueue_ingest_job(self, job_id: str, file_path: str, doc_id: str) -> str:
        """Push ingest job to ingest stream."""
        await self.client.hset(f"ingest_job:{job_id}", mapping={"retries": 0})
        payload = {"job_id": job_id, "file_path": file_path, "doc_id": doc_id}
        msg_id = await self.client.xadd(self.ingest_stream_key, {"payload": json.dumps(payload)})
        return msg_id

    async def publish_state(self, job_id: str, state: str):
        """Update hash and emit PubSub event."""
        await self.client.hset(f"job:{job_id}", "status", state)
        await self.client.publish(f"job_events:{job_id}", json.dumps({"type": "state_change", "state": state}))
        
    async def publish_token(self, job_id: str, token: str):
        """Emit PubSub event for raw token."""
        await self.client.publish(f"job_events:{job_id}", json.dumps({"type": "token", "content": token}))

    async def check_cancellation(self, job_id: str) -> bool:
        """Check if user cancelled job."""
        val = await self.client.hget(f"job:{job_id}", "cancelled")
        return val == "1"

    async def cancel_job(self, job_id: str):
        await self.client.hset(f"job:{job_id}", "cancelled", "1")

    async def ack_message(self, msg_id: str):
        await self.client.xack(self.stream_key, self.group_name, msg_id)
        
    async def ack_ingest_message(self, msg_id: str):
        await self.client.xack(self.ingest_stream_key, self.group_name, msg_id)

    async def move_to_dlq(self, job_id: str, payload: Dict[str, Any]):
        """Move unprocessable query job to DLQ."""
        await self.client.xadd(self.dlq_key, {"job_id": job_id, "payload": json.dumps(payload), "reason": "max_retries"})
        await self.publish_state(job_id, "FAILED")

    async def move_ingest_to_dlq(self, job_id: str, payload: Dict[str, Any]):
        """Move unprocessable ingest job to DLQ."""
        await self.client.xadd(self.ingest_dlq_key, {"job_id": job_id, "payload": json.dumps(payload), "reason": "max_retries"})

    async def subscribe(self, job_id: str):
        """Returns a pubsub object subscribed to the job events."""
        pubsub = self.client.pubsub()
        await pubsub.subscribe(f"job_events:{job_id}")
        return pubsub

    async def close(self):
        await self.client.aclose()
