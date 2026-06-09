import json
import asyncio
from fastapi.testclient import TestClient
from lexis.serving.api import router, query_fast_events

client = TestClient(router)

async def test_fast_mode():
    print("[Test] 1. Posting to Fast Mode...")
    response = client.post(
        "/v2/query/fast",
        json={"query": "Termination for cause"},
        headers={"X-API-Key": "test"}
    )
    
    assert response.status_code == 200
    job_id = response.json().get("job_id")
    print(f"✅ Fast Job Created: {job_id}")

    class MockRequest:
        async def is_disconnected(self): return False

    generator = await query_fast_events(job_id, MockRequest())
    
    print("\n--- FAST MODE SSE ---")
    async for raw_event in generator.body_iterator:
        print(raw_event.strip())
        
if __name__ == "__main__":
    asyncio.run(test_fast_mode())
