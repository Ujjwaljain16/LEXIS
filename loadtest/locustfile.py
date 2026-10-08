"""Load test for /v2/query/fast (3:1 fast:health mix is not meaningful here; deep mode is not load-tested).

    pip install -e ".[dev]"
    LEXIS_API_KEY=... locust -f loadtest/locustfile.py --host https://<your-deployment> \
        --users 5 --spawn-rate 1 --run-time 3m --headless --csv loadtest/results

Reports time-to-first-token (the first SSE `token` event) and total stream time per request as
separate Locust entries. No results are committed yet: run it against a real deployment, not localhost
with fakes, and publish the CSVs with the commit SHA and hardware.
"""
import json
import os
import time

from locust import HttpUser, between, task

QUERIES = [
    "What is the governing law of this agreement?",
    "When does the agreement expire?",
    "Is there a cap on liability?",
    "Can either party terminate for convenience?",
]
DOCUMENT_IDS = [d for d in os.getenv("LEXIS_DOCUMENT_IDS", "").split(",") if d]


class AnswerUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def ask(self):
        body = {"query": QUERIES[int(time.time()) % len(QUERIES)], "document_ids": DOCUMENT_IDS or None}
        headers = {"X-API-Key": os.environ["LEXIS_API_KEY"]}
        start = time.perf_counter()
        first_token = None
        with self.client.post("/v2/query/fast", json=body, headers=headers, stream=True,
                              catch_response=True, name="fast (total)") as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            event = None
            for raw in resp.iter_lines():
                line = raw.decode() if isinstance(raw, bytes) else raw
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: ") and event == "token" and first_token is None:
                    first_token = time.perf_counter() - start
                elif line.startswith("data: ") and event == "failed":
                    resp.failure(json.loads(line[6:]).get("error", "failed"))
                    return
            resp.success()
        if first_token is not None:
            self.environment.events.request.fire(
                request_type="SSE", name="fast (time to first token)", response_time=first_token * 1000,
                response_length=0, exception=None, context={})
