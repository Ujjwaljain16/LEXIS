"""
Sliding-window async rate limiter.

Rationale: a live HyPE ingestion run against Gemini's free tier (20
requests/minute for gemini-2.5-flash) showed that bounding CONCURRENCY
alone (a semaphore) is not the same as bounding RATE -- if calls complete
quickly, a small concurrency limit still lets far more than 20 requests
start within a minute, and nearly all of them get rejected with
RateLimitError. This tracks actual call start times in a trailing 60s
window and blocks acquire() until there is real headroom.
"""
import asyncio
import time
from collections import deque
from typing import Any, Awaitable, Callable, Deque


class AsyncRateLimiter:
    def __init__(self, max_per_minute: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep):
        if max_per_minute <= 0:
            raise ValueError("max_per_minute must be positive")
        self._max = max_per_minute
        self._clock = clock
        self._sleep = sleep
        self._timestamps: Deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Blocks until fewer than max_per_minute acquire() calls have
        succeeded in the trailing 60 seconds, then reserves a slot."""
        while True:
            async with self._lock:
                now = self._clock()
                # 60 here is a unit constant (seconds per minute, matching this class's
                # "per minute" contract), not a tunable threshold -- there is no config knob
                # that would make sense for it, unlike e.g. max_per_minute itself.
                while self._timestamps and now - self._timestamps[0] >= 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return
                wait_for = 60 - (now - self._timestamps[0])
            await self._sleep(max(wait_for, 0.01))
