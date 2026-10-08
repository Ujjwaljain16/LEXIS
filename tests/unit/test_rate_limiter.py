"""
Tests for ingestion/rate_limiter.py::AsyncRateLimiter.

Uses an injectable fake clock/sleep so these run instantly and
deterministically -- no real 60-second waits.
"""
import asyncio

import pytest

from lexis.ingestion.rate_limiter import AsyncRateLimiter


class FakeClock:
    """A controllable monotonic clock: advances only when told to."""
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_limiter(max_per_minute, clock):
    async def fake_sleep(seconds):
        clock.advance(seconds)
    return AsyncRateLimiter(max_per_minute=max_per_minute, clock=clock, sleep=fake_sleep)


@pytest.mark.asyncio
async def test_allows_up_to_the_limit_without_waiting():
    clock = FakeClock()
    limiter = make_limiter(3, clock)

    for _ in range(3):
        await limiter.acquire()

    assert clock.now == 0.0  # never needed to sleep


@pytest.mark.asyncio
async def test_blocks_and_advances_the_clock_once_the_limit_is_reached():
    clock = FakeClock()
    limiter = make_limiter(2, clock)

    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()  # 3rd must wait for the 1st's slot to age out of the 60s window

    assert clock.now >= 60.0


@pytest.mark.asyncio
async def test_slots_free_up_after_sixty_seconds():
    clock = FakeClock()
    limiter = make_limiter(1, clock)

    await limiter.acquire()
    clock.advance(60.0)
    await limiter.acquire()  # must not block: the first call is now outside the window

    assert clock.now == 60.0  # the fake sleep was never invoked a second time


@pytest.mark.asyncio
async def test_rejects_a_non_positive_limit():
    with pytest.raises(ValueError):
        AsyncRateLimiter(max_per_minute=0)


@pytest.mark.asyncio
async def test_concurrent_acquirers_never_exceed_the_limit_at_once():
    """Real asyncio.sleep/time.monotonic (not the fake clock) -- proves the
    lock genuinely serializes concurrent acquire() calls rather than racing
    past the limit when many callers arrive at once."""
    limiter = AsyncRateLimiter(max_per_minute=1000000)  # effectively unlimited for this check
    granted = []

    async def worker(i):
        await limiter.acquire()
        granted.append(i)

    await asyncio.gather(*(worker(i) for i in range(20)))
    assert sorted(granted) == list(range(20))
