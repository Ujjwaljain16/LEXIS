"""
API authentication and per-tenant rate limiting.

Auth is fail-closed: with no keys configured every request is refused (503, "auth not configured")
rather than silently accepted -- the previous behaviour fell back to a hardcoded dev key, which is
a credential in the source. Local development opts out explicitly with AUTH_DISABLED=true.
Keys come from LEXIS_API_KEYS="key1:tenantA,key2:tenantB" and are compared in constant time.

The tenant id this returns scopes the rate limit. It does NOT yet scope data: ingestion does not
write a tenant field into Qdrant/Postgres payloads, so isolation between tenants is not
implemented (see README "Honest limitations").
"""
import hmac
import time
from pathlib import Path
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional

from fastapi import HTTPException, Request, status

from lexis.config import settings

ANONYMOUS_TENANT = "anonymous"


def parse_api_keys(raw: str) -> Dict[str, str]:
    """"k1:tenantA,k2:tenantB" -> {"k1": "tenantA", "k2": "tenantB"}. A bare key maps to tenant
    "default". Malformed/empty entries are ignored."""
    keys: Dict[str, str] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, _, tenant = entry.partition(":")
        key = key.strip()
        if key:
            keys[key] = tenant.strip() or "default"
    return keys


def authenticate(provided: Optional[str], configured: Dict[str, str]) -> Optional[str]:
    """Tenant id for a valid key, else None. Constant-time over every configured key so response
    timing does not reveal which prefix matched."""
    if not provided:
        return None
    match: Optional[str] = None
    for key, tenant in configured.items():
        if hmac.compare_digest(provided.encode(), key.encode()):
            match = tenant
    return match


async def require_tenant(request: Request) -> str:
    if settings.auth_disabled:
        return ANONYMOUS_TENANT
    configured = parse_api_keys(settings.lexis_api_keys)
    if not configured:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication is not configured on this server.")
    tenant = authenticate(request.headers.get("X-API-Key"), configured)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return tenant


class SlidingWindowLimiter:
    """At most `limit` hits per `window_s` per key (tenant). In-process, so it is per replica: a
    multi-replica deployment needs a shared store (Redis) for a global limit."""

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def check(self, key: str) -> Optional[float]:
        """None if allowed (and records the hit); otherwise seconds until the next slot frees."""
        now = self._clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return self.window_s - (now - hits[0])
        hits.append(now)
        return None


def rate_limit_dependency(limiter: SlidingWindowLimiter):
    async def _dep(request: Request) -> None:
        tenant = await require_tenant(request)
        retry_after = limiter.check(tenant)
        if retry_after is not None:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded",
                                headers={"Retry-After": str(max(1, int(retry_after + 0.999)))})
    return _dep


def resolve_ingest_path(root: str, file_path: str) -> Path:
    """Resolves a client-supplied path against the ingest root and refuses anything that escapes
    it (absolute paths elsewhere, `..`, symlinks pointing out). Without this, the ingest endpoint
    is an arbitrary-file-read primitive: ingest any file the server can read, then query it."""
    base = Path(root).resolve()
    candidate = (base / file_path).resolve()
    if candidate != base and base not in candidate.parents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file_path must be inside the ingest directory")
    if not candidate.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found in the ingest directory")
    return candidate
