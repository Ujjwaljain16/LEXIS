import logging
from fastapi import APIRouter, Depends

# Import routers from the routes package
from lexis.serving.routes.query import router as query_router
from lexis.serving.routes.ingest import router as ingest_router
from lexis.serving.routes.citation import router as citation_router
from lexis.serving.security import require_tenant

logger = logging.getLogger(__name__)

# Main router for the API
router = APIRouter(prefix="/v2")

# Include sub-routers
# Everything except /health requires a valid API key (fail-closed; see serving/security.py).
_authed = [Depends(require_tenant)]
router.include_router(query_router, dependencies=_authed)
router.include_router(ingest_router, dependencies=_authed)
router.include_router(citation_router, dependencies=_authed)

# Basic health check or diagnostic at root if needed
@router.get("/health", tags=["Diagnostic"])
async def health_check():
    return {"status": "ok", "version": "v2"}
