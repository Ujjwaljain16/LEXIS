import logging
from fastapi import APIRouter, Request

# Import routers from the routes package
from lexis.serving.routes.query import router as query_router
from lexis.serving.routes.ingest import router as ingest_router
from lexis.serving.routes.citation import router as citation_router

logger = logging.getLogger(__name__)

# Main router for the API
router = APIRouter(prefix="/v2")

# Include sub-routers
router.include_router(query_router)
router.include_router(ingest_router)
router.include_router(citation_router)

# Basic health check or diagnostic at root if needed
@router.get("/health", tags=["Diagnostic"])
async def health_check():
    return {"status": "ok", "version": "v2"}
