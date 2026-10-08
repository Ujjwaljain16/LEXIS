"""
ASGI entrypoint:  uvicorn lexis.serving.app:app --host 0.0.0.0 --port 8000

`create_app()` is a factory so tests (and alternative deployments) can build an isolated app and
override dependencies; `app` is the process-wide instance uvicorn imports.
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from lexis.serving.api import router

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    application = FastAPI(
        title="LEXIS",
        version="v2",
        description="Citation-grounded legal-document RAG. Every answer is returned with the exact "
                    "source passages it cites; unsupported answers abstain.",
    )
    application.include_router(router)

    @application.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    return application


app = create_app()
