"""
ASGI entrypoint:  uvicorn lexis.serving.app:app --host 0.0.0.0 --port 8000

`create_app()` is a factory so tests (and alternative deployments) can build an isolated app and
override dependencies; `app` is the process-wide instance uvicorn imports.
"""
import logging

from fastapi import FastAPI

from lexis.serving.api import router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    application = FastAPI(
        title="LEXIS",
        version="v2",
        description="Citation-grounded legal-document RAG. Every answer is returned with the exact "
                    "source passages it cites; unsupported answers abstain.",
    )
    application.include_router(router)
    return application


app = create_app()
