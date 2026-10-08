import logging
from fastapi import APIRouter, HTTPException

from lexis.indexing.pg_client import PostgresClient, CitationReference, DocumentMetadata

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["Citations", "Documents"])
pg_client = PostgresClient()

@router.get("/citations/{pqac_id}", response_model=CitationReference)
async def get_citation(pqac_id: str):
    citation = await pg_client.get_citation(pqac_id)
    if not citation:
        raise HTTPException(status_code=404, detail="Citation not found")
    return citation

@router.get("/documents/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id: str):
    doc = await pg_client.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc
