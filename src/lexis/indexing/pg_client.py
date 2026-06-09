import os
import logging
from typing import Optional
from pydantic import BaseModel
import asyncpg

logger = logging.getLogger(__name__)

class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

class CitationReference(BaseModel):
    pqac_id: str
    document_id: str
    document_version: int
    document_hash: str
    page: int
    bbox: BoundingBox
    text_span: str
    chunk_id: str
    citation_confidence: float = 0.95

class DocumentMetadata(BaseModel):
    document_id: str
    title: str
    pdf_url: str
    version: int

class PostgresClient:
    def __init__(self, dsn: str = None):
        self.dsn = dsn or os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.dsn)

    async def initialize_schema(self):
        await self.connect()
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS citation_references (
                    pqac_id VARCHAR PRIMARY KEY,
                    document_id VARCHAR NOT NULL,
                    document_version INT,
                    document_hash VARCHAR,
                    page INT NOT NULL,
                    x0 FLOAT,
                    y0 FLOAT,
                    x1 FLOAT,
                    y1 FLOAT,
                    text_span TEXT,
                    chunk_id VARCHAR,
                    citation_confidence FLOAT
                );
                
                CREATE TABLE IF NOT EXISTS document_metadata (
                    document_id VARCHAR PRIMARY KEY,
                    title TEXT,
                    pdf_url TEXT,
                    version INT
                );
            ''')

    async def insert_citation(self, citation: CitationReference):
        await self.connect()
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO citation_references 
                (pqac_id, document_id, document_version, document_hash, page, x0, y0, x1, y1, text_span, chunk_id, citation_confidence)
                VALUES (, , , , , , , , , , , )
                ON CONFLICT (pqac_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    page = EXCLUDED.page,
                    x0 = EXCLUDED.x0,
                    y0 = EXCLUDED.y0,
                    x1 = EXCLUDED.x1,
                    y1 = EXCLUDED.y1,
                    text_span = EXCLUDED.text_span,
                    chunk_id = EXCLUDED.chunk_id;
            ''', 
            citation.pqac_id, citation.document_id, citation.document_version, citation.document_hash,
            citation.page, citation.bbox.x0, citation.bbox.y0, citation.bbox.x1, citation.bbox.y1,
            citation.text_span, citation.chunk_id, citation.citation_confidence)

    async def get_citation(self, pqac_id: str) -> Optional[CitationReference]:
        try:
            await self.connect()
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM citation_references WHERE pqac_id = ", pqac_id)
                if row:
                    return CitationReference(
                        pqac_id=row['pqac_id'],
                        document_id=row['document_id'],
                        document_version=row['document_version'] or 1,
                        document_hash=row['document_hash'] or "",
                        page=row['page'],
                        bbox=BoundingBox(x0=row['x0'], y0=row['y0'], x1=row['x1'], y1=row['y1']),
                        text_span=row['text_span'],
                        chunk_id=row['chunk_id'],
                        citation_confidence=row['citation_confidence'] or 0.95
                    )
        except Exception as e:
            logger.warning(f"Best-effort citation lookup failed for {pqac_id}: {e}")
        return None

    async def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        try:
            await self.connect()
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM document_metadata WHERE document_id = ", document_id)
                if row:
                    return DocumentMetadata(**dict(row))
        except Exception as e:
            logger.warning(f"Best-effort document lookup failed for {document_id}: {e}")
        return None
