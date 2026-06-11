"""
Ingestion Pipeline Orchestrator for LEXIS.

Rationale: The central nervous system of document ingestion. Wires together all independent modules.
Source Inspiration: RAGFlow pipeline and plan.md architectural workflow.
Deviations from Source Repos: Strictly uses asyncio for feature extraction to parallelize LLM IO. 
Expected Impact on Metrics: Handles end-to-end ingestion idempotently.
"""
import asyncio
import uuid
import hashlib
from typing import List
from qdrant_client.http import models

from lexis.indexing.schema import Chunk, ChunkMetadata
from lexis.ingestion.parser import LexisParser
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.ingestion.chunker import SemanticChunker
from lexis.ingestion.feature_extractor import FeatureExtractor
from lexis.indexing.raptor import LexisRaptor
from lexis.indexing.qdrant_client import LexisQdrantClient
from lexis.indexing.es_client import LexisElasticsearchClient
from lexis.indexing.pg_client import PostgresClient, CitationReference, BoundingBox

from lexis.ingestion.interfaces import BaseParser, BaseChunker, BaseEmbedder

class IngestionPipeline:
    def __init__(self, parser: BaseParser = None, embedder: BaseEmbedder = None, chunker: BaseChunker = None):
        self.parser = parser if parser is not None else LexisParser()
        self.embedder = embedder if embedder is not None else BGEM3Embedder()
        self.chunker = chunker if chunker is not None else SemanticChunker(embedder=self.embedder)
        self.feature_extractor = FeatureExtractor()
        self.raptor = LexisRaptor(embedder=self.embedder)
        
        self.qdrant = LexisQdrantClient()
        self.es = LexisElasticsearchClient()
        self.pg = PostgresClient()

    def _deterministic_uuid(self, string_id: str) -> str:
        """Qdrant requires pure UUIDs. We map our pqac- prefixed IDs to pure UUIDs."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

    async def ingest_document(self, file_path: str, doc_id: str, progress_callback=None):
        """
        End-to-End ingestion of a single document for Foundation Phase.
        """
        if progress_callback:
            await progress_callback("PARSING")
        print(f"[{doc_id}] Parsing Document...")
        elements = self.parser.parse(file_path, doc_id)
        
        if progress_callback:
            await progress_callback("CHUNKING")
        print(f"[{doc_id}] Semantic Chunking...")
        chunks: List[Chunk] = self.chunker.chunk(elements)

        if progress_callback:
            await progress_callback("INDEXING")
        print(f"[{doc_id}] Upserting to Databases...")
        await self._upsert_to_databases(chunks)
        
        if progress_callback:
            await progress_callback("COMPLETED")
        print(f"[{doc_id}] Ingestion Complete.")

    async def _upsert_to_databases(self, chunks: List[Chunk]):
        primary_points = []
        
        # 1. Primary Chunks
        texts = [c.content for c in chunks] # embed the CCH prepended content
        if not texts:
            return
            
        embeddings = self.embedder.embed_batch(texts)
        for c, emb in zip(chunks, embeddings):
            primary_points.append(models.PointStruct(
                id=self._deterministic_uuid(c.chunk_id),
                vector=emb.tolist(),
                payload={
                    "chunk_id": c.chunk_id, 
                    "doc_id": c.doc_id, 
                    "content": c.raw_content, 
                    "page_num": c.metadata.page_num,
                    "doc_type": c.metadata.document_type
                }
            ))

        # Upsert Qdrant
        if primary_points: 
            from lexis.config import settings
            await self.qdrant.upsert_chunks(settings.qdrant_collection_primary, primary_points)
        
        # Upsert ES
        if chunks:
            es_docs = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "content": c.raw_content, "doc_type": c.metadata.document_type, "source_file": c.metadata.source_file} for c in chunks]
            await self.es.index_documents(es_docs)
            
        # Upsert Postgres Citations
        await self.pg.initialize_schema()
        for c in chunks:
            bbox_list = c.metadata.bounding_box
            if bbox_list and len(bbox_list) >= 4:
                box = BoundingBox(x0=bbox_list[0], y0=bbox_list[1], x1=bbox_list[2], y1=bbox_list[3])
                citation = CitationReference(
                    pqac_id=c.pqac_key,
                    document_id=c.doc_id,
                    document_version=1,
                    document_hash="",
                    page=c.metadata.page_num or 1,
                    bbox=box,
                    text_span=c.raw_content,
                    chunk_id=c.chunk_id,
                    citation_confidence=1.0
                )
                await self.pg.insert_citation(citation)
