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

from lexis.config import settings
from lexis.indexing.schema import Chunk, ChunkMetadata
from lexis.ingestion.parser import LexisParser
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.ingestion.chunker import SemanticChunker
from lexis.ingestion.feature_extractor import FeatureExtractor
from lexis.ingestion.hype_generator import HyPEGenerator
from lexis.indexing.raptor import LexisRaptor
from lexis.indexing.qdrant_client import LexisQdrantClient
from lexis.indexing.bm25_index import LexisBM25Index
from lexis.indexing.pg_client import PostgresClient, CitationReference, BoundingBox

from lexis.ingestion.interfaces import BaseParser, BaseChunker, BaseEmbedder

class IngestionPipeline:
    def __init__(self, parser: BaseParser = None, embedder: BaseEmbedder = None, chunker: BaseChunker = None):
        self.parser = parser if parser is not None else LexisParser()
        self.embedder = embedder if embedder is not None else BGEM3Embedder()
        self.chunker = chunker if chunker is not None else SemanticChunker(embedder=self.embedder)
        self.feature_extractor = FeatureExtractor()
        self.hype_generator = HyPEGenerator()
        self.raptor = LexisRaptor(embedder=self.embedder)

        self.qdrant = LexisQdrantClient()
        self.bm25 = LexisBM25Index(index_dir=settings.bm25_index_dir)
        self.pg = PostgresClient()

    def _deterministic_uuid(self, string_id: str) -> str:
        """Qdrant requires pure UUIDs. We map our pqac- prefixed IDs to pure UUIDs."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

    async def _build_hype_points(self, chunks: List[Chunk]) -> List[models.PointStruct]:
        """One HyPEGenerator call per chunk, concurrency-bounded (see
        settings.hype_max_concurrent_requests -- a live run against Gemini's free tier
        confirmed firing every chunk's call at once, unbounded, blows through the
        20 req/min quota almost immediately and fails most of them with RateLimitError),
        then one batched embed_batch call over every generated question across all chunks
        (not per-chunk) -- matches how the primary/BM25 paths above already batch embeddings.

        Each point's payload denormalizes the chunk's real content (not just chunk_id)
        so retrieval/hybrid_retriever.py's HyPE path can build a Candidate directly from
        the hit, exactly like path_b_global/path_d_bm25 do -- no second lookup at query
        time, and no risk of apply_rrf ever surfacing an empty/wrong "content" for a
        chunk that RRF fusion first sees via this path (see fusion.py: the first
        candidate seen for a chunk_id wins its content/metadata).
        """
        semaphore = asyncio.Semaphore(settings.hype_max_concurrent_requests)

        async def bounded_generate(chunk: Chunk) -> List[str]:
            async with semaphore:
                return await self.hype_generator.generate_questions(chunk.raw_content)

        per_chunk_questions = await asyncio.gather(*(bounded_generate(c) for c in chunks))

        texts, owners = [], []
        for c, questions in zip(chunks, per_chunk_questions):
            for q in questions:
                if q and q.strip():
                    texts.append(q)
                    owners.append((c, q))
        if not texts:
            return []

        embeddings = self.embedder.embed_batch(texts)
        points = []
        for (c, q), emb in zip(owners, embeddings):
            point_id = self._deterministic_uuid(f"{c.chunk_id}|hype|{q}")
            points.append(models.PointStruct(
                id=point_id,
                vector=emb.tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "content": c.raw_content,
                    "question": q,
                    "doc_type": c.metadata.document_type,
                    "chunk_index": c.split_idx,
                }
            ))
        return points

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
                    "doc_type": c.metadata.document_type,
                    "chunk_index": c.split_idx
                }
            ))

        # Upsert Qdrant
        if primary_points:
            await self.qdrant.upsert_chunks(settings.qdrant_collection_primary, primary_points)

        # R2: HyPE question index (off by default -- settings.hype_enabled). Generates N
        # hypothetical questions per chunk (index-time LLM cost only, never at query time) and
        # embeds them into a separate collection so a short query can match a semantically close
        # hypothetical question instead of the (usually much longer, differently-phrased) chunk
        # text itself -- see LEXIS_FINAL_PLAN.md section 4, R2.
        if settings.hype_enabled and chunks:
            hype_points = await self._build_hype_points(chunks)
            if hype_points:
                await self.qdrant.upsert_chunks(settings.qdrant_collection_hype, hype_points)

        # Upsert BM25 (ADR-003: local bm25s index, replaces Elasticsearch).
        # index_text carries the CCH-prefixed text (doc title/type/section), matching what the
        # dense path already embeds above -- previously BM25 tokenized raw_content only, so
        # keyword search never benefited from that context. "content" stays raw_content, since
        # LexisBM25Index returns the corpus dict verbatim as each hit's payload and path_d_bm25.py
        # reads payload["content"] as the candidate's actual text -- indexing the CCH-prefixed
        # text under "content" would leak "Document: ...\nType: ...\nSection: ...\n\n" into every
        # BM25 candidate's content shown to the LLM/citations.
        if chunks:
            bm25_docs = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "content": c.raw_content,
                    "index_text": c.content,
                    "doc_type": c.metadata.document_type,
                    "source_file": c.metadata.source_file,
                    "chunk_index": c.split_idx,
                }
                for c in chunks
            ]
            self.bm25.add_documents(bm25_docs)
            
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
