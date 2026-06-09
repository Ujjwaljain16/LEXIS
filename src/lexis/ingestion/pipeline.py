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
from lexis.ingestion.raptor import LexisRaptor
from lexis.indexing.qdrant_client import LexisQdrantClient
from lexis.indexing.es_client import LexisElasticsearchClient
from lexis.indexing.pg_client import PostgresClient, CitationReference, BoundingBox

class IngestionPipeline:
    def __init__(self):
        self.parser = LexisParser()
        self.embedder = BGEM3Embedder()
        self.chunker = SemanticChunker(embedder=self.embedder)
        self.feature_extractor = FeatureExtractor()
        self.raptor = LexisRaptor(embedder=self.embedder)
        
        self.qdrant = LexisQdrantClient()
        self.es = LexisElasticsearchClient()
        self.pg = PostgresClient()

    def _deterministic_uuid(self, string_id: str) -> str:
        """Qdrant requires pure UUIDs. We map our pqac- prefixed IDs to pure UUIDs."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

    async def ingest_document(self, file_path: str, doc_id: str):
        """
        End-to-End ingestion of a single document.
        """
        print(f"[{doc_id}] Parsing PDF...")
        elements = self.parser.parse_pdf(file_path)
        
        print(f"[{doc_id}] Semantic Chunking...")
        raw_chunks = self.chunker.chunk_elements(elements)
        
        chunks: List[Chunk] = []
        for idx, rc in enumerate(raw_chunks):
            metadata = ChunkMetadata(
                source_file=file_path,
                page_num=rc["page_num"],
                bounding_box=rc["bounding_box"]
            )
            chunks.append(Chunk.create(
                doc_id=doc_id,
                split_idx=idx,
                raw_content=rc["text"],
                metadata=metadata
            ))

        print(f"[{doc_id}] Extracting Features (Propositions & HyPE) concurrently...")
        batch_size = 10
        all_features = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            tasks = [self.feature_extractor.extract_features(c) for c in batch]
            results = await asyncio.gather(*tasks)
            all_features.extend(results)

        print(f"[{doc_id}] Building RAPTOR Tree...")
        raptor_summaries = await self.raptor.build_tree(chunks)

        print(f"[{doc_id}] Upserting to Databases...")
        await self._upsert_to_databases(chunks, all_features, raptor_summaries)
        print(f"[{doc_id}] Ingestion Complete.")

    async def _upsert_to_databases(self, chunks: List[Chunk], features: List[dict], raptor_summaries: List[dict]):
        primary_points = []
        hype_points = []
        prop_points = []
        cluster_points = []
        
        # 1. Primary Chunks
        texts = [c.raw_content for c in chunks]
        if not texts:
            return
            
        embeddings = self.embedder.embed_batch(texts)
        for c, emb in zip(chunks, embeddings):
            primary_points.append(models.PointStruct(
                id=self._deterministic_uuid(c.chunk_id),
                vector=emb.tolist(),
                payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.raw_content, "page_num": c.metadata.page_num}
            ))
            
        # 2. HyPE and Propositions
        for chunk, feat in zip(chunks, features):
            questions = feat["hype"].hypothesis_questions
            if questions:
                q_text = " ".join(questions)
                q_emb = self.embedder.embed_text(q_text)
                hype_points.append(models.PointStruct(
                    id=self._deterministic_uuid(chunk.chunk_id + "_hype"),
                    vector=q_emb.tolist(),
                    payload={"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "questions": questions}
                ))
            
            for prop in feat["propositions"]:
                prop_str = f"{prop.subject} {prop.predicate} {prop.object}"
                p_emb = self.embedder.embed_text(prop_str)
                
                # Generate deterministic prop_id if not present
                prop_id_str = prop.prop_id if prop.prop_id else f"{chunk.chunk_id}_{prop_str}"
                
                prop_points.append(models.PointStruct(
                    id=self._deterministic_uuid(prop_id_str),
                    vector=p_emb.tolist(),
                    payload={"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "proposition": prop_str}
                ))

        # 3. RAPTOR
        if raptor_summaries:
            summary_texts = [s["chunk"].raw_content for s in raptor_summaries]
            sum_embs = self.embedder.embed_batch(summary_texts)
            for s, emb in zip(raptor_summaries, sum_embs):
                cluster_points.append(models.PointStruct(
                    id=self._deterministic_uuid(s["chunk"].chunk_id),
                    vector=emb.tolist(),
                    payload={"chunk_id": s["chunk"].chunk_id, "doc_id": s["chunk"].doc_id, "level": s["level"]}
                ))

        # Upsert Qdrant
        if primary_points: await self.qdrant.upsert_chunks("primary_v2", primary_points)
        if hype_points: await self.qdrant.upsert_chunks("hype_v2", hype_points)
        if prop_points: await self.qdrant.upsert_chunks("propositions_v2", prop_points)
        if cluster_points: await self.qdrant.upsert_chunks("clusters_v2", cluster_points)
        
        # Upsert ES
        if chunks:
            es_docs = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "content": c.raw_content} for c in chunks]
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
