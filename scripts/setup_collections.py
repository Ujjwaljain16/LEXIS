"""
Initialization script for LEXIS vector and search databases.
Run this script to set up Qdrant collections and Elasticsearch indices.

Rationale: Infrastructure-as-code initialization for databases.
Source Inspiration: Standard FastAPI app startup scripts.
Deviations: N/A.
Expected Impact: Idempotent initialization of required search infrastructure.
"""
import sys
import os
import asyncio

from lexis.indexing.qdrant_client import LexisQdrantClient
from lexis.indexing.es_client import LexisElasticsearchClient

async def main():
    print("Initializing Qdrant Collections...")
    qdrant = LexisQdrantClient()
    try:
        await qdrant.initialize_collections()
        print("✅ Qdrant Collections initialized: primary, hype, propositions, clusters")
    except Exception as e:
        print(f"❌ Failed to initialize Qdrant: {e}")

    print("Initializing Elasticsearch Indices...")
    es = LexisElasticsearchClient()
    try:
        await es.initialize_index()
        print(f"✅ Elasticsearch Index initialized: {es.index_name}")
    except Exception as e:
        print(f"❌ Failed to initialize Elasticsearch: {e}")

if __name__ == "__main__":
    asyncio.run(main())
