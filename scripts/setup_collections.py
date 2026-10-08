"""
Initialization script for LEXIS's vector and search infrastructure.
Run this script once (idempotent) before any ingestion.

Rationale: Infrastructure-as-code initialization for databases.
Source Inspiration: Standard FastAPI app startup scripts.
Deviations: ADR-003 replaced Elasticsearch with a local bm25s index (see
docs/ADR.md); there is no server-side index to initialize for it -- the
index directory is created lazily on first ingestion by LexisBM25Index.
Expected Impact: Idempotent initialization of required search infrastructure.
"""
import asyncio
import sys

from lexis.config import settings
from lexis.indexing.qdrant_client import LexisQdrantClient


async def main():
    print("Initializing Qdrant Collections...")
    qdrant = LexisQdrantClient()
    try:
        await qdrant.initialize_collections()
        print("✅ Qdrant Collections initialized: primary, hype, propositions, clusters")
    except Exception as e:
        # repr, not str: connection failures (timeouts, refused, auth) often have an EMPTY str(),
        # which previously printed a useless "Failed to initialize Qdrant: " with no hint at all.
        print(f"❌ Failed to initialize Qdrant: {type(e).__name__}: {e!r}")
        print("   Check QDRANT_URL / QDRANT_API_KEY (include https:// and the port), and that the "
              "Qdrant Cloud cluster is running (free clusters are suspended after inactivity).")
        sys.exit(1)

    print(f"BM25 index directory: {settings.bm25_index_dir} (created automatically on first ingestion, no separate init needed)")


if __name__ == "__main__":
    asyncio.run(main())
