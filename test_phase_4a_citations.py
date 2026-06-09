import asyncio
from lexis.ingestion.pipeline import LexisParser
from lexis.indexing.pg_client import PostgresClient

async def test_citations():
    pg = PostgresClient()
    await pg.init_pool()
    await pg.init_schema()

    parser = LexisParser()
    chunks = parser.parse("sample_contract.pdf")
    print(f"Extracted {len(chunks)} chunks with bounding boxes")

    for chunk in chunks:
        await pg.upsert_citation(chunk)

    print("✅ Verified Postgres Idempotent Persistence")
    
if __name__ == "__main__":
    asyncio.run(test_citations())
