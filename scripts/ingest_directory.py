"""Ingest every supported document in a directory (no Redis/worker needed).

    python scripts/ingest_directory.py data/demo_corpus

Writes chunks to Qdrant (dense) and the local bm25s index, using the same IngestionPipeline the
evaluation ran. doc_id is a stable hash of the file name, so re-running overwrites rather than
duplicates (point ids are deterministic). Prints the doc_id for each file -- pass those as
`document_ids` to /v2/query/fast to use document-scoped retrieval.

Heavy: this loads BGE-M3. On a 16 GB laptop prefer Colab for anything beyond a handful of files.
"""
import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".htm", ".html", ".txt"}


def doc_id_for(path: Path) -> str:
    return "doc-" + hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]


async def main(directory: Path) -> int:
    from lexis.ingestion.pipeline import IngestionPipeline

    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No supported files ({', '.join(sorted(SUPPORTED))}) under {directory}")
        return 1
    pipeline = IngestionPipeline()
    failed = 0
    for path in files:
        doc_id = doc_id_for(path)
        try:
            await pipeline.ingest_document(str(path), doc_id)
            print(f"OK    {doc_id}  {path.name}")
        except Exception as e:  # noqa: BLE001 - report and continue; one bad file must not abort the batch
            failed += 1
            print(f"FAIL  {path.name}: {type(e).__name__}: {e}")
    print(f"\n{len(files) - failed}/{len(files)} ingested.")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path)
    sys.exit(asyncio.run(main(ap.parse_args().directory)))
