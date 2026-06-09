import numpy as np
from typing import List, Any
import asyncio

from lexis.ingestion.interfaces import BaseParser, BaseChunker, BaseEmbedder
from lexis.ingestion.pipeline import IngestionPipeline

class FakeParser(BaseParser):
    def parse(self, file_path: str) -> List[Any]:
        print(f"FakeParser: Parsing {file_path}")
        return [{"text": "Fake element", "metadata": type("Meta", (), {"page_number": 1, "coordinates": None})()}]

class FakeEmbedder(BaseEmbedder):
    def embed_text(self, text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])
        
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [np.array([0.1, 0.2, 0.3]) for _ in texts]

class FakeChunker(BaseChunker):
    def chunk(self, elements: List[Any]) -> List[Any]:
        print("FakeChunker: Chunking elements")
        return [{"text": "Fake chunk text", "page_num": 1, "bounding_box": None}]

async def test_di():
    # Inject fake dependencies
    pipeline = IngestionPipeline(
        parser=FakeParser(),
        embedder=FakeEmbedder(),
        chunker=FakeChunker()
    )
    
    # Verify that the dependencies were correctly injected
    assert isinstance(pipeline.parser, FakeParser)
    assert isinstance(pipeline.embedder, FakeEmbedder)
    assert isinstance(pipeline.chunker, FakeChunker)
    
    print("✅ Dependency Injection successfully initialized the pipeline!")

if __name__ == "__main__":
    asyncio.run(test_di())
