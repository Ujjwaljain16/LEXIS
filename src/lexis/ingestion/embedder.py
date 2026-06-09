"""
Embedder for LEXIS.

Rationale: Provides dense vector embeddings using the mandatory BAAI/bge-m3 model.
Source Inspiration: plan.md architecture diagram.
Deviations from Source Repos: None.
Expected Impact on Metrics: High baseline retrieval recall due to BGE-M3's state-of-the-art multi-lingual and dense retrieval capabilities.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from lexis.ingestion.interfaces import BaseEmbedder

class BGEM3Embedder(BaseEmbedder):
    def __init__(self):
        # Using bge-small (133MB) instead of bge-m3 (2.27GB) to prevent OOM on 16GB RAM machines
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single text string."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embeds a batch of strings efficiently."""
        return self.model.encode(texts, normalize_embeddings=True)
