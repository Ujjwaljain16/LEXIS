import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from lexis.ingestion.interfaces import BaseEmbedder
from lexis.config import settings

_embedder: SentenceTransformer | None = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        # Load the exact model from configuration, defaulting to BAAI/bge-m3
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder

class BGEM3Embedder(BaseEmbedder):
    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single text string."""
        model = get_embedder()
        return model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embeds a batch of strings efficiently."""
        model = get_embedder()
        # show_progress_bar if batch is large
        return model.encode(
            texts, 
            batch_size=32, 
            normalize_embeddings=True, 
            show_progress_bar=len(texts) > 100
        )
