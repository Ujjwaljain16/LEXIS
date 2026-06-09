"""
Semantic Chunker for LEXIS.

Rationale: Groups unstructured elements into Tier A chunks based on cosine similarity drop-offs.
Source Inspiration: RAG_Techniques Semantic Chunking.
Deviations from Source Repos: Safely merges unstructured bounding boxes across grouped sentences to ensure UI highlights cover the expanded semantic chunk, not just a single sentence.
Expected Impact on Metrics: Increases MRR by ensuring complete semantic concepts are not split arbitrarily by token limits.
"""
from typing import List, Dict, Any
from unstructured.documents.elements import Element
import numpy as np
from lexis.ingestion.interfaces import BaseChunker

class SemanticChunker(BaseChunker):
    def __init__(self, embedder):
        self.embedder = embedder
        self.similarity_threshold = 0.85 # Cosine similarity threshold for semantic breaks

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

    def _extract_bbox(self, el: Element) -> List[float]:
        if el.metadata.coordinates and el.metadata.coordinates.points:
            pts = el.metadata.coordinates.points
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        return None

    def _merge_bboxes(self, box1: List[float], box2: List[float]) -> List[float]:
        if not box1: return box2
        if not box2: return box1
        return [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3]),
        ]

    def chunk(self, elements: List[Element]) -> List[Dict[str, Any]]:
        """
        Groups unstructured elements into semantic chunks.
        """
        valid_elements = [el for el in elements if str(el.text).strip()]
        if not valid_elements:
            return []

        texts = [el.text for el in valid_elements]
        embeddings = self.embedder.embed_batch(texts)

        chunks = []
        current_chunk_text = valid_elements[0].text
        current_bbox = self._extract_bbox(valid_elements[0])
        current_page = valid_elements[0].metadata.page_number or 1

        for i in range(1, len(valid_elements)):
            el = valid_elements[i]
            sim = self._cosine_similarity(embeddings[i-1], embeddings[i])
            page_num = el.metadata.page_number or 1
            
            # Split if similarity drops below threshold or page changes
            if sim < self.similarity_threshold or page_num != current_page:
                chunks.append({
                    "text": current_chunk_text,
                    "bounding_box": current_bbox,
                    "page_num": current_page
                })
                current_chunk_text = el.text
                current_bbox = self._extract_bbox(el)
                current_page = page_num
            else:
                current_chunk_text += " " + el.text
                new_box = self._extract_bbox(el)
                current_bbox = self._merge_bboxes(current_bbox, new_box)

        # Append final chunk
        chunks.append({
            "text": current_chunk_text,
            "bounding_box": current_bbox,
            "page_num": current_page
        })

        return chunks
