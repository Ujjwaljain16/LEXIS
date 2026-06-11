import re
import numpy as np
from typing import List, Dict, Any, Optional
from lexis.ingestion.interfaces import BaseChunker
from lexis.indexing.schema import Chunk, ChunkMetadata
from lexis.config import settings
import tiktoken

def count_tokens(text: str) -> int:
    # Approximate token count using fast tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

class SemanticChunker(BaseChunker):
    def __init__(self, embedder):
        self.embedder = embedder

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2))

    def _merge_bboxes(self, box1: Optional[List[float]], box2: Optional[List[float]]) -> Optional[List[float]]:
        if not box1: return box2
        if not box2: return box1
        return [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3]),
        ]

    def _build_cch_header(self, doc_title: str, doc_type: str, section: str) -> str:
        return f"Document: {doc_title}\nType: {doc_type}\nSection: {section}\n\n"

    def _split_text_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_len = 0
        
        for word in words:
            # approximation: 1 token ~ 0.75 words. For safety, just count simple spaces or use tiktoken exactly.
            # To be fast but safe, we'll over-split slightly if needed.
            current_chunk.append(word)
            if len(current_chunk) >= max_tokens * 0.75: 
                # exact check
                if count_tokens(" ".join(current_chunk)) >= max_tokens:
                    # pop the last word, save chunk
                    current_chunk.pop()
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def chunk(self, elements: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Groups raw element dictionaries into fully formed Chunk objects.
        Enforces token budgets and prepends CCH headers.
        """
        chunks: List[Chunk] = []
        
        if not elements:
            return chunks

        doc_title = "Unknown Document"
        current_section = "Introduction"
        
        accumulated_texts = []
        accumulated_meta = []
        
        def flush_accumulated():
            nonlocal accumulated_texts, accumulated_meta
            if not accumulated_texts:
                return
            
            combined_text = " ".join(accumulated_texts)
            
            # Step 1: Sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', combined_text.strip())
            sentences = [s for s in sentences if s.strip()]
            
            if len(sentences) <= 3:
                sub_chunks = [combined_text]
            else:
                embeddings = self.embedder.embed_batch(sentences)
                similarities = [
                    self._cosine_similarity(embeddings[i], embeddings[i+1])
                    for i in range(len(embeddings) - 1)
                ]
                
                split_indices = [0]
                current_tokens = count_tokens(sentences[0])
                for i, sim in enumerate(similarities):
                    next_tokens = count_tokens(sentences[i+1])
                    
                    # Split if semantic drop OR if target budget reached
                    if sim < settings.semantic_chunking_threshold or current_tokens + next_tokens > settings.chunk_target_tokens:
                        split_indices.append(i + 1)
                        current_tokens = next_tokens
                    else:
                        current_tokens += next_tokens
                        
                split_indices.append(len(sentences))
                
                sub_chunks = []
                for i in range(len(split_indices) - 1):
                    chunk_text = " ".join(sentences[split_indices[i]:split_indices[i+1]])
                    # Hard enforcement: if somehow it still exceeds max_tokens (e.g. giant sentence), split it raw.
                    if count_tokens(chunk_text) > settings.chunk_max_tokens:
                        sub_chunks.extend(self._split_text_by_tokens(chunk_text, settings.chunk_max_tokens))
                    else:
                        sub_chunks.append(chunk_text)

            # Create Chunk objects
            base_meta = accumulated_meta[0]
            doc_type = base_meta.get("doc_type", "unknown")
            doc_id = base_meta["doc_id"]
            
            for sub_idx, sub_text in enumerate(sub_chunks):
                if not sub_text.strip():
                    continue
                
                cch_header = self._build_cch_header(doc_title, doc_type, current_section)
                content_with_cch = cch_header + sub_text
                
                # Merge bboxes for the whole accumulation
                merged_bbox = None
                for m in accumulated_meta:
                    merged_bbox = self._merge_bboxes(merged_bbox, m.get("bounding_box"))
                
                meta_obj = ChunkMetadata(
                    source_file=base_meta["source_file"],
                    page_num=base_meta["page_num"],
                    bounding_box=merged_bbox,
                    document_type=doc_type
                )
                
                chunk_obj = Chunk.create(
                    doc_id=doc_id,
                    split_idx=base_meta["split_idx"] * 100 + sub_idx,
                    raw_content=sub_text,  # Clean content for display
                    metadata=meta_obj
                )
                
                # Override chunk_obj.content with the CCH pre-pended text (this is what gets embedded)
                chunk_obj.expanded_content = content_with_cch
                
                chunks.append(chunk_obj)
                
            accumulated_texts.clear()
            accumulated_meta.clear()

        for element in elements:
            block_type = element["block_type"]
            text = element["content"]
            
            if block_type == "title":
                flush_accumulated()
                current_section = text
                if len(text) < 100 and doc_title == "Unknown Document":
                    doc_title = text
            elif block_type in ("table", "figure"):
                flush_accumulated()
                # Tables/Figures standalone
                cch_header = self._build_cch_header(doc_title, element.get("doc_type", "unknown"), current_section)
                
                meta_obj = ChunkMetadata(
                    source_file=element["source_file"],
                    page_num=element["page_num"],
                    bounding_box=element.get("bounding_box"),
                    document_type=element.get("doc_type", "unknown")
                )
                chunk_obj = Chunk.create(
                    doc_id=element["doc_id"],
                    split_idx=element["split_idx"],
                    raw_content=text,
                    metadata=meta_obj
                )
                chunk_obj.expanded_content = cch_header + text
                chunks.append(chunk_obj)
            else:
                accumulated_texts.append(text)
                accumulated_meta.append(element)
                
                # Proactive flush
                if count_tokens(" ".join(accumulated_texts)) > settings.chunk_target_tokens:
                    flush_accumulated()

        flush_accumulated()
        return chunks
