"""
Reranking & Context Assembly Layer for LEXIS.

Rationale: Evaluates and re-orders the diverse chunks retrieved across the 4 paths, packing only the most relevant ones into the LLM context window.
Source Inspiration: BAAI/bge-reranker-v2-m3 and PaperQA context assembly.
Deviations from Source Repos: Strictly uses token-counting (tiktoken) to guarantee we do not exceed the 8K context limit of the generator LLM.
Expected Impact on Metrics: Solves the 'lost in the middle' phenomenon using explicit interleaving. Improves Context Precision significantly.
"""
import math
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import tiktoken
import logging

logger = logging.getLogger(__name__)

class ContextAssembler:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_tokens: int = 6000):
        # We use a base reranker for latency. v2-m3 is better but heavier.
        try:
            self.reranker = CrossEncoder(model_name)
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder {model_name}: {e}")
            self.reranker = None
            
        self.max_tokens = max_tokens
        # We use cl100k_base because LLaMA/Gemini tokenizers are similar enough for safe bounds,
        # and tiktoken is extremely fast in Python.
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def rerank_and_pack(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        1. Scores each chunk against the query using CrossEncoder.
        2. Sorts descending by score.
        3. Packs chunks into a list until max_tokens is reached.
        """
        if not chunks:
            return []

        # If reranker failed to load, just pack safely based on original retrieval order
        if not self.reranker:
            logger.warning("Reranker not loaded. Falling back to naive packing.")
            return self._pack_chunks(chunks)

        # Prepare pairs for the CrossEncoder
        pairs = []
        for chunk in chunks:
            # Handle heterogeneous payloads from Qdrant/ES
            text = chunk.get("text") or chunk.get("content") or chunk.get("proposition") or chunk.get("questions")
            
            if isinstance(text, list):
                text = " ".join(text)
                
            text = str(text) if text else ""
            pairs.append([query, text])

        try:
            # Calculate cross-encoder scores
            scores = self.reranker.predict(pairs)
            
            # Attach scores to chunks
            for i, chunk in enumerate(chunks):
                chunk["_relevance_score"] = float(scores[i])
                
            # Sort descending by relevance score
            ranked_chunks = sorted(chunks, key=lambda x: x.get("_relevance_score", -999.0), reverse=True)
            
            # Pack chunks safely
            return self._pack_chunks(ranked_chunks)
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return self._pack_chunks(chunks)

    def rerank_only(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Deep Mode Contract.
        Scores and sorts the chunks descending, trimming exactly to top_k.
        Does NOT apply token packing or Lost-in-the-Middle interleaving.
        """
        if not chunks:
            return []

        if not self.reranker:
            logger.warning("Reranker not loaded. Falling back to naive trimming.")
            return chunks[:top_k]

        pairs = []
        for chunk in chunks:
            text = chunk.get("text") or chunk.get("content") or chunk.get("proposition") or chunk.get("questions")
            if isinstance(text, list):
                text = " ".join(text)
            text = str(text) if text else ""
            pairs.append([query, text])

        try:
            scores = self.reranker.predict(pairs)
            for i, chunk in enumerate(chunks):
                chunk["_relevance_score"] = float(scores[i])
            
            ranked_chunks = sorted(chunks, key=lambda x: x.get("_relevance_score", -999.0), reverse=True)
            return ranked_chunks[:top_k]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks[:top_k]

    def _pack_chunks(self, ranked_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        packed = []
        current_tokens = 0
        
        for chunk in ranked_chunks:
            text = chunk.get("text") or chunk.get("content") or chunk.get("proposition") or chunk.get("questions")
            if isinstance(text, list):
                text = " ".join(text)
            text = str(text) if text else ""
            
            tokens = self._count_tokens(text)
            
            # Plus 20 for formatting overhead (e.g. "Source: [doc_id]\n")
            if current_tokens + tokens + 20 > self.max_tokens:
                break
                
            current_tokens += tokens + 20
            packed.append(chunk)
            
        # Apply Lost In The Middle Interleaving
        # Prioritizes highest signals at the start and end of the context
        interleaved = []
        left_ptr, right_ptr = 0, len(packed) - 1
        turn = True # True = left, False = right
        
        # Temp array to hold interleaved order
        temp_arr = [None] * len(packed)
        for i, chunk in enumerate(packed):
            if i % 2 == 0:
                temp_arr[left_ptr] = chunk
                left_ptr += 1
            else:
                temp_arr[right_ptr] = chunk
                right_ptr -= 1
                
        packed = temp_arr
            
        logger.info(f"Context Assembler packed and interleaved {len(packed)} chunks ({current_tokens}/{self.max_tokens} tokens)")
        return packed
