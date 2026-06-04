"""
WHAT: Convert ranked chunks into LLM-ready context string with citation keys.
WHY: Manages strict token budgets and enforces diversity across documents.
HOW: Uses tiktoken for budgeting. Builds context_map for downstream citation verification.
"""
import tiktoken
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Fast mode: Gemini Flash context window
MODEL_WINDOW = 128000
SYSTEM_RESERVED = 10000
ANSWER_RESERVED = 12000
CITATION_RESERVED = 2000
SAFETY_RESERVED = 2000

MAX_RETRIEVAL_BUDGET = MODEL_WINDOW - (SYSTEM_RESERVED + ANSWER_RESERVED + CITATION_RESERVED + SAFETY_RESERVED)

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning("tiktoken unavailable; using word count approximation.")
            return None
    return _tokenizer

def count_tokens(text: str) -> int:
    tokenizer = get_tokenizer()
    if tokenizer:
        return len(tokenizer.encode(text))
    # Approximation if tiktoken fails
    return len(text.split()) * 2

def enforce_diversity(chunks: List[Dict], max_per_doc: int = 3) -> List[Dict]:
    """
    Prevents a single large document from overwhelming the context window.
    """
    doc_counts = {}
    diverse_chunks = []
    
    for chunk in chunks:
        payload = chunk.get("payload", {})
        doc_id = payload.get("doc_id", "unknown_doc")
        
        current_count = doc_counts.get(doc_id, 0)
        if current_count < max_per_doc:
            diverse_chunks.append(chunk)
            doc_counts[doc_id] = current_count + 1
            
    return diverse_chunks

def assemble_context(
    chunks: List[Dict],
    mode: str = "fast"
) -> Tuple[str, List[str], Dict[str, Dict]]:
    """
    Returns: (context_string, valid_keys_list, context_map)
    context_map: {pqac_key -> structured chunk data with offsets/hashes}
    """
    # Dynamic Budgeting
    budget_ratio = 0.2 if mode == "fast" else 0.6
    token_budget = int(MAX_RETRIEVAL_BUDGET * budget_ratio)
    
    # Enforce diversity policy
    diverse_chunks = enforce_diversity(chunks, max_per_doc=3)
    
    # Deduplicate by chunk_id
    seen_ids = set()
    unique_chunks = []
    for chunk in diverse_chunks:
        cid = chunk.get("payload", {}).get("chunk_id")
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            unique_chunks.append(chunk)

    context_map: Dict[str, Dict] = {}
    lines = []
    current_tokens = 0

    for chunk in unique_chunks:
        payload = chunk["payload"]
        chunk_id = payload.get("chunk_id", "unknown")
        pqac_key = payload.get("pqac_key", f"pqac-{chunk_id[:8]}")
        content = chunk.get("expanded_content") or payload.get("content", "")
        
        # Build context line
        source = payload.get("source_file", "unknown")
        page = payload.get("page_num", 0)
        line = f"{pqac_key}: {content}\n[Source: {source}, Page {page}]"
        
        line_tokens = count_tokens(line)
        if current_tokens + line_tokens > token_budget:
            break
            
        lines.append(line)
        current_tokens += line_tokens
        
        # Store comprehensive context map for Stage 5 auditability
        context_map[pqac_key] = {
            "document_id": payload.get("doc_id", "unknown"),
            "document_hash": payload.get("document_hash", "unknown"),
            "document_version": payload.get("document_version", "1.0"),
            "chunk_id": chunk_id,
            "text": content,
            "bbox": payload.get("bounding_box"),
            "start_char": payload.get("start_char", 0),
            "end_char": payload.get("end_char", len(content))
        }

    context_string = "\n\n---\n\n".join(lines)
    valid_keys = list(context_map.keys())

    return context_string, valid_keys, context_map
