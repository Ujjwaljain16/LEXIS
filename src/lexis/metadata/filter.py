"""
WHAT: Strictly filters chunks based on Canonical Taxonomy IDs.
WHY: Regex string matching fails at scale. Exact ID matching is fast and accurate.
HOW: Compares canonical IDs assigned during ingestion against target IDs.
"""
from typing import List, Dict, Optional
from .schema import CanonicalMetadata

def prefilter_candidates(
    chunks: List[Dict],
    target_metadata: Optional[CanonicalMetadata] = None
) -> List[Dict]:
    """
    Stage 1 metadata filter.
    Compares exact canonical IDs. No regex. No string mapping.
    """
    if not target_metadata:
        return chunks
        
    survivors = []
    
    for chunk in chunks:
        payload = chunk.get("payload", {})
        
        # Check jurisdiction (e.g. IN-MH)
        if target_metadata.jurisdiction_id:
            chunk_jur = payload.get("jurisdiction_id")
            if chunk_jur and chunk_jur != target_metadata.jurisdiction_id:
                continue
                
        # Check document type (e.g. DOC-10K)
        if target_metadata.document_type_id:
            chunk_doc = payload.get("document_type_id")
            if chunk_doc and chunk_doc != target_metadata.document_type_id:
                continue
                
        # Check court
        if target_metadata.court_id:
            chunk_court = payload.get("court_id")
            if chunk_court and chunk_court != target_metadata.court_id:
                continue
                
        survivors.append(chunk)
        
    return survivors
