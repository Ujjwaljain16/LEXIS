from typing import Dict, Any, List
from lexis.retrieval.interfaces import Candidate
from lexis.reranking.map_reduce_filter import ResearchSession

def dict_to_candidate(chunk: Dict[str, Any]) -> Candidate:
    """Isolates the SentenceWindow dependency from the rest of the engine."""
    return Candidate(
        chunk_id=chunk.get("chunk_id", ""),
        score=float(chunk.get("_relevance_score", 0.0)),
        source_path=chunk.get("_source_path", ""),
        content=str(chunk.get("text") or chunk.get("content") or chunk.get("proposition") or ""),
        metadata=chunk # Retain entire original payload for the round-trip
    )

def candidate_to_dict(candidate: Candidate) -> Dict[str, Any]:
    """Reverts the Candidate back to the canonical Dict structure."""
    base = dict(candidate.metadata)
    base["chunk_id"] = candidate.chunk_id
    base["_relevance_score"] = candidate.score
    base["_source_path"] = candidate.source_path
    base["text"] = candidate.content
    return base

def flatten_research_graph(session: ResearchSession) -> List[Dict[str, Any]]:
    """Maps MapReduce Evidence Nodes back to the Synthesizer specification."""
    flattened = []
    for node in session.graph.nodes:
        # Reconstruct exactly the shape Synthesizer expects
        # chunk_id doesn't strictly matter for synthesis, but citations MUST map to source_path
        chunk_id = node.citations[0] if node.citations else "unknown"
        
        flattened.append({
            "chunk_id": chunk_id,
            "text": node.evidence_text,
            "_source_path": ", ".join(node.citations),
            "_relevance_score": node.confidence_score
        })
    return flattened
