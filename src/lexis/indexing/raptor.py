"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) for LEXIS.

Rationale: Clusters and recursively summarizes Tier A chunks into higher-order summaries (Tier B, C, etc.)
Source Inspiration: RAPTOR paper and original GitHub implementation.
Deviations from Source Repos: Simplified bottom-up naive k-means clustering per document instead of UMAP/GMM for initial production rollout. Uses Groq for rapid summarization.
Expected Impact on Metrics: Enables Deep Mode to correctly answer aggregation queries (e.g., 'Summarize the risks across all contracts').
"""
from litellm import acompletion
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict, Any
from lexis.config import settings
from lexis.indexing.schema import Chunk

class LexisRaptor:
    def __init__(self, embedder):
        self.embedder = embedder
        self.max_cluster_size = 5

    def _cluster_chunks(self, chunks: List[Chunk]) -> Dict[int, List[Chunk]]:
        """
        Uses simple K-Means to cluster chunks.
        Number of clusters is dynamically set based on max_cluster_size.
        """
        if not chunks:
            return {}

        embeddings = np.array([self.embedder.embed_text(c.raw_content) for c in chunks])
        num_clusters = max(1, len(chunks) // self.max_cluster_size)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunks[idx])
            
        return clusters

    async def _summarize_cluster(self, cluster: List[Chunk], level: int) -> str:
        """
        Uses Groq LLaMA-3 to generate a comprehensive summary of the cluster.
        """
        combined_text = "\n\n---\n\n".join([c.raw_content for c in cluster])
        system_prompt = (
            f"You are an expert summarizer. Synthesize the following text into a comprehensive summary. "
            f"Preserve key facts, dates, and obligations. Do not omit critical details."
        )
        
        try:
            response = await acompletion(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": combined_text}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    async def build_tree(self, leaf_chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        Recursively clusters and summarizes chunks until a single root node remains.
        """
        current_level_chunks = leaf_chunks
        all_summaries = []
        level = 1

        while len(current_level_chunks) > self.max_cluster_size:
            clusters = self._cluster_chunks(current_level_chunks)
            next_level_chunks = []
            
            for label, cluster in clusters.items():
                summary_text = await self._summarize_cluster(cluster, level)
                summary_chunk = Chunk.create(
                    doc_id=cluster[0].doc_id,
                    raw_content=summary_text,
                    split_idx=int(f"{level}00{label}"),  # Unique ID scheme for clusters
                    metadata=cluster[0].metadata # Inherit metadata from first child for simplicity
                )
                
                # We store the parent-child relationships implicitly through the tree metadata
                all_summaries.append({
                    "chunk": summary_chunk,
                    "level": level,
                    "children": [c.chunk_id for c in cluster]
                })
                next_level_chunks.append(summary_chunk)
                
            current_level_chunks = next_level_chunks
            level += 1
            
        # Final root summary
        if len(current_level_chunks) > 1:
            root_summary_text = await self._summarize_cluster(current_level_chunks, level)
            root_chunk = Chunk.create(
                doc_id=leaf_chunks[0].doc_id,
                raw_content=root_summary_text,
                split_idx=999999,
                metadata=leaf_chunks[0].metadata
            )
            all_summaries.append({
                "chunk": root_chunk,
                "level": level,
                "children": [c.chunk_id for c in current_level_chunks]
            })

        return all_summaries
