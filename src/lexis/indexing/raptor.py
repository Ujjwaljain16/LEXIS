"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) for LEXIS.

Rationale: Clusters and recursively summarizes Tier A chunks into higher-order summaries (Tier B, C, etc.)
Source Inspiration: RAPTOR paper and original GitHub implementation.
Deviations from Source Repos: 
- UMAP + GMM for sophisticated soft-density modeling, flattened to hard assignments for MVP.
- Concurrent LLM summarization with strict failure isolation.
Expected Impact on Metrics: Enables Deep Mode to correctly answer aggregation queries (e.g., 'Summarize the risks across all contracts').
"""
import asyncio
import logging
import uuid
import numpy as np
from typing import List, Dict, Any, Tuple
from litellm import acompletion
from sklearn.mixture import GaussianMixture
import umap

from lexis.config import settings
from lexis.indexing.schema import Chunk, ClusterSummary

logger = logging.getLogger(__name__)

class LexisRaptor:
    def __init__(self, embedder):
        self.embedder = embedder
        self.max_clusters = 50
        self.timeout = getattr(settings, "raptor_summarization_timeout", 45)

    def _perform_umap_gmm_clustering(self, embeddings: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
        """
        Uses UMAP for dimensionality reduction followed by Gaussian Mixture Models.
        Determines the optimal number of clusters using Bayesian Information Criterion (BIC).
        """
        n_samples = len(embeddings)
        if n_samples <= 2:
            # Too few samples to cluster properly, assign them all to one cluster
            return np.zeros(n_samples, dtype=int)
            
        # 1. Dimensionality Reduction (UMAP)
        # Adjust n_neighbors if there are very few samples
        actual_n_neighbors = min(n_neighbors, n_samples - 1)
        # Reduce dimension to capture local and global semantic density
        reducer = umap.UMAP(n_neighbors=actual_n_neighbors, n_components=min(5, n_samples-1), metric="cosine", random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # 2. Optimal Cluster Size Search (BIC with GMM)
        max_k = min(self.max_clusters, n_samples)
        best_gmm = None
        lowest_bic = float('inf')
        
        # Try k from 1 up to max_k
        for k in range(1, max_k + 1):
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=1)
            gmm.fit(reduced_embeddings)
            bic = gmm.bic(reduced_embeddings)
            
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
                
        if best_gmm is None:
            return np.zeros(n_samples, dtype=int)
            
        # 3. Hard Clustering Assignment
        # RAPTOR supports soft clustering (probabilities), but for the MVP 
        # we assign each chunk to the single highest-probability cluster.
        labels = best_gmm.predict(reduced_embeddings)
        return labels

    def _cluster_chunks(self, chunks: List[Chunk]) -> Dict[int, List[Chunk]]:
        """
        Groups chunks into semantic clusters using UMAP + GMM.
        """
        if not chunks:
            return {}

        embeddings = np.array([self.embedder.embed_text(c.content) for c in chunks])
        labels = self._perform_umap_gmm_clustering(embeddings)
        
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunks[idx])
            
        return clusters

    async def _summarize_cluster_task(self, cluster: List[Chunk], level: int, label: int) -> ClusterSummary | None:
        """
        Worker task for summarizing a single cluster with explicit failure isolation.
        Returns a strongly typed ClusterSummary object or None if failed.
        """
        combined_text = "\n\n---\n\n".join([c.content for c in cluster])
        system_prompt = (
            f"You are an expert summarizer. Synthesize the following text into a comprehensive summary. "
            f"Preserve key facts, dates, and obligations. Do not omit critical details."
        )
        
        cluster_id = f"lvl{level}-c{label}-{uuid.uuid4().hex[:8]}"
        child_ids = [c.chunk_id for c in cluster]

        try:
            response = await asyncio.wait_for(
                acompletion(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": combined_text}
                    ],
                    temperature=0.3
                ),
                timeout=self.timeout
            )
            summary_text = response.choices[0].message.content
            
            return ClusterSummary(
                cluster_id=cluster_id,
                level=level,
                summary_text=summary_text,
                child_chunk_ids=child_ids
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout during summarization of cluster {cluster_id}. Isolating failure.")
            return None
        except Exception as e:
            logger.warning(f"Error summarizing cluster {cluster_id}: {e}. Isolating failure.")
            return None

    async def build_tree(self, leaf_chunks: List[Chunk]) -> List[ClusterSummary]:
        """
        Recursively clusters and summarizes chunks using UMAP+GMM concurrently.
        Outputs a flat list of ClusterSummary objects that define the tree structure.
        """
        current_level_chunks = leaf_chunks
        all_summaries: List[ClusterSummary] = []
        level = 1

        while len(current_level_chunks) > 1:
            clusters = self._cluster_chunks(current_level_chunks)
            
            # If the algorithm decides everything belongs to 1 cluster, we break the loop to handle the root.
            if len(clusters) == 1:
                break
                
            tasks = []
            for label, cluster in clusters.items():
                tasks.append(self._summarize_cluster_task(cluster, level, label))
                
            # Run all summarization tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            next_level_chunks = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Unexpected crash in asyncio.gather for cluster summarization: {result}")
                    continue
                if result is not None:
                    # Save the strongly-typed summary
                    all_summaries.append(result)
                    
                    # Convert the summary into a generic Chunk so it can be re-clustered in the next loop
                    summary_chunk = Chunk.create(
                        doc_id=current_level_chunks[0].doc_id, # Inherit doc_id
                        raw_content=result.summary_text,
                        split_idx=int(f"{level}000{len(next_level_chunks)}"),
                        metadata=current_level_chunks[0].metadata # Inherit raw metadata for now
                    )
                    next_level_chunks.append(summary_chunk)

            # If no successful summaries were generated, we cannot proceed further up the tree
            if not next_level_chunks:
                logger.error(f"All cluster summarizations failed at level {level}. Aborting further tree construction.")
                break
                
            current_level_chunks = next_level_chunks
            level += 1
            
        # Final root summary computation (either starting from leaf or the highest tier B/C nodes)
        if len(current_level_chunks) > 0:
            root_summary_result = await self._summarize_cluster_task(current_level_chunks, level, 999)
            if root_summary_result is not None:
                all_summaries.append(root_summary_result)

        return all_summaries
