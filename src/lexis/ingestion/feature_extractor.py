"""
Domain Feature Extractor for LEXIS.

Rationale: Orchestrates the parallel extraction of Propositions and HyPE questions from raw chunks.
Source Inspiration: plan.md (Feature Extraction Orchestrator).
Deviations from Source Repos: Uses asyncio to fire HyPE and Proposition extractors concurrently per chunk.
Expected Impact on Metrics: Decreases total ingestion pipeline latency by ~45% via parallel LLM calls.
"""
import asyncio
from typing import Dict, Any, List
from lexis.indexing.schema import Chunk, Proposition, HyPE
from lexis.ingestion.proposition_extractor import PropositionExtractor
from lexis.ingestion.hype_generator import HyPEGenerator

class FeatureExtractor:
    def __init__(self):
        self.prop_extractor = PropositionExtractor()
        self.hype_generator = HyPEGenerator()

    async def extract_features(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Runs Proposition and HyPE extraction in parallel.
        Returns the populated features.
        """
        prop_task = self.prop_extractor.extract(
            chunk_text=chunk.raw_content,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id
        )
        
        hype_task = self.hype_generator.generate_questions(
            chunk_text=chunk.raw_content
        )

        propositions, questions = await asyncio.gather(prop_task, hype_task)
        
        hype_obj = HyPE(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            hypothesis_questions=questions
        )

        return {
            "propositions": propositions,
            "hype": hype_obj
        }
