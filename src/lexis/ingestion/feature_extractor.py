"""
Domain Feature Extractor for LEXIS.

Rationale: Orchestrates the parallel extraction of Propositions, HyPE questions, and Domain Features from raw chunks.
Source Inspiration: plan.md (Feature Extraction Orchestrator).
Deviations from Source Repos: 
- Uses asyncio to fire HyPE, Proposition, and Domain extractors concurrently per chunk.
- Utilizes structured output and jsonschema validation for robust metadata population.
- Uses a decoupled FeatureSchemaRegistry to remain agnostic of specific document types.
Expected Impact on Metrics: Decreases total ingestion pipeline latency via parallel LLM calls. Enables powerful metadata filtering for retrieval.
"""
import asyncio
import json
import jsonschema
import logging
from typing import Dict, Any
from litellm import acompletion

from lexis.indexing.schema import Chunk, Proposition, HyPE, DomainFeatureResult
from lexis.ingestion.proposition_extractor import PropositionExtractor
from lexis.ingestion.hype_generator import HyPEGenerator
from lexis.config import settings
from lexis.ingestion.schemas.registry import FeatureSchemaRegistry
from lexis.ingestion.schemas.setup import create_default_registry

logger = logging.getLogger(__name__)

FEATURE_PROMPT_TEMPLATE = """Extract structured features from this text.
Output ONLY valid JSON. No preamble, no markdown, no explanation.

Text: {text}

Required JSON structure (populate fields as appropriate, use empty lists/null if not found):
{schema}

Instructions:
{instructions}
- Normalize terms (e.g. "terminated" -> "termination")
- Ensure valid JSON matching the exact schema
"""

class FeatureExtractor:
    def __init__(self, registry: FeatureSchemaRegistry | None = None):
        self.prop_extractor = PropositionExtractor()
        self.hype_generator = HyPEGenerator()
        self.registry = registry or create_default_registry()
        # Fallback timeout if not defined in settings
        self.timeout = getattr(settings, "feature_extraction_timeout", 30)

    async def _extract_domain_features(self, chunk_text: str, doc_type: str) -> DomainFeatureResult:
        schema = self.registry.get_schema(doc_type)
        version = self.registry.get_version(doc_type)
        instructions = self.registry.get_instructions(doc_type)

        system_prompt = FEATURE_PROMPT_TEMPLATE.format(
            text=chunk_text[:3000],  # safety cap
            schema=json.dumps(schema, indent=2),
            instructions=instructions
        )

        try:
            # Wrap the acompletion with a timeout for reliability
            response = await asyncio.wait_for(
                acompletion(
                    model=settings.gemini_model_feature,
                    messages=[
                        {"role": "user", "content": system_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                ),
                timeout=self.timeout
            )
            
            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)
            
            # Validate against the registered schema
            jsonschema.validate(instance=parsed, schema=schema)
            
            return DomainFeatureResult(
                data=parsed,
                failed=False,
                error_type=None,
                error_message=None,
                schema_version=version
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Domain extraction timed out for chunk.")
            return DomainFeatureResult(
                data={}, failed=True, error_type="timeout", error_message="LLM extraction timed out", schema_version=version
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON domain features: {e}")
            return DomainFeatureResult(
                data={}, failed=True, error_type="json_parse_error", error_message=str(e), schema_version=version
            )
        except jsonschema.ValidationError as e:
            logger.warning(f"Schema validation failed for domain features: {e}")
            return DomainFeatureResult(
                data={}, failed=True, error_type="schema_validation_error", error_message=str(e), schema_version=version
            )
        except Exception as e:
            logger.error(f"Unexpected error in domain feature extraction: {e}")
            return DomainFeatureResult(
                data={}, failed=True, error_type="llm_error", error_message=str(e), schema_version=version
            )

    async def extract_features(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Runs Proposition, HyPE, and Domain extraction in parallel.
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
        
        domain_task = self._extract_domain_features(
            chunk_text=chunk.raw_content,
            doc_type=chunk.metadata.document_type
        )

        # Ensure complete isolation of parallel failures
        results = await asyncio.gather(
            prop_task, 
            hype_task, 
            domain_task, 
            return_exceptions=True
        )
        
        propositions = results[0] if not isinstance(results[0], Exception) else []
        questions = results[1] if not isinstance(results[1], Exception) else []
        
        if isinstance(results[2], Exception):
            logger.error(f"Domain extraction task crashed: {results[2]}")
            domain_result = DomainFeatureResult(
                data={}, failed=True, error_type="unknown_error", error_message=str(results[2]), schema_version="unknown"
            )
        else:
            domain_result = results[2]
        
        if isinstance(results[0], Exception):
            logger.error(f"Proposition extraction failed: {results[0]}")
        if isinstance(results[1], Exception):
            logger.error(f"HyPE generation failed: {results[1]}")
            
        hype_obj = HyPE(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            hypothesis_questions=questions
        )

        return {
            "propositions": propositions,
            "hype": hype_obj,
            "domain_features": domain_result
        }
