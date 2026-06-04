"""
WHAT: Extracts structured metadata from raw text during ingestion.
"""
import json
import logging
from litellm import acompletion
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

class RawMetadata(BaseModel):
    raw_jurisdiction: Optional[str]
    raw_document_type: Optional[str]

EXTRACTOR_PROMPT = """Extract the raw jurisdiction (e.g. 'State of New York', 'Delhi NCR') and document type (e.g. '10-K', 'Master Service Agreement') from this text.
If not found, return null.

Text: {text}

Output JSON:
{{
    "raw_jurisdiction": "...",
    "raw_document_type": "..."
}}
"""

async def extract_metadata(text: str, model: str = "gemini/gemini-2.5-flash") -> RawMetadata:
    """Runs during ingestion to extract raw strings before canonicalization."""
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": EXTRACTOR_PROMPT.format(text=text[:3000])}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content.strip())
        return RawMetadata(
            raw_jurisdiction=data.get("raw_jurisdiction"),
            raw_document_type=data.get("raw_document_type")
        )
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return RawMetadata(raw_jurisdiction=None, raw_document_type=None)
