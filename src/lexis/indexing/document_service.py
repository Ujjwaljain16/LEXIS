from pydantic import BaseModel, Field
from typing import Dict, Optional, Tuple

class BoundingBox(BaseModel):
    """Represents a bounding box on a PDF page."""
    x0: float
    y0: float
    x1: float
    y1: float

class CitationReference(BaseModel):
    """
    A first-class object representing a deterministic citation to a specific bounding box 
    on a specific page of a specific document.
    """
    document_id: str
    page: int
    bbox: BoundingBox
    text_span: str
    pqac_id: str
    confidence_score: Optional[float] = None
