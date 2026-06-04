"""
WHAT: Canonical metadata schemas for global legal taxonomy.
"""
from typing import Optional
from pydantic import BaseModel, Field

class CanonicalMetadata(BaseModel):
    jurisdiction_id: Optional[str] = Field(None, description="e.g. IN-MH, US-NY")
    document_type_id: Optional[str] = Field(None, description="e.g. DOC-10K, DOC-MSA")
    court_id: Optional[str] = Field(None, description="e.g. IN-SC")
