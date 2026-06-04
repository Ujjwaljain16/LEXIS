from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    RETRIEVAL = "RETRIEVAL"
    MAP_PHASE = "MAP_PHASE"
    REDUCE_PHASE = "REDUCE_PHASE"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class BaseLexisResponse(BaseModel):
    """
    Standard envelope for all API responses, ensuring trace IDs are always present
    for production debugging.
    """
    request_id: str
    trace_id: str
    parent_trace_id: Optional[str] = None
    job_id: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None

class DeepModeEnqueueRequest(BaseModel):
    query: str
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
