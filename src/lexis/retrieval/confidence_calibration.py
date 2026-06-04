"""
WHAT: Converts raw retrieval heuristics into calibrated probability scores.
WHY: Raw RRF/CrossEncoder scores are not true probabilities. Platt Scaling fixes this.
HOW: Uses scikit-learn LogisticRegression or IsotonicRegression.
"""
import logging
from typing import Protocol
from datetime import datetime
from pydantic import BaseModel
import numpy as np

# We import scikit-learn conditionally or assume it's installed.
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    pass

logger = logging.getLogger(__name__)

class CalibrationModelMetadata(BaseModel):
    dataset_version: str
    trained_at: datetime
    sample_count: int
    algorithm: str

class ConfidenceCalibrator(Protocol):
    def calibrate(self, top1: float, spread: float, agreement: float) -> float:
        """Takes raw heuristics and returns a probability between 0.0 and 1.0"""
        ...
    
    def get_metadata(self) -> CalibrationModelMetadata:
        ...

class PlattScalingCalibrator:
    """Uses Logistic Regression to map heuristics to probabilities."""
    
    def __init__(self, metadata: CalibrationModelMetadata, model=None):
        self.metadata = metadata
        self.model = model  # Trained sklearn LogisticRegression instance
        
    def calibrate(self, top1: float, spread: float, agreement: float) -> float:
        if not self.model:
            # Fallback if not trained: simple weighted sum (uncalibrated)
            return min(1.0, max(0.0, (top1 * 0.6) + (spread * 0.2) + (agreement * 0.2)))
            
        features = np.array([[top1, spread, agreement]])
        # predict_proba returns [prob_negative, prob_positive]
        return float(self.model.predict_proba(features)[0][1])
        
    def get_metadata(self) -> CalibrationModelMetadata:
        return self.metadata

def get_default_calibrator() -> ConfidenceCalibrator:
    """
    Factory method to return the active calibrator.
    In production, this would load a pickled sklearn model trained on the Eval Harness data.
    """
    meta = CalibrationModelMetadata(
        dataset_version="cuad_v1",
        trained_at=datetime.utcnow(),
        sample_count=0,
        algorithm="fallback_weights"
    )
    return PlattScalingCalibrator(metadata=meta)
