"""
WHAT: Config-driven trust scoring for web fallback sources.
WHY: Web retrieval can pollute the corpus. High trust sources get score=1.0, blogs get 0.2.
HOW: Loads trusted_sources.yaml to multiply CRAG scores.
"""
import os
import yaml
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_TRUST = 0.2  # Untrusted domains get heavily penalized
TRUST_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "trusted_sources.yaml")

class TrustScorer:
    def __init__(self, config_path: str = TRUST_CONFIG_PATH):
        self.trusted_sources = self._load_config(config_path)

    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            # Fallback default if file is missing
            logger.warning(f"Trust config not found at {path}, using defaults.")
            return {
                "sec.gov": 1.0,
                "law.cornell.edu": 0.98,
                "supremecourt.gov": 1.0,
                "congress.gov": 0.95
            }
            
        with open(path, "r") as f:
            try:
                data = yaml.safe_load(f)
                return data.get("trusted_sources", {})
            except Exception as e:
                logger.error(f"Failed to parse trust config: {e}")
                return {}

    def get_trust_score(self, url: str) -> float:
        """
        Extracts domain from URL and returns the configured trust score.
        """
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
                
            # Check for exact or suffix match (e.g., "investor.sec.gov" matches "sec.gov")
            for trusted_domain, score in self.trusted_sources.items():
                if domain == trusted_domain or domain.endswith(f".{trusted_domain}"):
                    return float(score)
                    
            return DEFAULT_TRUST
        except Exception:
            return DEFAULT_TRUST

# Singleton instance
_scorer = None
def get_scorer() -> TrustScorer:
    global _scorer
    if _scorer is None:
        _scorer = TrustScorer()
    return _scorer
