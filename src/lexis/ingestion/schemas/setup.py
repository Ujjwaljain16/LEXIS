from .registry import FeatureSchemaRegistry
from .definitions import LEGAL_SCHEMA, FINANCIAL_SCHEMA

def create_default_registry() -> FeatureSchemaRegistry:
    """
    Factory function for the FeatureSchemaRegistry.
    Uses Dependency Injection pattern to avoid global state and hidden side-effects.
    """
    registry = FeatureSchemaRegistry()
    
    # Register core schemas
    registry.register(
        doc_type="contract", 
        schema=LEGAL_SCHEMA, 
        version="v1"
    )
    registry.register(
        doc_type="regulation", 
        schema=LEGAL_SCHEMA, 
        version="v1"
    )
    registry.register(
        doc_type="10-k", 
        schema=FINANCIAL_SCHEMA, 
        version="v1"
    )
    registry.register(
        doc_type="financial", 
        schema=FINANCIAL_SCHEMA, 
        version="v1"
    )
    
    return registry
