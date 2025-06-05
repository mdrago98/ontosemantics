from dataclasses import field, dataclass
from datetime import datetime
from typing import Optional, Dict, List


@dataclass
class StoredEntity:
    """An entity stored in your knowledge graph"""

    # Basic identity
    text: str  # "diabetes" (original text from paper)
    canonical_name: str  # "diabetes mellitus" (standardized name)

    # Ontology information
    ontology_id: str = None  # "MONDO:0005015" (from MONDO ontology)
    biolink_type: str = None  # "Disease" (BioLink category)

    # Simple metadata
    confidence: float = 1.0  # How confident we are this is correct

@dataclass
class StoredRelationship:
    """Represents a stored relationship"""
    subject: StoredEntity
    predicate: str
    object: StoredEntity
    confidence: float
    evidence: str