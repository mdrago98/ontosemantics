from dataclasses import dataclass
from typing import List

from knowledge_engine.models.ontology_match import OntologyMatch


@dataclass
class EntityValidation:
    """Ontology validation result for entity"""
    entity_text: str
    is_valid: bool
    ontology_matches: List[OntologyMatch]
    confidence: float
    biolink_type: str = None

class EntityContext:
    pass

class SimilarEntity:
    pass
