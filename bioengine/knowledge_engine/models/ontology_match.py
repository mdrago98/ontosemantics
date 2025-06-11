from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OntologyMatch:
    entity_text: str
    ontology_id: str
    canonical_name: str
    biolink_type: str
    confidence: float
    synonyms: List[str]
    definition: Optional[str] = None
    parents: List[str] = None
    children: List[str] = None
