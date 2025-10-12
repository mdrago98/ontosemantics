from abc import ABC, abstractmethod
from typing import List, Dict

from bioengine.knowledge_engine.models.relationship import VerifiedRelationship


class KnowledgeStore(ABC):
    """Abstract base for knowledge storage backends"""

    @abstractmethod
    def store_relationship(self, relationship: VerifiedRelationship) -> bool:
        raise NotImplemented()

    @abstractmethod
    def query_relationships(self, subject: str = None, predicate: str = None,
                            object: str = None, min_confidence: float = 0.0,
                            limit: int = 100) -> List[StoredRelationship]:
        """Simple query with basic filters"""
        raise NotImplemented()

    @abstractmethod
    def get_context_for_entities(self, entities: List[str]) -> Dict:
        raise NotImplemented()
