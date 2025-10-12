from abc import ABC, abstractmethod
from typing import List

from bioengine.knowledge_engine.models.entity import EntityValidation


class OntologyManager(ABC):
    """Abstract base for ontology management"""

    @abstractmethod
    def validate_entity(self, entity: str) -> EntityValidation:
        raise NotImplemented()

    @abstractmethod
    def get_entity_context(self, entity: str) -> EntityContext:
        raise NotImplemented()

    @abstractmethod
    def find_similar_entities(self, entity: str) -> List[SimilarEntity]:
        raise NotImplemented()