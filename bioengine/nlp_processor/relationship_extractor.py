from abc import ABC, abstractmethod
from typing import List, Dict

from knowledge_engine.models.relationship import ExtractedRelationship, VerificationResult


class RelationshipVerifier(ABC):
    """Abstract base for relationship verification methods"""

    @abstractmethod
    def verify_relationship(self, relationship: ExtractedRelationship, context: Dict = None) -> VerificationResult:
        raise NotImplementedError()

    @abstractmethod
    def get_verification_confidence(self, result: VerificationResult) -> float:
        raise NotImplemented()


class RelationshipExtractor(ABC):
    """Abstract base for relationship extraction methods"""

    @abstractmethod
    def extract_relationships(self, text: str, context: Dict = None) -> List[ExtractedRelationship]:
        raise NotImplementedError()


