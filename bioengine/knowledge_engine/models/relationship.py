from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ExtractedRelationship:
    """Relationship extracted from text"""
    subject: str
    predicate: str
    object: str
    evidence_text: str
    extraction_method: str
    confidence: float
    metadata: Dict = field(default_factory=dict)

@dataclass
class VerificationResult:
    """Result of relationship verification"""
    original_relationship: ExtractedRelationship
    is_valid: bool
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    verification_method: str
    ontological_support: Dict = field(default_factory=dict)

@dataclass
class VerifiedRelationship:
    """Relationship that passed verification"""
    extracted: ExtractedRelationship
    verification: VerificationResult
    final_confidence: float
    enrichments: Dict = field(default_factory=dict)  # Additional info from verification
