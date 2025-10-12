from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from bioengine.knowledge_engine.models.relationship import ExtractedRelationship
from bioengine.knowledge_engine.models.stored_relationship import StoredRelationship


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    support: int  # Total gold standard relations

@dataclass
class DetailedEvaluation:
    """Detailed evaluation results"""
    overall_metrics: PerformanceMetrics
    per_predicate_metrics: Dict[str, PerformanceMetrics]
    per_entity_type_metrics: Dict[str, PerformanceMetrics]
    confusion_matrix: Dict
    missed_relations: List[Dict]
    false_positive_relations: List[Dict]
    confidence_analysis: Dict

class RelationshipEvaluator:
    """Evaluate extracted relationships against gold standard"""

    def __init__(self, matching_strategy: str = "exact"):
        """
        Args:
            matching_strategy: "exact", "fuzzy", or "semantic"
        """
        self.matching_strategy = matching_strategy

    def evaluate(self,
                 extracted_relations: List[ExtractedRelationship],
                 gold_relations: List[StoredRelationship]) -> DetailedEvaluation:
        """Main evaluation method"""

        # Convert to comparable format
        extracted_triples = self._extract_triples(extracted_relations)
        gold_triples = self._extract_gold_triples(gold_relations)

        # Find matches
        matches = self._find_matches(extracted_triples, gold_triples)

        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(matches, extracted_triples, gold_triples)

        # Calculate per-predicate metrics
        per_predicate_metrics = self._calculate_per_predicate_metrics(
            matches, extracted_triples, gold_triples
        )

        # Calculate per-entity-type metrics (if available)
        per_entity_type_metrics = self._calculate_per_entity_type_metrics(
            matches, extracted_relations, gold_relations
        )

        # Generate detailed analysis
        confusion_matrix = self._generate_confusion_matrix(matches, extracted_triples, gold_triples)
        missed_relations = self._find_missed_relations(matches, gold_triples)
        false_positive_relations = self._find_false_positives(matches, extracted_triples)
        confidence_analysis = self._analyze_confidence(extracted_relations, matches)

        return DetailedEvaluation(
            overall_metrics=overall_metrics,
            per_predicate_metrics=per_predicate_metrics,
            per_entity_type_metrics=per_entity_type_metrics,
            confusion_matrix=confusion_matrix,
            missed_relations=missed_relations,
            false_positive_relations=false_positive_relations,
            confidence_analysis=confidence_analysis
        )

    def _extract_triples(self, extracted_relations: List[ExtractedRelationship]) -> List[Dict]:
        """Convert ExtractedRelationship objects to comparable triples"""
        triples = []
        for rel in extracted_relations:
            triples.append({
                'subject': self._normalize_entity(rel.subject),
                'predicate': self._normalize_predicate(rel.predicate),
                'object': self._normalize_entity(rel.object),
                'confidence': rel.confidence,
                'original': rel
            })
        return triples

    def _extract_gold_triples(self, gold_relations: List[StoredRelationship]) -> List[Dict]:
        """Convert StoredRelationship objects to comparable triples"""
        triples = []
        for rel in gold_relations:
            triples.append({
                'subject': self._normalize_entity(rel.subject.text),
                'predicate': self._normalize_predicate(rel.predicate),
                'object': self._normalize_entity(rel.object.text),
                'confidence': 1.0,  # Gold standard assumed perfect
                'original': rel
            })
        return triples

    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity text for comparison"""
        # Convert to lowercase and strip whitespace
        normalized = entity.lower().strip()

        # Remove common variations
        normalized = normalized.replace('-', ' ')
        normalized = normalized.replace('_', ' ')

        # Handle common medical abbreviations
        abbreviation_map = {
            'mi': 'myocardial infarction',
            'dm': 'diabetes mellitus',
            'cad': 'coronary artery disease',
            'cvd': 'cardiovascular disease',
            # Add more as needed
        }

        return abbreviation_map.get(normalized, normalized)

    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate for comparison"""
        normalized = predicate.lower().strip()

        # Map similar predicates
        predicate_map = {
            'causes': 'causes',
            'leads_to': 'causes',
            'results_in': 'causes',
            'associated_with': 'associated_with',
            'correlated_with': 'associated_with',
            'treats': 'treats',
            'therapy_for': 'treats',
            'increases_risk_for': 'increases_risk_for',
            'risk_factor_for': 'increases_risk_for',
        }

        return predicate_map.get(normalized, normalized)

    def _find_matches(self, extracted_triples: List[Dict], gold_triples: List[Dict]) -> Dict:
        """Find matching relationships between extracted and gold standard"""

        matches = {
            'true_positives': [],
            'false_positives': [],
            'false_negatives': []
        }

        # Convert gold triples to set for faster lookup
        gold_set = set()
        gold_dict = {}

        for gold in gold_triples:
            key = (gold['subject'], gold['predicate'], gold['object'])
            gold_set.add(key)
            gold_dict[key] = gold

        # Find true positives and false positives
        matched_gold = set()

        for extracted in extracted_triples:
            key = (extracted['subject'], extracted['predicate'], extracted['object'])

            if key in gold_set:
                # True positive
                matches['true_positives'].append({
                    'extracted': extracted,
                    'gold': gold_dict[key]
                })
                matched_gold.add(key)
            else:
                # Check for fuzzy matches if strategy allows
                fuzzy_match = self._find_fuzzy_match(extracted, gold_triples)
                if fuzzy_match:
                    matches['true_positives'].append({
                        'extracted': extracted,
                        'gold': fuzzy_match,
                        'match_type': 'fuzzy'
                    })
                    matched_gold.add((fuzzy_match['subject'], fuzzy_match['predicate'], fuzzy_match['object']))
                else:
                    # False positive
                    matches['false_positives'].append(extracted)

        # Find false negatives (gold relations not matched)
        for gold in gold_triples:
            key = (gold['subject'], gold['predicate'], gold['object'])
            if key not in matched_gold:
                matches['false_negatives'].append(gold)

        return matches

    def _find_fuzzy_match(self, extracted: Dict, gold_triples: List[Dict]) -> Dict:
        """Find fuzzy matches for extracted relation"""
        if self.matching_strategy == "exact":
            return None

        # Simple fuzzy matching - could be enhanced with embeddings
        for gold in gold_triples:
            if (self._entities_similar(extracted['subject'], gold['subject']) and
                    extracted['predicate'] == gold['predicate'] and
                    self._entities_similar(extracted['object'], gold['object'])):
                return gold

        return None

    def _entities_similar(self, entity1: str, entity2: str, threshold: float = 0.8) -> bool:
        """Check if two entities are similar (simple string similarity)"""
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, entity1, entity2).ratio()
        return similarity >= threshold

    def _calculate_metrics(self, matches: Dict, extracted_triples: List[Dict],
                           gold_triples: List[Dict]) -> PerformanceMetrics:
        """Calculate precision, recall, F1"""

        tp = len(matches['true_positives'])
        fp = len(matches['false_positives'])
        fn = len(matches['false_negatives'])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return PerformanceMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            support=len(gold_triples)
        )

    def _calculate_per_predicate_metrics(self, matches: Dict, extracted_triples: List[Dict],
                                         gold_triples: List[Dict]) -> Dict[str, PerformanceMetrics]:
        """Calculate metrics for each predicate type"""

        # Group by predicate
        predicates = set()
        for triple in extracted_triples + gold_triples:
            predicates.add(triple['predicate'])

        per_predicate = {}

        for predicate in predicates:
            # Filter matches for this predicate
            pred_tp = [m for m in matches['true_positives']
                       if m['extracted']['predicate'] == predicate]
            pred_fp = [t for t in matches['false_positives']
                       if t['predicate'] == predicate]
            pred_fn = [t for t in matches['false_negatives']
                       if t['predicate'] == predicate]

            tp = len(pred_tp)
            fp = len(pred_fp)
            fn = len(pred_fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_predicate[predicate] = PerformanceMetrics(
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                support=len([t for t in gold_triples if t['predicate'] == predicate])
            )

        return per_predicate

    def _calculate_per_entity_type_metrics(self, matches: Dict,
                                           extracted_relations: List[ExtractedRelationship],
                                           gold_relations: List[StoredRelationship]) -> Dict[str, PerformanceMetrics]:
        """Calculate metrics per entity type if available"""
        # This would require entity type information in your data
        # Placeholder for now
        return {}

    def _generate_confusion_matrix(self, matches: Dict, extracted_triples: List[Dict],
                                   gold_triples: List[Dict]) -> Dict:
        """Generate confusion matrix for predicates"""

        # Create predicate confusion matrix
        predicates = set()
        for triple in extracted_triples + gold_triples:
            predicates.add(triple['predicate'])

        predicates = sorted(list(predicates))
        confusion = defaultdict(lambda: defaultdict(int))

        # True positives (correct predictions)
        for match in matches['true_positives']:
            pred = match['extracted']['predicate']
            confusion[pred][pred] += 1

        # False positives (predicted but not in gold)
        for fp in matches['false_positives']:
            pred = fp['predicate']
            confusion[pred]['NOT_IN_GOLD'] += 1

        # False negatives (in gold but not predicted)
        for fn in matches['false_negatives']:
            gold_pred = fn['predicate']
            confusion['NOT_PREDICTED'][gold_pred] += 1

        return dict(confusion)

    def _find_missed_relations(self, matches: Dict, gold_triples: List[Dict]) -> List[Dict]:
        """Find relations that were missed (false negatives)"""
        return [triple for triple in matches['false_negatives']]

    def _find_false_positives(self, matches: Dict, extracted_triples: List[Dict]) -> List[Dict]:
        """Find relations that were incorrectly extracted (false positives)"""
        return [triple for triple in matches['false_positives']]

    def _analyze_confidence(self, extracted_relations: List[ExtractedRelationship],
                            matches: Dict) -> Dict:
        """Analyze confidence scores of extracted relations"""

        tp_confidences = [m['extracted']['confidence'] for m in matches['true_positives']]
        fp_confidences = [rel['confidence'] for rel in matches['false_positives']]

        return {
            'tp_confidence_mean': np.mean(tp_confidences) if tp_confidences else 0,
            'tp_confidence_std': np.std(tp_confidences) if tp_confidences else 0,
            'fp_confidence_mean': np.mean(fp_confidences) if fp_confidences else 0,
            'fp_confidence_std': np.std(fp_confidences) if fp_confidences else 0,
            'tp_count': len(tp_confidences),
            'fp_count': len(fp_confidences)
        }

def print_evaluation_results(evaluation: DetailedEvaluation):
    """Pretty print evaluation results"""

    print("🎯 RELATIONSHIP EXTRACTION EVALUATION RESULTS")
    print("=" * 60)

    # Overall metrics
    overall = evaluation.overall_metrics
    print("\n📊 Overall Performance:")
    print(f"   Precision: {overall.precision:.3f}")
    print(f"   Recall:    {overall.recall:.3f}")
    print(f"   F1-Score:  {overall.f1_score:.3f}")
    print(f"   Support:   {overall.support} gold relations")

    # Per-predicate breakdown
    print("\n📋 Per-Predicate Performance:")
    for predicate, metrics in evaluation.per_predicate_metrics.items():
        print(f"   {predicate:20s} | P: {metrics.precision:.3f} | R: {metrics.recall:.3f} | F1: {metrics.f1_score:.3f} | Support: {metrics.support}")

    # Confidence analysis
    conf = evaluation.confidence_analysis
    print("\n🎲 Confidence Analysis:")
    print(f"   True Positives  - Mean: {conf['tp_confidence_mean']:.3f} ± {conf['tp_confidence_std']:.3f}")
    print(f"   False Positives - Mean: {conf['fp_confidence_mean']:.3f} ± {conf['fp_confidence_std']:.3f}")

    # Error analysis
    print("\n❌ Error Analysis:")
    print(f"   Missed Relations (FN): {len(evaluation.missed_relations)}")
    print(f"   False Positives (FP):  {len(evaluation.false_positive_relations)}")

    if evaluation.missed_relations:
        print("\n   Top Missed Relations:")
        for i, missed in enumerate(evaluation.missed_relations[:5]):
            print(f"     {i+1}. {missed['subject']} --{missed['predicate']}--> {missed['object']}")

    if evaluation.false_positive_relations:
        print("\n   Top False Positives:")
        for i, fp in enumerate(evaluation.false_positive_relations[:5]):
            print(f"     {i+1}. {fp['subject']} --{fp['predicate']}--> {fp['object']} (conf: {fp['confidence']:.3f})")

# Usage example:
"""
# Evaluate your extraction results
evaluator = RelationshipEvaluator(matching_strategy="fuzzy")
evaluation = evaluator.evaluate(extracted_relations, gold_standard_relations)

# Print results
print_evaluation_results(evaluation)

# Access specific metrics
print(f"Overall F1-Score: {evaluation.overall_metrics.f1_score:.3f}")
print(f"'causes' F1-Score: {evaluation.per_predicate_metrics['causes'].f1_score:.3f}")
"""