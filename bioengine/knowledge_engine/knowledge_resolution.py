import re
import string
from typing import List, Dict, Tuple, Set, Optional
from difflib import SequenceMatcher
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from rapidfuzz import fuzz, process
import networkx as nx
import pickle
import json
from pathlib import Path

class OntologicalKnowledgeInjector:
    def __init__(self, ontology_graph: nx.Graph, max_context_length: int = 128):
        self.ontology_graph = ontology_graph
        self.max_context_length = max_context_length

        # Load scientific NLP model
        try:
            self.nlp = spacy.load("en_core_sci_lg")
        except OSError:
            print("Warning: en_core_sci_lg not found, using en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Build optimized lookup structures
        self._build_lookup_structures()

        # Initialize similarity components
        self._initialize_similarity_components()

        # Cache for expensive computations
        self._similarity_cache = {}
        self._mapping_cache = {}

    def _build_lookup_structures(self):
        """Build optimized data structures for fast entity lookup"""
        print("Building ontology lookup structures...")

        # Primary index: normalized term -> [ontology_ids]
        self.term_to_ids = defaultdict(list)

        # Secondary indices for different matching strategies
        self.exact_match_index = {}  # exact string -> ontology_id
        self.prefix_index = defaultdict(set)  # prefix -> set of ontology_ids
        self.suffix_index = defaultdict(set)  # suffix -> set of ontology_ids
        self.word_index = defaultdict(set)  # word -> set of ontology_ids
        self.abbreviation_index = {}  # abbreviation -> ontology_id

        # Metadata storage
        self.id_to_metadata = {}

        for node_id in self.ontology_graph.nodes():
            node_data = self.ontology_graph.nodes[node_id]

            # Extract all possible terms for this concept
            terms = self._extract_all_terms(node_id, node_data)

            # Store metadata
            self.id_to_metadata[node_id] = {
                'name': node_data.get('name', ''),
                'synonyms': node_data.get('synonyms', []),
                'definition': node_data.get('definition', ''),
                'ontology': node_data.get('ontology', ''),
                'all_terms': terms
            }

            # Index all terms
            for term in terms:
                normalized = self._normalize_text(term)
                if normalized:
                    self.term_to_ids[normalized].append(node_id)
                    self._index_term(term, node_id)

        print(f"✅ Indexed {len(self.term_to_ids)} unique terms from {len(self.ontology_graph.nodes())} concepts")

    def _extract_all_terms(self, node_id: str, node_data: Dict) -> List[str]:
        """Extract all possible terms (names, synonyms, abbreviations) for a concept"""
        terms = []

        # Primary name
        if node_data.get('name'):
            terms.append(node_data['name'])

        # Synonyms
        synonyms = node_data.get('synonyms', [])
        if isinstance(synonyms, list):
            terms.extend([s for s in synonyms if isinstance(s, str) and s.strip()])

        # Generate abbreviations and variants
        if node_data.get('name'):
            terms.extend(self._generate_term_variants(node_data['name']))

        # Add node ID itself (for exact ID matches)
        terms.append(node_id)

        return [term.strip() for term in terms if term and term.strip()]

    def _generate_term_variants(self, term: str) -> List[str]:
        """Generate common variants of a term (abbreviations, plurals, etc.)"""
        variants = []

        # Abbreviations (first letters of words)
        words = re.findall(r'\b[A-Za-z]+\b', term)
        if len(words) > 1:
            abbreviation = ''.join(word[0].upper() for word in words)
            variants.append(abbreviation)

            # Common abbreviation patterns
            if len(words) <= 4:  # Only for reasonable length terms
                variants.append(''.join(word[0].lower() for word in words))

        # Plural/singular variations
        if term.endswith('s') and len(term) > 3:
            variants.append(term[:-1])  # Remove 's'
        elif not term.endswith('s'):
            variants.append(term + 's')   # Add 's'

        # Remove parenthetical information
        clean_term = re.sub(r'\s*\([^)]*\)', '', term).strip()
        if clean_term != term:
            variants.append(clean_term)

        # Handle hyphenated terms
        if '-' in term:
            variants.append(term.replace('-', ' '))
            variants.append(term.replace('-', ''))

        return variants

    def _index_term(self, term: str, node_id: str):
        """Index a term using multiple strategies for fast lookup"""
        normalized = self._normalize_text(term)
        if not normalized:
            return

        # Exact match index
        self.exact_match_index[normalized] = node_id

        # Word-based index
        words = normalized.split()
        for word in words:
            if len(word) > 2:  # Skip very short words
                self.word_index[word].add(node_id)

        # Prefix/suffix indices (for partial matching)
        if len(normalized) >= 3:
            for i in range(3, min(len(normalized) + 1, 8)):  # Prefixes of length 3-7
                self.prefix_index[normalized[:i]].add(node_id)
            for i in range(3, min(len(normalized) + 1, 8)):  # Suffixes of length 3-7
                self.suffix_index[normalized[-i:]].add(node_id)

        # Abbreviation detection
        if len(normalized) <= 6 and normalized.isupper():
            self.abbreviation_index[normalized] = node_id

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent matching"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove punctuation but keep hyphens and periods in abbreviations
        text = re.sub(r'[^\w\s\-\.]', '', text)

        # Normalize special cases
        text = text.replace('α', 'alpha').replace('β', 'beta').replace('γ', 'gamma')

        return text

    def _initialize_similarity_components(self):
        """Initialize components for similarity calculation"""
        # Build TF-IDF vectorizer for semantic similarity
        all_terms = []
        for node_id, metadata in self.id_to_metadata.items():
            # Combine name, synonyms, and definition for rich representation
            text_parts = []
            if metadata['name']:
                text_parts.append(metadata['name'])
            text_parts.extend(metadata['synonyms'][:5])  # Limit synonyms
            if metadata['definition']:
                text_parts.append(metadata['definition'][:200])  # Limit definition length

            combined_text = ' '.join(text_parts)
            all_terms.append(combined_text)

        if all_terms:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )

            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_terms)
                self.node_ids_list = list(self.id_to_metadata.keys())
                print("✅ TF-IDF vectorizer initialized for semantic similarity")
            except Exception as e:
                print(f"Warning: Could not initialize TF-IDF vectorizer: {e}")
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None

    def _map_to_ontology(self, entity_text: str, entity_type: str = None) -> List[str]:
        """Enhanced entity mapping with multiple strategies and confidence scoring"""
        # Check cache first
        cache_key = (entity_text.lower(), entity_type)
        if cache_key in self._mapping_cache:
            return self._mapping_cache[cache_key]

        candidates = []
        normalized_entity = self._normalize_text(entity_text)

        # Strategy 1: Exact match (highest priority)
        exact_matches = self._exact_match_search(normalized_entity)
        candidates.extend([(node_id, 1.0, 'exact') for node_id in exact_matches])

        # Strategy 2: High-confidence fuzzy matching
        if len(candidates) < 3:
            fuzzy_matches = self._fuzzy_match_search(entity_text, threshold=0.85)
            candidates.extend(fuzzy_matches)

        # Strategy 3: Word-based matching
        if len(candidates) < 3:
            word_matches = self._word_based_search(normalized_entity)
            candidates.extend(word_matches)

        # Strategy 4: Semantic similarity (if available)
        if len(candidates) < 3 and self.tfidf_vectorizer:
            semantic_matches = self._semantic_similarity_search(entity_text)
            candidates.extend(semantic_matches)

        # Strategy 5: Abbreviation expansion
        abbreviation_matches = self._abbreviation_search(entity_text.upper())
        candidates.extend(abbreviation_matches)

        # Strategy 6: Partial matching (lowest priority)
        if len(candidates) < 3:
            partial_matches = self._partial_match_search(normalized_entity)
            candidates.extend(partial_matches)

        # Remove duplicates and sort by confidence
        unique_candidates = {}
        for node_id, confidence, method in candidates:
            if node_id not in unique_candidates or confidence > unique_candidates[node_id][0]:
                unique_candidates[node_id] = (confidence, method)

        # Sort by confidence and filter by entity type if provided
        sorted_candidates = []
        for node_id, (confidence, method) in unique_candidates.items():
            # Type filtering
            if entity_type and not self._matches_entity_type(node_id, entity_type):
                confidence *= 0.7  # Reduce confidence for type mismatch

            if confidence > 0.5:  # Minimum confidence threshold
                sorted_candidates.append((node_id, confidence, method))

        sorted_candidates.sort(key=lambda x: x[1], reverse=True)

        # Cache and return top 5 results
        result = [node_id for node_id, _, _ in sorted_candidates[:5]]
        self._mapping_cache[cache_key] = result

        return result

    def _exact_match_search(self, normalized_text: str) -> List[str]:
        """Find exact matches in the index"""
        matches = []

        # Direct lookup in exact match index
        if normalized_text in self.exact_match_index:
            matches.append(self.exact_match_index[normalized_text])

        # Check term to IDs mapping
        if normalized_text in self.term_to_ids:
            matches.extend(self.term_to_ids[normalized_text])

        return list(set(matches))  # Remove duplicates

    def _fuzzy_match_search(self, entity_text: str, threshold: float = 0.8) -> List[Tuple[str, float, str]]:
        """High-performance fuzzy matching using rapidfuzz"""
        matches = []

        # Use rapidfuzz for fast fuzzy matching
        all_terms = list(self.exact_match_index.keys())

        if all_terms:
            # Get top matches using rapidfuzz
            fuzzy_results = process.extract(
                entity_text.lower(),
                all_terms,
                scorer=fuzz.WRatio,
                limit=10
            )

            for term, score, _ in fuzzy_results:
                if score >= threshold * 100:  # rapidfuzz uses 0-100 scale
                    confidence = score / 100.0
                    node_id = self.exact_match_index[term]
                    matches.append((node_id, confidence, 'fuzzy'))

        return matches

    def _word_based_search(self, normalized_text: str) -> List[Tuple[str, float, str]]:
        """Search based on word overlap"""
        matches = []
        query_words = set(normalized_text.split())

        if not query_words:
            return matches

        candidate_nodes = set()
        for word in query_words:
            if word in self.word_index:
                candidate_nodes.update(self.word_index[word])

        # Score candidates based on word overlap
        for node_id in candidate_nodes:
            metadata = self.id_to_metadata.get(node_id, {})
            all_text = ' '.join([
                metadata.get('name', ''),
                ' '.join(metadata.get('synonyms', [])[:3])
            ])

            normalized_candidate = self._normalize_text(all_text)
            candidate_words = set(normalized_candidate.split())

            if candidate_words:
                overlap = len(query_words & candidate_words)
                union = len(query_words | candidate_words)
                jaccard_score = overlap / union if union > 0 else 0

                if jaccard_score > 0.4:
                    matches.append((node_id, jaccard_score, 'word_overlap'))

        return matches

    def _semantic_similarity_search(self, entity_text: str) -> List[Tuple[str, float, str]]:
        """Search using semantic similarity via TF-IDF"""
        if not self.tfidf_vectorizer or not hasattr(self, 'tfidf_matrix'):
            return []

        matches = []

        try:
            # Transform query text
            query_vector = self.tfidf_vectorizer.transform([entity_text])

            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Get top similar terms
            top_indices = np.argsort(similarities)[-10:][::-1]  # Top 10

            for idx in top_indices:
                similarity_score = similarities[idx]
                if similarity_score > 0.3:  # Threshold for semantic similarity
                    node_id = self.node_ids_list[idx]
                    matches.append((node_id, similarity_score, 'semantic'))

        except Exception as e:
            print(f"Warning: Semantic similarity search failed: {e}")

        return matches

    def _abbreviation_search(self, entity_text: str) -> List[Tuple[str, float, str]]:
        """Search for abbreviation matches"""
        matches = []

        # Direct abbreviation lookup
        if entity_text in self.abbreviation_index:
            node_id = self.abbreviation_index[entity_text]
            matches.append((node_id, 0.9, 'abbreviation'))

        # Generate possible abbreviations from entity text
        words = re.findall(r'\b[A-Za-z]+\b', entity_text)
        if len(words) > 1:
            generated_abbrev = ''.join(word[0].upper() for word in words)
            if generated_abbrev in self.abbreviation_index:
                node_id = self.abbreviation_index[generated_abbrev]
                matches.append((node_id, 0.8, 'generated_abbreviation'))

        return matches

    def _partial_match_search(self, normalized_text: str) -> List[Tuple[str, float, str]]:
        """Search using partial matching (prefixes/suffixes)"""
        matches = []

        # Prefix matching
        for i in range(min(len(normalized_text), 7), 2, -1):
            prefix = normalized_text[:i]
            if prefix in self.prefix_index:
                for node_id in self.prefix_index[prefix]:
                    confidence = i / len(normalized_text)  # Longer match = higher confidence
                    matches.append((node_id, confidence * 0.6, 'prefix'))
                break  # Take only the longest prefix match

        # Suffix matching
        for i in range(min(len(normalized_text), 7), 2, -1):
            suffix = normalized_text[-i:]
            if suffix in self.suffix_index:
                for node_id in self.suffix_index[suffix]:
                    confidence = i / len(normalized_text)
                    matches.append((node_id, confidence * 0.5, 'suffix'))
                break  # Take only the longest suffix match

        return matches

    def _matches_entity_type(self, node_id: str, entity_type: str) -> bool:
        """Check if a node matches the expected entity type"""
        metadata = self.id_to_metadata.get(node_id, {})
        ontology_name = metadata.get('ontology', '').lower()

        # Define type mappings
        type_mappings = {
            'disease': ['mondo', 'doid'],
            'chemical': ['chebi'],
            'drug': ['chebi'],
            'gene': ['hgnc', 'ensembl'],
            'protein': ['uniprot'],
            'phenotype': ['hp'],
            'anatomy': ['uberon'],
            'cell': ['cl'],
            'process': ['go']
        }

        entity_type_lower = entity_type.lower()
        expected_ontologies = type_mappings.get(entity_type_lower, [])

        return ontology_name in expected_ontologies if expected_ontologies else True

    def _similarity_score(self, text1: str, text2: str) -> float:
        """Enhanced similarity calculation using multiple methods"""
        # Check cache
        cache_key = (text1.lower(), text2.lower())
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # Handle exact matches
        if text1.lower() == text2.lower():
            self._similarity_cache[cache_key] = 1.0
            return 1.0

        scores = []

        # 1. String similarity (Jaro-Winkler)
        jaro_score = fuzz.ratio(text1.lower(), text2.lower()) / 100.0
        scores.append(jaro_score * 0.3)

        # 2. Token-based similarity
        token_score = fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100.0
        scores.append(token_score * 0.3)

        # 3. Partial ratio (substring matching)
        partial_score = fuzz.partial_ratio(text1.lower(), text2.lower()) / 100.0
        scores.append(partial_score * 0.2)

        # 4. Word overlap (Jaccard)
        words1 = set(self._normalize_text(text1).split())
        words2 = set(self._normalize_text(text2).split())

        if words1 and words2:
            jaccard = len(words1 & words2) / len(words1 | words2)
            scores.append(jaccard * 0.2)

        # Final weighted score
        final_score = sum(scores)

        # Cache and return
        self._similarity_cache[cache_key] = final_score
        return final_score

    def get_mapping_explanation(self, entity_text: str, mapped_ids: List[str]) -> Dict:
        """Provide explanation for why entities were mapped to specific ontology concepts"""
        explanations = {}

        for node_id in mapped_ids[:3]:  # Top 3 explanations
            metadata = self.id_to_metadata.get(node_id, {})

            # Calculate different similarity scores
            similarities = {}
            if metadata.get('name'):
                similarities['name'] = self._similarity_score(entity_text, metadata['name'])

            # Check synonym matches
            best_synonym_score = 0
            best_synonym = ""
            for synonym in metadata.get('synonyms', [])[:5]:
                if isinstance(synonym, str):
                    score = self._similarity_score(entity_text, synonym)
                    if score > best_synonym_score:
                        best_synonym_score = score
                        best_synonym = synonym

            if best_synonym_score > 0:
                similarities['best_synonym'] = best_synonym_score
                similarities['best_synonym_text'] = best_synonym

            explanations[node_id] = {
                'canonical_name': metadata.get('name', ''),
                'ontology': metadata.get('ontology', ''),
                'similarities': similarities,
                'definition': metadata.get('definition', '')[:100] + "..." if metadata.get('definition') else None
            }

        return explanations

    def extract_entity_context(self,
        entity_id: str, max_items: int = 6
    ) -> List[str]:
        """
        Extract ontological context for an entity to enrich text understanding.

        Args:
            entity_id: The ontology ID (e.g., "MONDO:0005015")
            ontology_graph: NetworkX graph with ontology data
            max_items: Maximum number of context items to return

        Returns:
            List of context strings like ["diabetes is_a metabolic_disorder", "insulin treats diabetes"]
        """
        if entity_id not in self.ontology_graph:
            return []

        context = []
        node_data = self.ontology_graph.nodes.get(entity_id, {})
        entity_name = node_data.get("name", entity_id.split(":")[-1])

        # 1. Definition context (most important)
        definition = node_data.get("definition", "")
        if definition:
            # Extract key relationship from definition
            patterns = [
                r"characterized by (.+?)(?:\.|,)",
                r"caused by (.+?)(?:\.|,)",
                r"results in (.+?)(?:\.|,)",
                r"associated with (.+?)(?:\.|,)",
            ]
            for pattern in patterns:
                match = re.search(pattern, definition, re.IGNORECASE)
                if match:
                    context.append(
                        f"{entity_name} {pattern.split('(')[0].strip()} {match.group(1).strip()}"
                    )
                    break

        # 2. Parent relationships (is-a hierarchy)
        parents = list(self.ontology_graph.predecessors(entity_id))
        for parent_id in parents[:2]:  # Max 2 parents
            parent_name = self.ontology_graph.nodes.get(parent_id, {}).get("name")
            if parent_name and parent_name != entity_name:
                context.append(f"{entity_name} is_a {parent_name}")

        # 3. Direct relationships with other entities
        neighbors = list(self.ontology_graph.neighbors(entity_id))
        for neighbor_id in neighbors[:3]:  # Max 3 neighbors
            neighbor_data = self.ontology_graph.nodes.get(neighbor_id, {})
            neighbor_name = neighbor_data.get("name")
            neighbor_ontology = neighbor_data.get("ontology", "")
            entity_ontology = node_data.get("ontology", "")

            if neighbor_name and neighbor_name != entity_name:
                # Infer relationship type based on ontology domains
                if entity_ontology == "mondo" and neighbor_ontology == "chebi":
                    context.append(f"{neighbor_name} treats {entity_name}")
                elif entity_ontology == "chebi" and neighbor_ontology == "mondo":
                    context.append(f"{entity_name} treats {neighbor_name}")
                elif entity_ontology == "mondo" and neighbor_ontology == "hp":
                    context.append(f"{entity_name} causes {neighbor_name}")
                else:
                    context.append(f"{entity_name} related_to {neighbor_name}")

        # Remove duplicates and return top items
        unique_context = []
        seen = set()
        for item in context:
            if item not in seen and len(item.strip()) > 5:
                unique_context.append(item.strip())
                seen.add(item)
                if len(unique_context) >= max_items:
                    break

        return unique_context

    def enrich_text_with_ontology(self, text: str, ontology_manager) -> Tuple[str, Dict]:
        """Enhanced text enrichment with better entity mapping"""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        ontological_context = []
        entity_mappings = {}
        mapping_explanations = {}

        for entity_text, entity_type in entities:
            # Enhanced mapping
            mapped_entities = self._map_to_ontology(entity_text, entity_type)

            if mapped_entities:
                best_match = mapped_entities[0]
                entity_mappings[entity_text.lower()] = best_match

                # Get context for the best match
                context = self.extract_entity_context(best_match)
                ontological_context.extend(context[:3])  # Limit context per entity

                # Get explanation
                explanations = self.get_mapping_explanation(entity_text, mapped_entities[:1])
                mapping_explanations[entity_text] = explanations

        # Create enriched text with controlled context
        enriched_text = text
        if ontological_context:
            # Select most relevant context (avoid repetition)
            unique_context = list(dict.fromkeys(ontological_context))[:8]  # Max 8 context items
            context_str = " [ONTO] " + " ; ".join(unique_context)
            enriched_text = text + context_str

        return enriched_text, {
            'entity_mappings': entity_mappings,
            'mapping_explanations': mapping_explanations,
            'context_items': len(ontological_context)
        }

    def save_lookup_structures(self, filepath: str):
        """Save precomputed lookup structures for faster loading"""
        data = {
            'term_to_ids': dict(self.term_to_ids),
            'exact_match_index': self.exact_match_index,
            'word_index': {k: list(v) for k, v in self.word_index.items()},
            'abbreviation_index': self.abbreviation_index,
            'id_to_metadata': self.id_to_metadata
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"✅ Saved lookup structures to {filepath}")

    def load_lookup_structures(self, filepath: str) -> bool:
        """Load precomputed lookup structures"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.term_to_ids = defaultdict(list, data['term_to_ids'])
            self.exact_match_index = data['exact_match_index']
            self.word_index = defaultdict(set, {k: set(v) for k, v in data['word_index'].items()})
            self.abbreviation_index = data['abbreviation_index']
            self.id_to_metadata = data['id_to_metadata']

            print(f"✅ Loaded lookup structures from {filepath}")
            return True

        except Exception as e:
            print(f"❌ Failed to load lookup structures: {e}")
            return False
