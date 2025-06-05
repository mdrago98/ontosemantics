import pronto
from pathlib import Path
import requests
from biolink_model.datamodel.model import *

from knowledge_engine.models.ontology_match import OntologyMatch

OBO_URLS = {
        'mondo': 'http://purl.obolibrary.org/obo/mondo.obo',
        'hp': 'http://purl.obolibrary.org/obo/hp.obo',
        'chebi': 'http://purl.obolibrary.org/obo/chebi.obo',
        'go': 'http://purl.obolibrary.org/obo/go.obo',
        'cl': 'http://purl.obolibrary.org/obo/cl.obo',
        'uberon': 'http://purl.obolibrary.org/obo/uberon.obo',
        'doid': 'http://purl.obolibrary.org/obo/doid.obo'
}

TERM_MAPPING = {
    Disease: ['mondo', 'doid'],
    PhenotypicFeature: ['hp'],
    ChemicalEntity: ['chebi'],
    BiologicalProcess: ['go'],
    MolecularActivity: ['go'],
    CellularComponent: ['go'],
    Cell: ['cl'],
    OrganismalEntity: ['uberon']
}

class OntologyManager:
    def __init__(self, ontology_dir: Path = Path("../../data/ontologies"), obo_urls=None, term_mapping=None):
        if term_mapping is None:
            term_mapping = TERM_MAPPING
        if obo_urls is None:
            obo_urls = OBO_URLS
        self.ontology_dir = ontology_dir
        self.ontologies = {}
        self.term_mapping = term_mapping

        self.obo_urls = obo_urls

        self.load_ontologies()

    def download_and_load_ontologies(self, ontologies_to_load: List[str] = None):
        """Download OBO files and load with Pronto"""
        if ontologies_to_load is None:
            # just load essential ones
            ontologies_to_load = ['mondo', 'hp', 'chebi']  # Start small

        self.ontology_dir.mkdir(exist_ok=True)

        for onto_name in ontologies_to_load:
            obo_file = self.ontology_dir / f"{onto_name}.obo"

            if not obo_file.exists():
                print(f"Downloading {onto_name}...")
                response = requests.get(self.obo_urls[onto_name])
                with open(obo_file, 'wb') as f:
                    f.write(response.content)

            print(f"Loading {onto_name}...")
            try:
                self.ontologies[onto_name] = pronto.Ontology(str(obo_file))
                print(f"✅ Loaded {onto_name}: {len(self.ontologies[onto_name])} terms")
            except Exception as e:
                print(f"❌ Failed to load {onto_name}: {e}")

    def load_ontologies(self):
        """Load ontologies that are already downloaded"""
        for onto_name, url in self.obo_urls.items():
            obo_file = self.ontology_dir / f"{onto_name}.obo"
            if obo_file.exists():
                try:
                    print(f"Loading {onto_name}...")
                    self.ontologies[onto_name] = pronto.Ontology(str(obo_file))
                    print(f"✅ {onto_name}: {len(self.ontologies[onto_name])} terms")
                except Exception as e:
                    print(f"❌ Failed to load {onto_name}: {e}")

    def search_ontology(self, query: str, ontology_name: str, max_results: int = 3) -> List[OntologyMatch]:
        """Search within a specific ontology"""
        if ontology_name not in self.ontologies:
            return []

        ontology = self.ontologies[ontology_name]
        matches = []
        query_lower = query.lower()

        # Search through all terms
        for term in ontology.terms():
            score = 0.0

            # Check name (exact match gets highest score)
            if term.name and term.name.lower() == query_lower:
                score = 1.0
            elif term.name and query_lower in term.name.lower():
                score = 0.8
            elif term.name and term.name.lower() in query_lower:
                score = 0.7

            # Check synonyms
            if score < 0.8:  # Only check synonyms if no good name match
                for synonym in term.synonyms:
                    if synonym.description.lower() == query_lower:
                        score = max(score, 0.95)
                    elif query_lower in synonym.description.lower():
                        score = max(score, 0.75)

            # If we found a match, create OntologyMatch
            if score > 0.6:
                # Determine biolink type
                biolink_type = self._get_biolink_type(ontology_name, term)

                matches.append(OntologyMatch(
                    entity_text=query,
                    ontology_id=term.id,
                    canonical_name=term.name or str(term.id),
                    biolink_type=biolink_type,
                    confidence=score,
                    synonyms=[syn.description for syn in term.synonyms],
                    definition=term.definition.strip() if term.definition else None,
                    parents=[parent.name for parent in term.superclasses(with_self=False, distance=1)],
                    children=[child.name for child in term.subclasses(with_self=False, distance=1)]
                ))

        # Sort by confidence and return top results
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:max_results]

    def validate_and_enrich_entity(self, entity_text: str) -> List[OntologyMatch]:
        """Main method for RAG integration - search across all relevant ontologies"""
        all_matches = []

        # Search each loaded ontology
        for ontology_name in self.ontologies.keys():
            matches = self.search_ontology(entity_text, ontology_name)
            all_matches.extend(matches)

        # Sort all matches by confidence
        all_matches.sort(key=lambda x: x.confidence, reverse=True)

        return all_matches[:5]  # Return top 5 matches across all ontologies

    def _get_biolink_type(self, ontology_name: str, term) -> str:
        """Map ontology to BioLink type"""
        mapping = {
            'mondo': 'Disease',
            'doid': 'Disease',
            'hp': 'PhenotypicFeature',
            'chebi': 'ChemicalSubstance',
            'go': self._classify_go_term(term),
            'cl': 'Cell',
            'uberon': 'OrganismalEntity'
        }

        return mapping.get(ontology_name, 'NamedThing')

    def _classify_go_term(self, term) -> str:
        """Classify GO terms into BioLink types"""
        # Check GO namespace
        if hasattr(term, 'namespace'):
            if term.namespace == 'biological_process':
                return 'BiologicalProcess'
            elif term.namespace == 'molecular_function':
                return 'MolecularActivity'
            elif term.namespace == 'cellular_component':
                return 'CellularComponent'

        # Fallback to checking parents
        for parent in term.superclasses(with_self=False):
            if 'GO:0008150' in str(parent.id):  # biological_process
                return 'BiologicalProcess'
            elif 'GO:0003674' in str(parent.id):  # molecular_function
                return 'MolecularActivity'
            elif 'GO:0005575' in str(parent.id):  # cellular_component
                return 'CellularComponent'

        return 'BiologicalProcess'  # Default

    def get_term_hierarchy(self, ontology_id: str) -> Dict:
        """Get full hierarchy information for a term"""
        for ontology_name, ontology in self.ontologies.items():
            try:
                term = ontology[ontology_id]
                return {
                    'term': term.name,
                    'definition': term.definition.strip() if term.definition else None,
                    'parents': [{'id': p.id, 'name': p.name} for p in term.superclasses(with_self=False, distance=1)],
                    'children': [{'id': c.id, 'name': c.name} for c in term.subclasses(with_self=False, distance=1)],
                    'ancestors': [{'id': a.id, 'name': a.name} for a in term.superclasses(with_self=False)],
                    'descendants': [{'id': d.id, 'name': d.name} for d in term.subclasses(with_self=False)]
                }
            except KeyError:
                continue

        return {}