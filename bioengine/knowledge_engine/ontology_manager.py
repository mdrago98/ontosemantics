import httpx
import pronto
from pathlib import Path
import re
import tqdm
from biolink_model.datamodel.model import *
from typing import Dict, List, Optional
import owlready2 as owl
import networkx as nx
import pickle

from knowledge_engine.ontology_embeddings import OntologyEmbeddingMixin
from knowledge_engine.models.ontology_match import OntologyMatch

# Extended URL mapping to include OWL versions
OBO_URLS = {
    'mondo': 'http://purl.obolibrary.org/obo/mondo.obo',
    'hp': 'http://purl.obolibrary.org/obo/hp.obo',
    'go': 'https://purl.obolibrary.org/obo/go/go-basic.obo',
    'cl': 'http://purl.obolibrary.org/obo/cl.obo',
    'uberon': 'http://purl.obolibrary.org/obo/uberon.obo',
    'chebi': 'http://purl.obolibrary.org/obo/chebi.obo'
}

# OWL versions (often more complete)
OWL_URLS = {
    'mondo': 'http://purl.obolibrary.org/obo/mondo/mondo-full.owl',
    'hp': 'https://purl.obolibrary.org/obo/hp/hp-full.owl',
    'go': 'http://purl.obolibrary.org/obo/go/go-full.owl',
    'cl': 'http://purl.obolibrary.org/obo/cl.owl',
    'uberon': 'http://purl.obolibrary.org/obo/uberon.owl',
    'chebi': 'http://purl.obolibrary.org/obo/chebi.owl',
}

TERM_MAPPING = {
    Disease: ['mondo'],
    PhenotypicFeature: ['hp'],
    ChemicalEntity: ['chebi'],
    BiologicalProcess: ['go'],
    MolecularActivity: ['go'],
    CellularComponent: ['go'],
    Cell: ['cl'],
    OrganismalEntity: ['uberon']
}

class OntologyManager(OntologyEmbeddingMixin):
    def __init__(self,
                 ontology_dir: Path = Path("../../data/ontologies"),
                 use_owlready: bool = True,
                 use_pronto: bool = False,
                 obo_urls: Dict = None,
                 owl_urls: Dict = None,
                 term_mapping: Dict = None):
        super().__init__()
        self.ontology_dir = ontology_dir
        self.use_owlready = use_owlready
        self.use_pronto = use_pronto

        # Initialize URL mappings
        self.obo_urls = obo_urls or OBO_URLS
        self.owl_urls = owl_urls or OWL_URLS
        self.term_mapping = term_mapping or TERM_MAPPING

        # Storage for different ontology types
        self.pronto_ontologies = {}
        self.owlready_ontologies = {}
        self.owl_world = None

        # NetworkX graph for efficient querying
        self.ontology_graph = nx.DiGraph()
        self.term_to_ontology = {}

        # Initialize owlready2 world
        if self.use_owlready:
            self.owl_world = owl.World()

        self.load_ontologies()

    async def download_and_load_ontologies(self, ontologies_to_load: List[str] = None):
        """Download both OBO and OWL files and load with appropriate parsers"""
        if ontologies_to_load is None:
            ontologies_to_load = set(self.obo_urls.keys()) | set(self.owl_urls.keys())

        self.ontology_dir.mkdir(exist_ok=True)

        async with httpx.AsyncClient(timeout=300.0) as client:  # Extended timeout for large ontologies
            failed = {}

            for onto_name in tqdm.tqdm(ontologies_to_load, desc="Downloading ontologies"):
                # Download OBO if using pronto
                if self.use_pronto and onto_name in self.obo_urls:
                    obo_file = self.ontology_dir / f"{onto_name}.obo"
                    await self._download_file(client, self.obo_urls[onto_name], obo_file, onto_name, "OBO")

                # Download OWL if using owlready2
                if self.use_owlready and onto_name in self.owl_urls:
                    owl_file = self.ontology_dir / f"{onto_name}.owl"
                    await self._download_file(client, self.owl_urls[onto_name], owl_file, onto_name, "OWL")

            # Load after downloading
            self.load_ontologies()

    async def _download_file(self, client: httpx.AsyncClient, url: str, filepath: Path, name: str, format_type: str):
        """Helper to download a single file"""
        if not filepath.exists():
            try:
                print(f"Downloading {name}.{format_type.lower()}...")
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded {name}.{format_type.lower()}")

            except Exception as e:
                print(f"❌ Failed to download {name}.{format_type.lower()}: {e}")

    def load_ontologies(self):
        """Load ontologies with both pronto and owlready2"""
        print("Loading ontologies...")

        # Load with pronto (OBO files)
        if self.use_pronto:
            self._load_pronto_ontologies()

        # Load with owlready2 (OWL files)
        if self.use_owlready:
            self._load_owlready_ontologies()

        # Build unified graph
        self._build_unified_graph()

        print(f"✅ Loaded {len(self.pronto_ontologies)} pronto + {len(self.owlready_ontologies)} owlready2 ontologies")
        print(f"📊 Total terms in graph: {len(self.ontology_graph.nodes())}")

    def _load_pronto_ontologies(self):
        """Load OBO files with pronto"""
        for onto_name in self.obo_urls.keys():
            obo_file = self.ontology_dir / f"{onto_name}.obo"
            if obo_file.exists():
                try:
                    print(f"Loading {onto_name} with pronto...")
                    self.pronto_ontologies[onto_name] = pronto.Ontology(str(obo_file))
                    print(f"✅ {onto_name}: {len(self.pronto_ontologies[onto_name])} terms")
                except Exception as e:
                    print(f"❌ Failed to load {onto_name} with pronto: {e}")

    def _load_owlready_ontologies(self):
        """Load OWL files with owlready2"""
        for onto_name in self.owl_urls.keys():
            owl_file = self.ontology_dir / f"{onto_name}.owl"
            if owl_file.exists():
                try:
                    print(f"Loading {onto_name} with owlready2...")

                    # Create namespace for this ontology
                    onto_iri = f"http://purl.obolibrary.org/obo/{onto_name}.owl"
                    onto = self.owl_world.get_ontology(onto_iri)
                    onto.load()

                    self.owlready_ontologies[onto_name] = onto

                    # Count classes (terms)
                    term_count = len(list(onto.classes()))
                    print(f"✅ {onto_name}: {term_count} classes")

                except Exception as e:
                    print(f"❌ Failed to load {onto_name} with owlready2: {e}")

    def _iri_to_curie(self, iri_or_name: str) -> str:
        """Normalize OWLReady names/IRIs like MONDO_0000001 → MONDO:0000001; otherwise pass through."""
        if iri_or_name is None:
            return None
        s = str(iri_or_name)
        # If already a CURIE like HP:0000001
        if re.match(r'^[A-Za-z]+:\d+$', s):
            return s
        # Names like MONDO_0000001 or IRIs ending with that pattern
        m = re.search(r'([A-Za-z]+)_(\d+)$', s)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
        # Fallback: try /obo/PREFIX_0000001 in IRI
        m = re.search(r'/obo/([A-Za-z]+)_(\d+)', s)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
        return s

    def _build_unified_graph(self):
        """Build a unified NetworkX graph from all loaded ontologies"""
        print("Building unified ontology graph...")

        # Add terms from pronto ontologies
        for onto_name, ontology in self.pronto_ontologies.items():
            for term in ontology.terms():
                term_id = self._iri_to_curie(str(term.id))
                self.ontology_graph.add_node(
                    term_id,
                    name=term.name,
                    ontology=onto_name,
                    definition=term.definition.strip() if term.definition else None,
                    synonyms=[syn.description for syn in term.synonyms]
                )
                self.term_to_ontology[term_id] = onto_name

                # is_a parents
                for parent in term.superclasses(with_self=False, distance=1):
                    parent_id = self._iri_to_curie(str(parent.id))
                    self.ontology_graph.add_edge(parent_id, term_id, relation='is_a', source='pronto')

                # other OBO relationships (typed)
                # term.relationships() may not exist; use term.relations (dict {Relationship: set(Term)})
                try:
                    for rel, targets in getattr(term, "relations", {}).items():
                        rel_id = getattr(rel, "id", None) or getattr(rel, "name", None) or str(rel)
                        rel_id = str(rel_id)
                        for tgt in targets:
                            tgt_id = self._iri_to_curie(str(tgt.id))
                            # direction is subject (term) -> object (tgt)
                            self.ontology_graph.add_edge(term_id, tgt_id, relation=rel_id, source='pronto')
                except Exception:
                    pass
        # Add terms from owlready2 ontologies
        for onto_name, ontology in self.owlready_ontologies.items():
            for cls in ontology.classes():
                curie = self._iri_to_curie(cls.name or str(cls.iri))

                # label/synonyms/definition
                labels = cls.label if hasattr(cls, 'label') else []
                name = labels[0] if labels else curie
                synonyms = []
                for attr in ('hasExactSynonym', 'hasRelatedSynonym', 'hasBroadSynonym', 'hasNarrowSynonym'):
                    if hasattr(cls, attr):
                        syns = getattr(cls, attr)
                        if isinstance(syns, list):
                            synonyms.extend([s for s in syns if isinstance(s, str)])
                definition = None
                if hasattr(cls, 'definition'):
                    definition = cls.definition[0] if cls.definition else None

                self.ontology_graph.add_node(
                    curie,
                    name=name,
                    ontology=onto_name,
                    definition=definition,
                    synonyms=synonyms,
                    iri=str(cls.iri)
                )
                self.term_to_ontology[curie] = onto_name

                # is_a
                for parent in cls.is_a:
                    # Parent could be a class OR a Restriction
                    #  A) direct named parent
                    if hasattr(parent, 'name') and parent.name:
                        parent_id = self._iri_to_curie(parent.name)
                        self.ontology_graph.add_edge(parent_id, curie, relation='is_a', source='owlready')

                    #  B) restriction: some/only/min/max etc.
                    from owlready2 import Restriction
                    if isinstance(parent, Restriction):
                        try:
                            prop = getattr(parent, 'property', None)
                            filler = getattr(parent, 'value', None)
                            if prop is not None and hasattr(prop, 'name'):
                                rel = prop.name
                            else:
                                rel = 'owl_restriction'

                            # filler could be a class, a union, etc.
                            targets = []
                            # Single class
                            if hasattr(filler, 'name'):
                                targets = [filler]
                            # Collection
                            elif hasattr(filler, '__iter__'):
                                targets = [t for t in filler if hasattr(t, 'name')]

                            for tgt in targets:
                                tgt_id = self._iri_to_curie(tgt.name or str(tgt.iri))
                                # direction: subject (cls) -> object (tgt) via property
                                self.ontology_graph.add_edge(curie, tgt_id, relation=rel, source='owlready')
                        except Exception:
                            pass

    def search_ontology_unified(self, query: str, max_results: int = 5) -> List[OntologyMatch]:
        """Search across all loaded ontologies using the unified graph"""
        matches = []
        query_lower = query.lower()

        for node_id, node_data in self.ontology_graph.nodes(data=True):
            score = 0.0
            name = node_data.get('name', '')
            synonyms = node_data.get('synonyms', [])

            # Score based on name match
            if name and name.lower() == query_lower:
                score = 1.0
            elif name and query_lower in name.lower():
                score = 0.8
            elif name and name.lower() in query_lower:
                score = 0.7

            # Score based on synonym match
            if score < 0.8:
                for synonym in synonyms:
                    if isinstance(synonym, str):
                        if synonym.lower() == query_lower:
                            score = max(score, 0.95)
                        elif query_lower in synonym.lower():
                            score = max(score, 0.75)

            if score > 0.6:
                # Get hierarchy information
                parents = list(self.ontology_graph.predecessors(node_id))
                children = list(self.ontology_graph.successors(node_id))

                # Determine biolink type
                ontology_name = node_data.get('ontology', '')
                biolink_type = self._get_biolink_type(ontology_name, node_data)

                matches.append(OntologyMatch(
                    entity_text=query,
                    ontology_id=node_id,
                    canonical_name=name or node_id,
                    biolink_type=biolink_type,
                    confidence=score,
                    synonyms=synonyms,
                    definition=node_data.get('definition'),
                    parents=parents,
                    children=children
                ))

        # Sort by confidence and return top results
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:max_results]

    def get_semantic_similarity(self, term1_id: str, term2_id: str) -> float:
        """Calculate semantic similarity using graph structure"""
        if term1_id not in self.ontology_graph or term2_id not in self.ontology_graph:
            return 0.0

        try:
            # Get ancestors for both terms
            ancestors1 = set(nx.ancestors(self.ontology_graph, term1_id))
            ancestors2 = set(nx.ancestors(self.ontology_graph, term2_id))

            # Add the terms themselves
            ancestors1.add(term1_id)
            ancestors2.add(term2_id)

            # Calculate Jaccard similarity
            intersection = len(ancestors1 & ancestors2)
            union = len(ancestors1 | ancestors2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def get_shortest_path(self, term1_id: str, term2_id: str) -> Optional[List[str]]:
        """Get shortest path between two terms in the ontology graph"""
        try:
            # Convert to undirected for path finding
            undirected = self.ontology_graph.to_undirected()
            return nx.shortest_path(undirected, term1_id, term2_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def perform_reasoning(self, ontology_name: str) -> Dict:
        """Perform logical reasoning using owlready2"""
        if ontology_name not in self.owlready_ontologies:
            return {"error": "Ontology not loaded with owlready2"}

        try:
            # Create a reasoner (HermiT, Pellet, or FaCT++)
            owl.sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)

            ontology = self.owlready_ontologies[ontology_name]

            # Get inferred information
            inferred_classes = []
            for cls in ontology.classes():
                if cls.equivalent_to:
                    inferred_classes.append({
                        'class': cls.name,
                        'equivalent_to': [eq.name for eq in cls.equivalent_to if hasattr(eq, 'name')]
                    })

            return {
                "inferred_classes": len(inferred_classes),
                "sample_inferences": inferred_classes[:5]
            }

        except Exception as e:
            return {"error": f"Reasoning failed: {str(e)}"}

    def get_term_relationships(self, term_id: str, max_depth: int = 2) -> Dict:
        """Get comprehensive relationship information for a term"""
        if term_id not in self.ontology_graph:
            return {}

        relationships = {
            'direct_parents': list(self.ontology_graph.predecessors(term_id)),
            'direct_children': list(self.ontology_graph.successors(term_id)),
            'ancestors': [],
            'descendants': []
        }

        # Get ancestors up to max_depth
        try:
            ancestors = nx.ancestors(self.ontology_graph, term_id)
            relationships['ancestors'] = list(ancestors)
        except:
            pass

        # Get descendants up to max_depth
        try:
            descendants = nx.descendants(self.ontology_graph, term_id)
            relationships['descendants'] = list(descendants)
        except:
            pass

        return relationships

    def export_subgraph(self, term_ids: List[str], output_file: str = None) -> nx.DiGraph:
        """Export a subgraph containing specified terms and their relationships"""
        # Get all related nodes
        all_nodes = set(term_ids)
        for term_id in term_ids:
            if term_id in self.ontology_graph:
                all_nodes.update(self.ontology_graph.predecessors(term_id))
                all_nodes.update(self.ontology_graph.successors(term_id))

        # Create subgraph
        subgraph = self.ontology_graph.subgraph(all_nodes).copy()

        # Save if requested
        if output_file:
            with open(output_file, 'wb') as f:
                pickle.dump(subgraph, f)

        return subgraph

    def get_ontology_statistics(self) -> Dict:
        """Get comprehensive statistics about loaded ontologies"""
        stats = {
            'pronto_ontologies': {},
            'owlready_ontologies': {},
            'unified_graph': {
                'total_terms': len(self.ontology_graph.nodes()),
                'total_relationships': len(self.ontology_graph.edges()),
                'connected_components': nx.number_weakly_connected_components(self.ontology_graph)
            }
        }

        # Pronto statistics
        for name, onto in self.pronto_ontologies.items():
            stats['pronto_ontologies'][name] = {
                'terms': len(list(onto.terms())),
                'relationships': len(list(onto.relationships()))
            }

        # Owlready2 statistics
        for name, onto in self.owlready_ontologies.items():
            classes = list(onto.classes())
            properties = list(onto.properties())
            stats['owlready_ontologies'][name] = {
                'classes': len(classes),
                'properties': len(properties),
                'individuals': len(list(onto.individuals()))
            }

        return stats

    def search_ontology(self, query: str, ontology_name: str, max_results: int = 3) -> List[OntologyMatch]:
        """Legacy method - search within a specific ontology"""
        if ontology_name in self.pronto_ontologies:
            return self._search_pronto_ontology(query, ontology_name, max_results)
        elif ontology_name in self.owlready_ontologies:
            return self._search_owlready_ontology(query, ontology_name, max_results)
        return []

    def _search_pronto_ontology(self, query: str, ontology_name: str, max_results: int) -> List[OntologyMatch]:
        """Search pronto ontology (original implementation)"""
        if ontology_name not in self.pronto_ontologies:
            return []

        ontology = self.pronto_ontologies[ontology_name]
        matches = []
        query_lower = query.lower()

        for term in ontology.terms():
            score = 0.0

            if term.name and term.name.lower() == query_lower:
                score = 1.0
            elif term.name and query_lower in term.name.lower():
                score = 0.8
            elif term.name and term.name.lower() in query_lower:
                score = 0.7

            if score < 0.8:
                for synonym in term.synonyms:
                    if synonym.description.lower() == query_lower:
                        score = max(score, 0.95)
                    elif query_lower in synonym.description.lower():
                        score = max(score, 0.75)

            if score > 0.6:
                biolink_type = self._get_biolink_type(ontology_name, {'name': term.name})

                matches.append(OntologyMatch(
                    entity_text=query,
                    ontology_id=term.id,
                    canonical_name=term.name or str(term.id),
                    biolink_type=biolink_type,
                    confidence=score,
                    synonyms=[syn.description for syn in term.synonyms],
                    definition=term.definition.strip() if term.definition else None,
                    parents=[str(parent.id) for parent in term.superclasses(with_self=False, distance=1)],
                    children=[str(child.id) for child in term.subclasses(with_self=False, distance=1)]
                ))

        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:max_results]

    def _search_owlready_ontology(self, query: str, ontology_name: str, max_results: int) -> List[OntologyMatch]:
        """Search owlready2 ontology"""
        if ontology_name not in self.owlready_ontologies:
            return []

        ontology = self.owlready_ontologies[ontology_name]
        matches = []
        query_lower = query.lower()

        for cls in ontology.classes():
            score = 0.0
            labels = cls.label if hasattr(cls, 'label') else []
            name = labels[0] if labels else (cls.name or str(cls.iri))

            # Name matching
            if name.lower() == query_lower:
                score = 1.0
            elif query_lower in name.lower():
                score = 0.8
            elif name.lower() in query_lower:
                score = 0.7

            # Synonym matching
            if score < 0.8:
                synonyms = []
                if hasattr(cls, 'hasExactSynonym'):
                    synonyms.extend(cls.hasExactSynonym)
                if hasattr(cls, 'hasRelatedSynonym'):
                    synonyms.extend(cls.hasRelatedSynonym)

                for synonym in synonyms:
                    if isinstance(synonym, str):
                        if synonym.lower() == query_lower:
                            score = max(score, 0.95)
                        elif query_lower in synonym.lower():
                            score = max(score, 0.75)

            if score > 0.6:
                # Get definition
                definition = None
                if hasattr(cls, 'definition'):
                    definition = cls.definition[0] if cls.definition else None

                # Get parents and children
                parents = [p.name or str(p.iri) for p in cls.is_a if hasattr(p, 'name')]
                children = [c.name or str(c.iri) for c in cls.subclasses() if hasattr(c, 'name')]

                biolink_type = self._get_biolink_type(ontology_name, {'name': name})

                matches.append(OntologyMatch(
                    entity_text=query,
                    ontology_id=cls.name or str(cls.iri),
                    canonical_name=name,
                    biolink_type=biolink_type,
                    confidence=score,
                    synonyms=synonyms if 'synonyms' in locals() else [],
                    definition=definition,
                    parents=parents,
                    children=children
                ))

        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:max_results]

    def validate_and_enrich_entity(self, entity_text: str) -> List[OntologyMatch]:
        """Main method for entity validation - now uses unified search"""
        return self.search_ontology_unified(entity_text, max_results=5)

    def _get_biolink_type(self, ontology_name: str, term_data: Dict) -> str:
        """Map ontology to BioLink type (enhanced for owlready2)"""
        mapping = {
            'mondo': 'Disease',
            'doid': 'Disease',
            'hp': 'PhenotypicFeature',
            'chebi': 'ChemicalSubstance',
            'go': self._classify_go_term_enhanced(term_data),
            'cl': 'Cell',
            'uberon': 'OrganismalEntity'
        }

        return mapping.get(ontology_name, 'NamedThing')

    def _classify_go_term_enhanced(self, term_data: Dict) -> str:
        """Enhanced GO term classification"""
        name = term_data.get('name', '').lower()

        # Use name-based heuristics if namespace not available
        if 'process' in name or 'pathway' in name:
            return 'BiologicalProcess'
        elif 'activity' in name or 'function' in name:
            return 'MolecularActivity'
        elif 'component' in name or 'complex' in name:
            return 'CellularComponent'

        return 'BiologicalProcess'  # Default
