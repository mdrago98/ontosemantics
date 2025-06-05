from py2neo import Table

from bioengine.knowledge_engine.connections import Connection
from bioengine.knowledge_engine.models import ModelFactory
from nlp_processor.entity_normalization import normalize_batch
from nlp_processor.spacy_factory import MedicalSpacyFactory
from bioengine.knowledge_engine import OntologyStoreConnection
from utils.pythonic_name import get_pythonic_name

# """UNWIND {terms} as term
#                 MATCH (node)
#                 WHERE toLower(node.label) =~ ('.*' + toLower(term) + '.*') and 'Class' in Labels(node)
#                 return node, term, LABELS(node)
#                 Union
#                 UNWIND {terms} as term
#                 match(node:Class)--(m:Class) where toLower(term) in node.synonym
#                 return node, term, LABELS(node)"""

WEIGHTS = {
    'DOID': '3',
    'CHEBI': '2',
    'RO': '-1',
    'GO': '4',
    'CL': '3',
    'FMA': '2',
    'MONDO': '-1'
}


def get_mapping_query() -> str:
    """
    A wrapper to the term matching query
    :return: the string representing the parameterized query
    """
    return """UNWIND {terms} as term
                MATCH (node:Class)-[r]-(m:Class)
                WHERE toLower(node.label) = toLower(term)
                RETURN node, term, LABELS(node), COUNT(r)
                UNION
                UNWIND {terms} as term
                match(node:Class)-[r]-(m:Class) where toLower(term) in [list_elem IN node.synonym | toLower(list_elem)]
                return node, term, LABELS(node), COUNT(r)"""


# [list_elem IN n.synonym | toLower(list_elem)]
def get_ontology_mapping(terms: list, driver: Connection) -> Table:
    """
    A helper function for executing a cypher to obtain the term mappings
    :param terms: the terms to map
    :param driver: a neo4j connection instance
    :return: a table representing the mapping results
    """
    str_terms = [str(term) for term in terms]
    return driver.execute_string_query(get_mapping_query(), terms=str_terms).to_table()


def map_relations_with_ontology_terms(relations: list, entities: list = None,
                                      driver: Connection = None, nlp=None, weights= None) -> tuple:
    """
    A method for mapping entities to ontology terms from relations
    :param pmid: of the  document from which relations where extracted
    :param nlp: a spacy instance
    :param entities: a list of named entities to query
    :param driver: a neo4j Driver abstraction
    """
    if weights is None:
        weights = WEIGHTS
    if entities is None:
        entities = []
    if driver is None:
        driver = OntologyStoreConnection()
    if nlp is None:
        nlp = MedicalSpacyFactory.factory(enable_ner=False, enable_benepar=False)
    terms = sum([[relation.effector, relation.effectee] for relation in relations], [])
    terms += entities
    terms = list(set(terms))
    alternate_term_dict, alternate_terms = normalize_batch(terms, nlp)
    terms += alternate_terms
    relation_terms = get_ontology_mapping(terms, driver)
    term_ranking: dict = {}
    mapping: dict = {}
    for row in relation_terms:
        model_identifier = [classifier for classifier in row[2] if classifier not in ('Class', '_Class', 'Obsolete')]
        graph_object = ModelFactory.factory(model_identifier[0])
        for key, item in row[0].items():
            key = get_pythonic_name(key)
            setattr(graph_object, get_pythonic_name(key), item)
        document_repr = list(dict(filter(lambda value: row[1] in value[1], alternate_term_dict.items())).keys())[0]
        score = int(weights[graph_object.ontology_prefix]) * int(row[3]) if \
            graph_object.ontology_prefix in weights else int(row[3])
        term_ranking.update({graph_object.iri: score})
        if document_repr in mapping:
            mapping[document_repr] += [graph_object]
        else:
            mapping[document_repr] = [graph_object]
    return mapping, alternate_term_dict, term_ranking
