from collections import namedtuple

from pandas import DataFrame

from cypher_engine.connections import Connection
from src.bioengine.cypher_engine import OntologyStoreConnection


def filter_relation_terms(relation_part: list) -> list:
    return list(filter(lambda x: not isinstance(x, list), relation_part))


def get_mapping_query() -> str:
    """
    A wrapper to the term matching query
    :return: the string representing the parameterized query
    """
    return """UNWIND {terms} as term
                MATCH (n:Class)--(m:Class)
                WHERE n.label =~ term
                RETURN n, term, LABELS(n)"""


def get_ontology_mapping(terms: list, driver: Connection) -> DataFrame:
    return driver.execute_string_query(get_mapping_query(), terms=terms).to_data_frame()


def map_relation_with_ontology_terms(relation: namedtuple, driver: Connection = None) -> dict:
    """
    A method for mapping entities to ontology terms from relations
    :param relation: the relation extracted
    :param driver: a ne4j Driver abstraction
    """
    if driver is None:
        driver = OntologyStoreConnection()
    effector_terms = {relation.effector: driver.get_nodes(relation.effector)}
    objects = filter_relation_terms(relation.effectee)
    effectee_terms = {obj: driver.get_nodes(obj) for obj in objects}
    combined_terms = {**effector_terms, **effectee_terms}
    return {key: value for key, value in combined_terms.items() if len(value) > 0}


def map_relation_terms_with_ontology_terms(relations: list, driver: Connection = None) -> dict:
    """
    A method for mapping terms from relations to terms in an ontology
    :param driver:
    :param relations: a list of relations
    """
    if driver is None:
        driver = OntologyStoreConnection()
    terms = {}
    for relation in relations:
        rel_terms = map_relation_with_ontology_terms(relation, driver)
        terms = {**terms, **rel_terms}
    return terms

# test_relation = ({'Central diabetes insipidus': ([], [])}, 'is',
#                  {'rare disease': ([], ['hypothalamus', 'and', 'neurohypophysis'])})
# map_relation_with_ontology_terms(test_relation)

