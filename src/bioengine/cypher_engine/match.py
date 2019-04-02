from collections import namedtuple

from cypher_engine.connections import Connection
from preprocessor.extensions.noun_verb_noun import Relation, Part
from src.bioengine.cypher_engine import OntologyStoreConnection


#TODO: add function to return a dictionary noun : ontology terms
def enrich_relation_parts(part: dict, driver: Connection) -> dict:
    """
    A method for iterating through part of relation and augmenting it with terms from an ontology
    :param driver: a neo4j driver abstraction
    :param part: a dictionary representation of a relationship part
    """
    mapping = {}
    if part:
        nouns = sum([[key] + value.nouns for key, value in part.items() if key is not None and value.nouns is not None],
                    [])
        mapping = {noun: driver.get_nodes(noun) for noun in nouns if len(noun) > 0}
    return mapping


def map_relation_with_ontology_terms(relation: namedtuple, driver: Connection = None) -> dict:
    """
    A method for mapping entities to ontology terms from relations
    :param relation: the relation extracted
    :param driver: a ne4j Driver abstraction
    """
    if driver is None:
        driver = OntologyStoreConnection()
    effector_terms = enrich_relation_parts(relation.effector, driver)
    effectee_terms = enrich_relation_parts(relation.effectee, driver)
    return {**effector_terms, **effectee_terms}


def map_relation_terms_with_ontology_terms(relations: list, driver: Connection = None) -> dict:
    """
    A method for mapping terms from relations to terms in an ontology
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

