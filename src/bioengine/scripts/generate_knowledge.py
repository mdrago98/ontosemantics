from py2neo import Node, Subgraph

from cypher_engine.biolink_mapping import get_relationship_node, get_association, get_publication_node, get_providers, \
    link_publication_to_provider, link_entities_to_publication
from cypher_engine.connections.knowledge_graph_connection import KnowledgeGraphConnection
from cypher_engine.match import map_relations_with_ontology_terms
from biolinkmodel.datamodel import NamedThing
from preprocessor.extensions.svo import Relation
from pandas import read_csv
from os.path import join

# base_dir = '/home/drago/test_eval/'
# pmid = '26508947'
# driver = KnowledgeGraphConnection()
# csv = read_csv(join(base_dir, pmid, 'relations.csv'))
# with open(join(base_dir, pmid, 'entities.txt')) as file:
#     entities = file.readlines()
# with open(join(base_dir, pmid, 'doc.txt')) as file:
#     doc = file.read()
# with open(join(base_dir, pmid, 'authors.txt')) as file:
#     authors = file.readlines()
# entities = [entity.strip() for entity in entities]
# authors = [author.strip() for author in authors]
# relations = [Relation(row['Subject'], row['Action'], row['Object'], row['Negation']) for index, row in csv.iterrows()]
# terms, alternate_term_dictionary, term_score = map_relations_with_ontology_terms(relations, entities=entities)


def generate_doc_details(pmid: str, doc: str, authors: list, entities, terms) -> tuple:
    """
    A function that generates a subgraph of document details
    :param terms: A term lookup dictionary
    :param doc: the document text
    :param pmid: the pmid of the article
    :param authors: a list of authors
    :param entities: a list of entities
    """
    sub_graph = None
    entity_sub_graph = None
    for entity in entities:
        node, entity_relations = get_relationship_node(entity, terms)
        if node is not None:
            entity_sub_graph = node if sub_graph is None else sub_graph | node
        if entity_relations is not None:
            entity_sub_graph |= entity_relations
    publication: Node = get_publication_node(doc, pmid, pmid)
    if entities is not None and len(entities) > 0:
        entity_sub_graph |= link_entities_to_publication(list(entity_sub_graph.nodes), publication, entity_sub_graph)
    sub_graph = publication
    if authors is not None and len(authors) > 0:
        providers: list = get_providers(authors)
        sub_graph |= link_publication_to_provider(providers, publication, sub_graph)
    return sub_graph, entity_sub_graph, publication


def sort_terms(term_dict: dict, term_scores: dict) -> dict:
    """
    A function that sorts entries by score
    :param term_dict: the term dictionary
    :param term_scores: a dictionary of iri: scores
    :return: the term dict with sorted values
    """
    for keys, values in term_dict.items():
        values.sort(key=lambda graph_obj: term_scores[graph_obj.iri], reverse=True)
    return term_dict


def get_document_subgraph(doc_relations: list, terms: dict, pmid: str, publication: Node) -> tuple:
    """
    A function for generating th subgraph of relations for a given document
    :param doc_relations: the document subject verb object relations
    :param terms: the term dictionary
    :param pmid: the pmid
    :param publication:
    :return: a tuple representing the subgraph and the association
    """
    associations = []
    sub_graph = None
    for relation in doc_relations:
        if relation.effector in terms and relation.effectee in terms:
            subject_node, subject_relations = get_relationship_node(relation.effector, terms)
            object_node, object_relations = get_relationship_node(relation.effectee, terms)
            associations += [get_association(relation.relation, subject_node, object_node, relation.negation,
                                             pmid=pmid)]
            if sub_graph is None:
                sub_graph = subject_node | object_node
            else:
                sub_graph |= subject_node | object_node
            sub_graph |= link_entities_to_publication(list(sub_graph.nodes), publication, sub_graph)
            if subject_relations is not None:
                sub_graph |= subject_relations
            if object_relations is not None:
                sub_graph |= object_relations
    return sub_graph, associations

# terms = sort_terms(terms, term_score)
#
# detail_sub_graph, entity_sub_graph, publication = generate_doc_details(pmid, doc, authors, entities, terms)
#
# sub_graph = None
# associations = []
# for relation in relations:
#     if relation.effector in terms and relation.effectee in terms:
#         subject_node, subject_relations = get_relationship_node(relation.effector, terms)
#         object_node, object_relations = get_relationship_node(relation.effectee, terms)
#         associations += [get_association(relation.relation, subject_node, object_node, relation.negation)]
#         if sub_graph is None:
#             sub_graph = subject_node | object_node
#         else:
#             sub_graph |= subject_node | object_node
#         sub_graph |= link_entities_to_publication(list(sub_graph.nodes), publication, sub_graph)
#         if subject_relations is not None:
#             sub_graph |= subject_relations
#         if object_relations is not None:
#             sub_graph |= object_relations
#
# sub_graph |= entity_sub_graph
#
# tx = driver.driver.begin()
# tx.merge(sub_graph, primary_label=NamedThing.__class__.__name__, primary_key='iri')
# tx.merge(detail_sub_graph, primary_label=NamedThing.__class__.__name__, primary_key='name')
# for association in associations:
#     tx.merge(association, primary_label=association.__class__.__name__, primary_key='relation')
# tx.commit()
