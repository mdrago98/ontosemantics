from itertools import repeat
from multiprocessing.pool import ThreadPool
from os import scandir, path

import plac
from py2neo import Node

from knowledge_engine.biolink_mapping import get_relationship_node, get_association, get_publication_node, get_providers, \
    link_publication_to_provider, link_entities_to_publication, commit_sub_graph
from bioengine.knowledge_engine.connections import Connection
from bioengine.knowledge_engine.connections.knowledge_graph_connection import KnowledgeGraphConnection
from bioengine.knowledge_engine.match import map_relations_with_ontology_terms
from biolinkmodel.datamodel import NamedThing
from bioengine.nlp_processor.extensions.svo import Relation
from pandas import read_csv
from bioengine import logger


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
    if entities is not None and len(entities) > 0 and entity_sub_graph is not None:
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


def gen_graph(doc, name, authors, pmid, entities, driver: Connection):
    abstract = str(doc)
    doc_relations = doc._.noun_verb_chunks
    doc_relations = [
        Relation(str(relation.effector), str(relation.relation), str(relation.effectee[1]), relation.negation)
        for relation in doc_relations]
    logger.info(f'mapping terms for {name}')
    terms, alternate_term_dictionary, term_score = map_relations_with_ontology_terms(doc_relations)
    detail_sub_graph, entity_sub_graph, publication = generate_doc_details(pmid, abstract, authors, entities, terms)
    terms = sort_terms(terms, term_score)
    sub_graph, associations = get_document_subgraph(doc_relations, terms, pmid, publication)
    logger.info(f'generating sub graph for {name}')
    tx = driver().driver.begin()
    tx.merge(sub_graph, primary_label=NamedThing.__class__.__name__, primary_key='iri')
    tx.merge(detail_sub_graph, primary_label=NamedThing.__class__.__name__, primary_key='name')
    for association in associations:
        tx.merge(association, primary_label=association.__class__.__name__, primary_key='relation')
    tx.commit()


def main(in_dir):
    driver = KnowledgeGraphConnection()
    existing_pmids = list(driver.execute_string_query('MATCH (n:Publication) RETURN n.pmid as pmid').to_data_frame()['pmid'])
    # existing_pmids = []
    abstracts = []
    pmids = []
    author_lists = []
    relationships = []
    entities = []
    for pmid in scandir(in_dir):
        if pmid.name not in existing_pmids:
            pmids += [pmid.name]
            base_path = path.join(in_dir, pmid.name)
            with open(path.join(base_path, 'entities.txt')) as file:
                doc_entities = file.readlines()
                doc_entities = [entity.strip() for entity in doc_entities]
                entities.append(doc_entities)
            with open(path.join(base_path, 'authors.txt')) as file:
                authors = file.readlines()
                author_lists.append([author.strip() for author in authors])
            with open(path.join(base_path, 'doc.txt')) as file:
                abstracts += [file.read()]
            csv = read_csv(path.join(base_path, 'relations.csv'))
            relationships.append([Relation(row['Subject'], row['Action'], row['Object'], row['Negation'])
                                  for index, row in csv.iterrows()])

    # for pmid, abstract, relationship, rel_entities, author_list in zip(pmids, abstracts, relationships, entities, author_lists):
    #     map_abstracts(pmid, abstract, relationship, rel_entities, author_list, driver)
    pool = ThreadPool(3)
    pool.starmap(map_abstracts, zip(pmids,
                                    abstracts,
                                    relationships,
                                    entities,
                                    author_lists,
                                    repeat(driver)))


def map_abstracts(pmid, abstract, doc_relations, entities, authors, driver):
    logger.info(f'mapping terms for {pmid}')
    terms, alternate_term_dictionary, term_score = map_relations_with_ontology_terms(doc_relations, entities=entities)
    detail_sub_graph, entity_sub_graph, publication = generate_doc_details(pmid, abstract, authors, entities, terms)
    terms = sort_terms(terms, term_score)
    sub_graph, associations = get_document_subgraph(doc_relations, terms, pmid, publication)
    commit_sub_graph(driver, sub_graph, detail_sub_graph, associations, entity_sub_graph)


if __name__ == '__main__':
    plac.call(main)
