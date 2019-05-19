from itertools import repeat
from multiprocessing.pool import ThreadPool
from os import path, walk, scandir

import plac
from pandas import read_csv

from cypher_engine.biolink_mapping import commit_sub_graph
from cypher_engine.connections.knowledge_graph_connection import KnowledgeGraphConnection
from cypher_engine.match import map_relations_with_ontology_terms
from preprocessor.extensions.svo import Relation
from scripts.generate_knowledge import generate_doc_details, sort_terms, get_document_subgraph
from src.bioengine import logger


def map_abstracts(pmid, abstract, doc_relations, authors, driver):
    logger.info(f'mapping terms for {pmid}')
    terms, alternate_term_dictionary, term_score = map_relations_with_ontology_terms(doc_relations)
    detail_sub_graph, entity_sub_graph, publication = generate_doc_details(pmid, abstract, authors, list(terms.keys()), terms)
    terms = sort_terms(terms, term_score)
    sub_graph, associations = get_document_subgraph(doc_relations, terms, pmid, publication)
    commit_sub_graph(driver, sub_graph, detail_sub_graph, associations)


def main(in_dir):
    driver = KnowledgeGraphConnection()
    existing_pmids = list(driver.execute_string_query('MATCH (n:Publication) RETURN n.pmid as pmid').to_data_frame()['pmid'])
    abstracts = []
    pmids = []
    author_lists = []
    relationships = []
    for pmid in scandir(in_dir):
        if pmid.name not in existing_pmids:
            pmids += [pmid.name]
            base_path = path.join(in_dir, pmid.name)
            with open(path.join(base_path, 'authors.txt')) as file:
                authors = file.readlines()
                author_lists.append([author.strip() for author in authors])
            with open(path.join(base_path, 'doc.txt')) as file:
                abstracts += [file.read()]
            csv = read_csv(path.join(base_path, 'relations.csv'))
            relationships.append([Relation(row['Subject'], row['Action'], row['Object'], row['Negation'])
                                  for index, row in csv.iterrows()])

    for pmid, abstract, relationship, author_list in zip(pmids, abstracts, relationships, author_lists):
        map_abstracts(pmid, abstract, relationship, author_list, driver)
    # pool = ThreadPool(3)
    # pool.starmap(map_abstracts, zip(pmids,
    #                                 abstracts,
    #                                 relationships,
    #                                 author_lists,
    #                                 repeat(driver)))



if __name__ == '__main__':
    plac.call(main)
