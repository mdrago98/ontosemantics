from os import path, scandir

import plac
from pandas import read_csv

from knowledge_engine.connections.knowledge_graph_connection import KnowledgeGraphConnection
from nlp_processor.extensions.svo import Relation
from scripts import map_abstracts


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

    for pmid, abstract, relationship, rel_entities, author_list in zip(pmids, abstracts, relationships, entities, author_lists):
        map_abstracts(pmid, abstract, relationship, rel_entities, author_list, driver)
    # pool = ThreadPool(3)
    # pool.starmap(map_abstracts, zip(pmids,
    #                                 abstracts,
    #                                 relationships,
    #                                 author_lists,
    #                                 repeat(driver)))



if __name__ == '__main__':
    plac.call(main)
