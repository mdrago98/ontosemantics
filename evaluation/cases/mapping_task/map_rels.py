from os import scandir, path
from random import choices

import plac
from pandas import read_csv

from bioengine.knowledge_engine.connections.knowledge_graph_connection import KnowledgeGraphConnection
from bioengine.scripts import map_abstracts
from bioengine.nlp_processor.extensions.svo import Relation
from src.bioengine import logger


def main(in_dir, svo_path, output_loc=''):
    driver = KnowledgeGraphConnection()
    existing_pmids = list(driver.execute_string_query('MATCH (n:Publication) RETURN n.pmid as pmid').to_data_frame()['pmid'])
    abstracts = []
    pmids = choices([pmid.name for pmid in scandir(in_dir) if pmid not in existing_pmids], k=40)
    # pmids = ['8312983', '24114426', '24190587', '25006961', '11745287', '3125850', '3191389', '11105626', '7967231',
    #          '9071336', '15276093', '16083708', '8701013', '2614930', '23846525', '20431083', '17828434', '7248895',
    #          '1756784', '10986547']
    author_lists = []
    relationships = []
    extracted_rels = read_csv(svo_path)
    for pmid in pmids:
        base_path = path.join(in_dir, pmid)
        with open(path.join(base_path, 'authors.txt')) as file:
            authors = file.readlines()
            author_lists.append([author.strip() for author in authors])
        with open(path.join(base_path, 'doc.txt')) as file:
            abstracts += [file.read()]
        filtered_task_df = extracted_rels[extracted_rels.pmid == int(pmid)]
        relationships.append([Relation(row['chemical'], '', row['disease'], False)
                              for index, row in filtered_task_df.iterrows()])

    for pmid, abstract, relationship, author_list in zip(pmids, abstracts, relationships, author_lists):
        logger.info(f'Mapping {pmid}')
        map_abstracts(pmid, abstract, relationship, [], author_list, driver)
    logger.info(f'Mapped {pmids}')


if __name__ == '__main__':
    plac.call(main)
