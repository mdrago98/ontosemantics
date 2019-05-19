from itertools import repeat
from multiprocessing.pool import ThreadPool
from os import path, makedirs

import plac
from biolinkmodel.datamodel import NamedThing
from pandas import read_csv, DataFrame

from cypher_engine.biolink_mapping import commit_sub_graph
from cypher_engine.connections.knowledge_graph_connection import KnowledgeGraphConnection
from cypher_engine.match import map_relations_with_ontology_terms
from preprocessor.extensions.svo import Relation
from preprocessor.spacy_factory import MedicalSpacyFactory
from scripts.generate_knowledge import generate_doc_details, get_document_subgraph, sort_terms
from scripts.get_nouns import read_and_parse
from src.bioengine import logger


def map_abstracts(pmid, doc, authors, driver):
    abstract = doc.text
    doc_relations = doc._.noun_verb_chunks
    doc_relations = [
        Relation(str(relation.effector), str(relation.relation), str(relation.effectee[1]), relation.negation)
        for relation in doc_relations]
    logger.info(f'mapping terms for {pmid}')
    terms, alternate_term_dictionary, term_score = map_relations_with_ontology_terms(doc_relations)
    detail_sub_graph, entity_sub_graph, publication = generate_doc_details(pmid, abstract, authors, [], terms)
    terms = sort_terms(terms, term_score)
    sub_graph, associations = get_document_subgraph(doc_relations, terms, pmid, publication)
    commit_sub_graph(driver, sub_graph, detail_sub_graph, associations)


def main(cdr_rel, out):
    frame = read_csv(cdr_rel)
    ids = list(set(frame['pmid']))
    nlp = MedicalSpacyFactory.factory()
    documents, authors = read_and_parse(id_list=ids, nlp=nlp)
    for pubmed_id, doc in documents.items():
        out_dir = path.join(out, pubmed_id)
        if not path.exists(out_dir):
            makedirs(out_dir)
        doc_relations = doc._.noun_verb_chunks
        doc_relations = [
                    Relation(str(relation.effector), str(relation.relation), str(relation.effectee[1]), relation.negation)
                    for relation in doc_relations]
        frame = DataFrame(doc_relations, columns=['Subject', 'Action', 'Object', 'Negation'])
        frame.to_csv(path.join(out_dir, 'relations.csv'))

        with open(path.join(out_dir, 'authors.txt'), 'w') as file:
            for author in authors[pubmed_id]:
                file.write(f'{author}\n')
        with open(path.join(out_dir, 'doc.txt'), 'w') as file:
            file.write(doc.text)
    # pool = ThreadPool(8)
    # pool.starmap(map_abstracts, zip([doc[0] for doc in list(documents.items())],
    #                                 [doc[1] for doc in list(documents.items())],
    #                                 [author[1] for author in list(authors.items())],
    #                                 repeat(driver)))


if __name__ == '__main__':
    plac.call(main)
