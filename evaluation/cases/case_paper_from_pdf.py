from collections import namedtuple
from itertools import repeat
from multiprocessing.pool import ThreadPool

from build.lib.knowledge.datamodel import NamedThing
from cypher_engine.connections import Connection
from cypher_engine.connections.knowledge_graph_connection import KnowledgeGraphConnection
from cypher_engine.match import map_relations_with_ontology_terms
from dochandlers.txt_file_handler import read_from_file
from nlp_processor.extensions.svo import Relation
from nlp_processor.spacy_factory import MedicalSpacyFactory
from scripts.generate_knowledge import generate_doc_details, sort_terms, get_document_subgraph
from src.bioengine import logger
from utils.citation_utils import strip_citations


def main(nlp: MedicalSpacyFactory, driver: Connection, pmc_pg):
    # text = ' '.join(pmc_pg.text)
    text = pmc_pg.text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = strip_citations(text)
    name = pmc_pg.name
    authors = pmc_pg.authors
    logger.info(f'started generating knowledge graph for {name}')
    doc = nlp(text)
    abstract = ' '.join(list(sent.text for sent in list(doc.sents)[0:6]))
    doc_relations = doc._.noun_verb_chunks
    doc_relations = [
        Relation(str(relation.effector), str(relation.relation), str(relation.effectee[1]), relation.negation)
        for relation in doc_relations]
    logger.info(f'mapping terms for {name}')
    terms, alternate_term_dictionary, term_score = map_relations_with_ontology_terms(doc_relations)
    detail_sub_graph, entity_sub_graph, publication = generate_doc_details(pmc_pg.id, abstract, authors, [], terms)
    terms = sort_terms(terms, term_score)
    sub_graph, associations = get_document_subgraph(doc_relations, terms, pmc_pg.id, publication)
    logger.info(f'generating sub graph for {name}')
    tx = driver().driver.begin()
    tx.merge(sub_graph, primary_label=NamedThing.__class__.__name__, primary_key='iri')
    tx.merge(detail_sub_graph, primary_label=NamedThing.__class__.__name__, primary_key='name')
    for association in associations:
        tx.merge(association, primary_label=association.__class__.__name__, primary_key='relation')
    tx.commit()


if __name__ == '__main__':
    nlp = MedicalSpacyFactory.factory()
    driver = KnowledgeGraphConnection()
    page_object = namedtuple('PageObject', 'name id authors text')
    pmc_list = [
                page_object('Signalling through the insulin receptor', '4', ['Jonathan P Whitehead', 'Sharon F Clark'], read_from_file('/home/drago/PycharmProjects/bioengine/resources/extracted_text/whitehead2000.txt'))
                ]
    pool = ThreadPool(8)
    pool.starmap(main, zip(repeat(nlp), repeat(KnowledgeGraphConnection), pmc_list))
