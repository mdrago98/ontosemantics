import logging
from itertools import repeat
from multiprocessing.pool import ThreadPool

from biolinkmodel.datamodel import NamedThing

from bioengine.knowledge_engine.connections import Connection
from bioengine.knowledge_engine.connections.knowledge_graph_connection import KnowledgeGraphConnection
from bioengine.knowledge_engine.match import map_relations_with_ontology_terms
from bioengine.dochandlers.page_objects.pmc_page_object import PMCPageObject
from bioengine.nlp_processor.extensions.svo import Relation
from bioengine.nlp_processor.spacy_factory import MedicalSpacyFactory
from bioengine.scripts import sort_terms, generate_doc_details, get_document_subgraph

logger = logging.getLogger(__name__)


def main(nlp: MedicalSpacyFactory, driver: Connection, pmc_pg):
    text = ' '.join(pmc_pg.text)
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    name, authors = pmc_pg.get_article_details()
    logger.info(f'started generating knowledge graph for {name}')
    doc = nlp(text)
    ents = [str(entity) for entity in doc.ents]
    abstract = pmc_pg.get_abstract()
    doc_relations = doc._.noun_verb_chunks
    doc_relations = [
        Relation(str(relation.effector), str(relation.relation), str(relation.effectee[1]), relation.negation)
        for relation in doc_relations]
    logger.info(f'mapping terms for {name}')
    terms, alternate_term_dictionary, term_score = map_relations_with_ontology_terms(doc_relations)
    detail_sub_graph, entity_sub_graph, publication = generate_doc_details(pmc_pg.id, abstract, authors, ents, terms)
    terms = sort_terms(terms, term_score)
    sub_graph, associations = get_document_subgraph(doc_relations, terms, pmc_pg.id, publication)
    logger.info(f'generating sub graph for {name}')
    tx = driver().driver.begin()
    tx.merge(sub_graph, primary_label=NamedThing.__class__.__name__, primary_key='iri')
    tx.merge(detail_sub_graph, primary_label=NamedThing.__class__.__name__, primary_key='name')
    for association in associations:
        tx.merge(association, primary_label=association.__class__.__name__, primary_key='relation')
    tx.commit()


# [PMCPageObject('28974775', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5894887/'),
#                 OUPPageObject('11238471', 'https://academic.oup.com/jcem/article/86/3/972/2847394'),
#                 SpringerPageObject('14722654', 'https://link.springer.com/article/10.1007%2Fs00125-003-1313-3'),
#                 NaturePageObject('11742412', 'https://www.nature.com/articles/414799a')]
# wilson's disease:
# PMCPageObject('26692151', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4678372/')
if __name__ == '__main__':
    nlp = MedicalSpacyFactory.factory()
    driver = KnowledgeGraphConnection()
    # pmc_list = [WileyPageObject('1', '/home/drago/Downloads', True),
    #     BioScientificaPageObject('21498522', 'https://jme.bioscientifica.com/view/journals/jme/47/1/R1.xml')]
    pmc_list = [
                PMCPageObject('26692151', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4678372/')
                ]
    pool = ThreadPool(10)
    pool.starmap(main, zip(repeat(nlp), repeat(KnowledgeGraphConnection), pmc_list))

# handle = Entrez.efetch(db="pubmed", id=','.join(map(str, pm_list)),
#                        rettype="xml", retmode="text")
# papers = Entrez.read(handle)
# abstracts = {
#     pubmed_article['MedlineCitation']['PMID']: pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][
#         0]
#     for pubmed_article in papers['PubmedArticle']
#     if 'Abstract' in pubmed_article['MedlineCitation']['Article']}
# documents = [doc for doc in nlp.pipe([str(abstract) for _, abstract in abstracts.items()], batch_size=16,
#                                      n_threads=10)]
#
# sents = sum([list(document.sents) for document in documents], [])
# with open('sentence.tsv', 'w') as file:
#     file.write('sent_number \t sentence\n')
#     for sent_index in range(len(sents)):
#         file.write(f'{sent_index} \t {sents[sent_index]}\n')

# doc = nlp('The PON1 102V allele appears to be associated with an increased risk for prostate cancer.')
# print(doc._.noun_verb_chunks)
