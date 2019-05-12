import plac
from Bio import Entrez

from preprocessor.extensions.svo import Relation
from src.bioengine import logger
from preprocessor.spacy_factory import MedicalSpacyFactory
from os import path, makedirs
from pandas import DataFrame


def search(query, retmax: int = 20):
    Entrez.email = 'matthew.drago.16@um.edu.mt'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax=retmax,
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    return results


def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'matthew.drago.16@um.edu.mt'
    handle = Entrez.efetch(db='pubmed',
                           rettype='full',
                           retmode='xml',
                           id=ids)
    return Entrez.read(handle)


def read_and_parse(query: str, size: int, nlp=None, batch_size: int = 100, threads: int = 12) -> tuple:
    """
    A method that reads and parses entries from entrez
    :param query: the query string
    :param size: the size of the
    :param nlp: a spacy abstraction of the model
    :param batch_size: the batch size
    :param threads: the number of max threads to occupy
    :return: a tuple containing a dictionary of parsed documents and a dictionary of pmids and authors
    """
    if nlp is None:
        nlp = MedicalSpacyFactory.factory()

    id_list = search(query, size)['IdList']
    handle = Entrez.efetch(db="pubmed", id=','.join(map(str, id_list)),
                           rettype="xml", retmode="text")
    papers = Entrez.read(handle)
    abstracts = {pubmed_article['MedlineCitation']['PMID']: pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                 for pubmed_article in papers['PubmedArticle']
                 if 'Abstract' in pubmed_article['MedlineCitation']['Article']}
    authors = {str(pubmed_article['MedlineCitation']['PMID']): [f'{author["LastName"]}, {author["ForeName"]}'
                                                           for author in pubmed_article['MedlineCitation']['Article']['AuthorList']]
               for pubmed_article in papers['PubmedArticle']}
    documents = [doc for doc in nlp.pipe([str(abstract) for _, abstract in abstracts.items()], batch_size=batch_size,
                                         n_threads=threads)]
    return {str(pubmed_id): documents[index] for index, pubmed_id in enumerate(abstracts.keys())}, authors


def main(directory='', query='diabetes', size=4):
    """
    Main method obtains abstracts and parses them and organises them according to their pmid
    :param directory: the root directory where to save the output
    :param query: the query term
    :param size: the amount of pubmed articles to get
    """

    documents, authors = read_and_parse(query, size)
    relations = {}
    pmid_errors = []
    for pubmed_id, doc in documents.items():
        out_dir = path.join(directory, pubmed_id)
        if not path.exists(out_dir):
            makedirs(out_dir)
        doc_relations = doc._.noun_verb_chunks
        doc_relations = [Relation(relation.effector, relation.relation, relation.effectee[1], relation.negation)
                         for relation in doc_relations if str(relation.effector) in str(doc.ents)
                         and str(relation.effectee[1]) in str(doc.ents)]
        frame = DataFrame(doc_relations, columns=['Subject', 'Action', 'Object', 'Negation'])
        relations[pubmed_id] = frame
        frame.to_csv(path.join(out_dir, 'relations.csv'))
        with open(path.join(out_dir, 'entities.txt'), 'w') as file:
            for entity in doc.ents:
                file.write(f'{entity}\n')
        with open(path.join(out_dir, 'authors.txt'), 'w') as file:
            for author in authors[pubmed_id]:
                file.write(f'{author}\n')
        with open(path.join(out_dir, 'doc.txt'), 'w') as file:
            file.write(doc.text)
    if len(pmid_errors) > 0:
        logger.debug(f'Failed PMIDS: {pmid_errors}')


if __name__ == '__main__':
    plac.call(main)
