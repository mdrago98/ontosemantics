import plac
from Bio import Entrez

from src.bioengine.spacy_factory import MedicalSpacyFactory
from os import path, makedirs


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
    Entrez.email = 'your.email@example.com'
    handle = Entrez.efetch(db='pubmed',
                           rettype='full',
                           retmode='xml',
                           id=ids)
    return Entrez.read(handle)


def read_and_parse(query: str, size: int, batch_size: int = 100, threads: int = 12):
    # id_list = search(query, size)['IdList']
    id_list = ['30071825']
    handle = Entrez.efetch(db="pubmed", id=','.join(map(str, id_list)),
                           rettype="xml", retmode="text")
    papers = Entrez.read(handle)
    abstracts = [pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                 for pubmed_article in papers['PubmedArticle']
                 if 'Abstract' in pubmed_article['MedlineCitation']['Article']]
    nlp = MedicalSpacyFactory.factory()
    documents = [doc for doc in nlp.pipe([str(abstract) for abstract in abstracts], batch_size=batch_size,
                                     n_threads=threads)]
    return {str(pubmed_id): documents[index] for index, pubmed_id in enumerate(id_list)}


def main(directory='', query='diabetes', size=5):
    """
    Main method obtains abstracts and parses them and organises them according to their pmid
    :param directory: the root directory where to save the output
    :param query: the query term
    :param size: the amount of pubmed articles to get
    """
    res = read_and_parse(query, size)
    for pubmed_id, doc in res.items():
        out_dir = path.join(directory, pubmed_id)
        if not path.exists(out_dir):
            makedirs(out_dir)
        relations = sum(doc._.noun_verb_chunks, [])
        with open(path.join(out_dir, 'relations.txt'), 'w') as file:
            for relation in relations:
                file.write(f'\nEffector: {relation[0]}, Verb: {relation[1]}, Efectee: {relation[2]}')
        with open(path.join(out_dir, 'doc.txt'), 'w') as file:
            file.write(doc.text)


if __name__ == '__main__':
    plac.call(main)
