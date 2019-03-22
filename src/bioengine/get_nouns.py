
from Bio import Entrez, Medline

from src.bioengine.spacy_factory import MedicalSpacyFactory


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
    results = Entrez.read(handle)
    return results


if __name__ == '__main__':

    results = search('diabetes')
    id_list = search('diabetes', 5)['IdList']
    handle = Entrez.efetch(db="pubmed", id=','.join(map(str, id_list)),
                           rettype="xml", retmode="text")
    papers = Entrez.read(handle)
    abstracts = [pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                 for pubmed_article in papers['PubmedArticle']
                 if 'Abstract' in pubmed_article['MedlineCitation']['Article']]
    abstract_dict = dict(zip(id_list, abstracts))
    nlp = MedicalSpacyFactory.factory()
    res = [doc for doc in nlp.pipe([str(abstract) for abstract in abstracts], batch_size=100, n_threads=16)]

    print(res)





