from Bio import Entrez
from pandas import DataFrame

from spacy_factory import MedicalSpacyFactory


def search(query, retmax: int = 20):
    Entrez.email = 'matthew.drago.16@um.edu.mt'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax=retmax,
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    return results


nlp = MedicalSpacyFactory.factory()

# pm_list = ['29659364', '26508947', '27239592', '26913870', '30071825', '29675260', '29463074', '25083957', '30067922', '27156762', '28476225', '28420857', '21842608', '28258576', '28920918', '28626085', '28919622', '28483362', '26181158', '17026722', '29621695', '29353447', '28586735', '27788409', '29597131', '29957557', '28505557', '29974352', '29902718', '30099326', '29651637', '29388117', '30022435', '30039339', '28719811', '29763858', '29156384', '29614476', '27598996', '26255741']
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

doc = nlp('The PON1 102V allele appears to be associated with an increased risk for prostate cancer.')
print(doc._.noun_verb_chunks)
