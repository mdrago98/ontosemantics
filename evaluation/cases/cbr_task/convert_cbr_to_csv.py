import plac
from bs4 import BeautifulSoup
from pandas import DataFrame


def main(path, out='/home/drago/PycharmProjects/bioengine/resources/evaluation_data'):
    with open(path) as xml_file:
        soup = BeautifulSoup(xml_file.read(), "lxml-xml")
        documents = soup.findAll('document')
        relations = []
        for document in documents:
            doc_id = document.find('id').text
            entry_dict = {key: annotation.find('text').text
                          for passage in document.findAll('passage') for annotation in passage.findAll('annotation')
                          for key in annotation.find('infon', attrs={'key': 'MESH'}).text.split('|')}
            relationships = [(doc_id, entry_dict[relation.find('infon', attrs={'key': 'Chemical'}).text],
                                              entry_dict[relation.find('infon', attrs={'key': 'Disease'}).text])
                             for relation in document.findAll('relation')]
            relations += relationships
        DataFrame(relations, columns=['pmid', 'chemical', 'disease']).to_csv(out)


if __name__ == '__main__':
    plac.call(main)
