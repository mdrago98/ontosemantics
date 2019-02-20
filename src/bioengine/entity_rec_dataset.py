import csv
from random import shuffle
from src.bioengine.cypher_engine import Connection, queries
from nltk.corpus import stopwords

special_chars = ['"', "'", '{', '}', '[', ']', '/', ',', '!', '@', '%', '^', '*', '&', '(', ')']
special_chars = [(char, '0')for char in special_chars]

connection = Connection()
results = list(connection.execute_query(queries.get_go_labels_by_sub_ontology('molecular_function', limit=5000)).records())
molecular_function = [(result[0], 'PROCESS') for result in results]

results = list(connection.execute_query(queries.get_go_labels_by_sub_ontology('biological_process', limit=5000)).records())
bio_process = [(result[0], 'PROCESS') for result in results]

results = list(connection.execute_query(queries.get_go_labels_by_sub_ontology('cellular_component', limit=5000)).records())
cell_comp = [(result[0], 'GENE') for result in results]

go_terms = molecular_function + bio_process + cell_comp

results = list(connection.execute_query(queries.get_chebi_labels(5000)).records())
chebi = [(result[0], 'CHEM') for result in results]
results = list(connection.execute_query(queries.get_doid_labels(5000)).records())
disease = [(result[0], 'DISEASE') for result in results]

stop_words = [(word, '0')for word in stopwords.words('english')]

with open('/home/drago/PycharmProjects/bioengine/resources/words.txt') as f:
    english_words = f.readlines()

english_words = [x.strip() for x in english_words]

dataset = set(go_terms + chebi + disease)

english_words = filter(lambda x: x not in dataset, english_words)
english_words = [(word, '0') for word in english_words]
shuffle(english_words)
stop_words = list(filter(lambda x: x not in dataset, stop_words))
shuffle(stop_words)
# with open("/home/drago/PycharmProjects/bioengine/resources/wikigold.conll.txt") as tsv:
#     wiki_gold = [(x.strip().split(' ')[0], '0') for x in tsv]

# shuffle(wiki_gold)

dataset = set(go_terms + chebi + disease + english_words[:20000] + stop_words + special_chars)

with open('/home/drago/PycharmProjects/bioengine/stanford-ner-tagger/train/data.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for row in dataset:
        tsv_writer.writerow(row)
