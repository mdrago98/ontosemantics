from nltk import sent_tokenize

from src.bioengine.cypher_engine import OntologyStoreConnection
from src.bioengine.dochandlers.page_objects.pmc_page_object import PMCPageObject
from src.bioengine.dochandlers.txt_file_handler import write_to_file
from src.bioengine.preprocessor import TaggerFactory

import re

graph = OntologyStoreConnection()
corpus = PMCPageObject('PMC5894887').get_text()
corpus = [line.strip('\n') for line in corpus]
corpus = [re.sub(r'[^\x00-\x7F]+', ' ', line) for line in corpus]
corpus = "e.g. Red algae: Aqueous extracts of Gracilaria corticata and Sargassum oligocystum inhibited the " \
         "proliferation of human leukemic cell lines. Both ethanol and methanol extracts of Gracilaria tenuistipitata " \
         "reportedly had anti-proliferative effects on Ca9-22 oral cancer cells and were involved in cellular " \
         "apoptosis, DNA damage, and oxidative stress. [example source: PMC3674937] "
sentences = sent_tokenize(corpus)

# model = Model.load_external('medacy_model_clinical_notes')
# annotation = model.predict(corpus)
#
# tagger = TaggerFactory.factory(sentences=sentences, tagger_type='stanford')
# entities = tagger.tag_sentences()
# print(entities)
with TaggerFactory.factory(sentences=sentences) as tagger:
    entities = tagger.tag_sentences()
    terms = []


# tagger = TaggerFactory.factory(sentences, tagger_type='genia')
# gen_entities = tagger.tag_sentences()
# print(gen_entities)