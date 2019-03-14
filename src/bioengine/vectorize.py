import spacy

import cupy.cuda

# Load English tokenizer, tagger, parser, NER and word vectors
from src.bioengine.dochandlers.page_objects.wikipedia_page_object import WikipediaPageObject
spacy.prefer_gpu()

print("loading model")

# nlp = spacy.load('/home/drago/PycharmProjects/bioengine/resources/vectors/Pubmed-vec')
nlp = spacy.load('/home/drago/PycharmProjects/bioengine/resources/med_model/model0')


test = WikipediaPageObject('Diabetes_mellitus')
corpus = ' '.join(test.get_text())
doc = nlp(corpus)

similar_tokens = [token for token in doc if token.similarity(nlp('diabetes')) > 0.3]

print(set(similar_tokens))


# engine = PipelineEngine(pipes_structure, Context(doc), [0, 1, 2])
# print(engine.process())
# for token in doc:
#     print(token.text, token.has_vector, token.vector_norm, token.is_oov)

# print(list(doc.noun_chunks))

# visualize_embeddings(wv=doc.vector, vocabulary=nlp.vocab.strings)

# for key in nlp.vocab.vectors.keys():
#     print(key, nlp.vocab.strings[key])
# text = "One important cytoplasmic component of the cell-ECM adhe-sions, integrin-linked kinase (ILK), was identified approx-imately 8 years ago in a yeast two-hybrid screen based on its interaction with the Bl integrin cytoplasmic domain [12]."
# doc = nlp(text)
# engine = PipelineEngine(pipes_structure, Context(doc), [0, 1, 2])
# print(engine.process())
# print(list(doc.noun_chunks))
