import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.bioengine.dochandlers.page_objects.wikipedia_page_object import WikipediaPageObject

corpus = WikipediaPageObject('Diabetes_mellitus').get_text()
corpus = ' '.join(corpus)
sentences = nltk.sent_tokenize(corpus)

jar = '/home/drago/PycharmProjects/bioengine/stanford-ner-tagger/stanford-ner.jar'
model = '/home/drago/PycharmProjects/bioengine/stanford-ner-tagger/bio-ner-model.ser.gz'

# Prepare NER tagger with english model
ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')
stop_words = set(stopwords.words('english'))


# to do investigate stemming
def tag_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    words = set(words) - stop_words
    return ner_tagger.tag(words)


# Run NER tagger on words
tags = [tag_sentence(sentence) for sentence in sentences]

print(tags)

with open('test.txt', 'w') as file:
    for tag in tags:
        for tup in tag:
            to_write = ' '.join(tup)
        to_write += '\n'
        file.write(to_write)


