import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import stopwords as nltk_stop_words

from bioengine.nlp_processor.taggers.tagger import Tagger


class StanfordTagger(Tagger):
    """
    A class that abstracts the stanford NER tagger.
    """

    def __init__(self, sentences: list, config: dict, stopwords: set = None, lang: str = 'english'):
        """
        A constructor that initializes a Tagger object.
        :param sentences: A list of sentences to tag
        :param config: A dict of config properties. By default gets properties from sys config
        :param stopwords: A set of stopwords. Defaults to the standard stopword set by nltk.
        :param lang: A string specifying the language. Defaults to 'english'
        """
        self.sentences = sentences
        if stopwords is None:
            self.stop_words = set(nltk_stop_words.words(lang))
        else:
            self.stop_words = stopwords
        self.jar = config['stanford_jar']
        self.model = config['stanford_ner_model']
        self.tagger = StanfordNERTagger(self.model, self.jar, encoding='utf8')

    def tag_sentences(self):
        """
        A function that extracts concepts from a list of sentences.
        :return: a list of concepts
        """
        return [self.__tag_sentence__(sentence) for sentence in self.sentences]

    def __tag_sentence__(self, sentence: str) -> list:
        """
        A private helper class that invokes the stanford ner tagger.
        :param sentence: A string representation of the sentence
        :return: a list of tagged word tuples
        """
        words = nltk.word_tokenize(sentence)
        words = set(words) - self.stop_words
        return self.tagger.tag(words)
