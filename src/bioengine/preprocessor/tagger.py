import nltk
from nltk.tag.stanford import StanfordNERTagger

from settings import Config


class Tagger:
    """
    A class that abstracts the stanford NER tagger.
    """

    def __init__(self, sentences: list, config: dict = None, stopwords: set = None, lang: str = 'english'):
        """
        A constrictor that initializes a Tagger object.
        :param sentences: A list of sentences to tag
        :param config: A dict of config properties. By default gets properties from sys config
        :param stopwords: A set of stopwords. Defaults to the standard stopword set by nltk.
        :param lang: A string specifying the language. Defaults to 'english'
        """
        self.sentences = sentences
        if config is None:
            config = Config().get_config('ner')
        if stopwords is None:
            self.stop_words = set(stopwords.words(lang))
        else:
            self.stop_words = stopwords
        self.jar = config['stanford_jar']
        self.model = config['stanford_ner_model']
        self.tagger = StanfordNERTagger(self.model, self.jar, encoding='utf8')
        self.tags = [self.__tag_sentence__(sentence) for sentence in sentences]

    def __tag_sentence__(self, sentence: str) -> list:
        """
        A private helper class that invokes the stanford ner tagger.
        :param sentence: A string representation of the sentence
        :return: a list of tagged word tuples
        """
        words = nltk.word_tokenize(sentence)
        words = set(words) - self.stop_words
        return self.tagger.tag(words)
