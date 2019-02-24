from geniatagger import GeniaTagger as GenTag

from src.bioengine.preprocessor.taggers.tagger import Tagger


class GeniaTagger(Tagger):
    """
    A  class that abstracts a Genia Tagger.
    """
    def __init__(self, sentences: list, config: dict = None, args=None):
        """
        Constructs a new GeniaTagger.
        :param sentences: a list of sentences from which to extract the concepts
        :param config: a dictionary of config parameters
        :param args: a list of arguments for the genia tagger
        """
        if args is None:
            args = []
        self.sentences = sentences
        self.tagger = GenTag(config['genia_tagger'], args)

    def tag_sentences(self):
        """
        A function that extracts concepts from a list of sentences.
        :return: a list of concepts
        """
        return [self.tagger.parse(sentence) for sentence in self.sentences]
