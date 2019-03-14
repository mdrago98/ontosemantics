import becas

from settings import Config
from src.bioengine.preprocessor.taggers import Tagger
from json import JSONDecoder

class BecasTagger(Tagger):

    def __init__(self, sentences: list, config: dict = None):
        if config is None:
            config = Config().get_property('becas')
        self.sentences = sentences
        self.becas = becas
        self.becas.email = config['email']
        self.becas.tool = config['tool']

    def tag_sentences(self):
        """
        A helper function that returns a group of tagged entities from text
        :return:
        """
        return JSONDecoder().decode(self.becas.export_text(' '.join(self.sentences), 'json'))

