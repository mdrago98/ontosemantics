import becas

from settings import Config
from src.bioengine.preprocessor.taggers import Tagger
from json import JSONDecoder


class BecasTagger(Tagger):
    """
    A wrapper class that calls a becas tagger
    """
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
        json_entities: str = self.becas.export_text(' '.join(self.sentences), 'json')
        return JSONDecoder().decode(json_entities)

    def meta_map_trigger_regex(self):
        """
        A function hat wraps around a regular expression for getting the object from the trigger value given by metamap.
        :return: a compiled regex
        """
        return compile(r'\[\"%*([a-zA-Z\d*\s]*)\"{1}')

