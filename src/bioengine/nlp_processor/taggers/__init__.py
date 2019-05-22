from settings import Config
from src.bioengine.nlp_processor.taggers.genia_tagger import GeniaTagger
from src.bioengine.nlp_processor.taggers.metamap_tagger import MetaMapTagger
from src.bioengine.nlp_processor.taggers.stanford_tagger import StanfordTagger
from src.bioengine.nlp_processor.taggers.tagger import Tagger
from src.bioengine.nlp_processor.taggers.becas_tagger import BecasTagger


class TaggerFactory:
    """
    A tagger factory class that simplifies the creation of new taggers.
    """
    @staticmethod
    def factory(sentences: list, tagger_type: str = 'becas') -> Tagger:
        """
        A factory that initializes the creation of a new Tagger.
        :param sentences: A list of sentences from which to extract the concepts from
        :param tagger_type: the type of tagger to use; genia | stanford | metamap
        :return: A tagger instance
        """
        conf = Config().get_property('ner')
        tagger_instance = None
        if tagger_type is 'genia':
            tagger_instance = GeniaTagger(sentences, config=conf)
        elif tagger_type is 'stanford':
            tagger_instance = StanfordTagger(sentences, config=conf)
        elif tagger_type is 'metamap':
            tagger_instance = MetaMapTagger(sentences, config=conf)
        elif tagger_type is 'becas':
            tagger_instance = BecasTagger(sentences)
        return tagger_instance
