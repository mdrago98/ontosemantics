import spacy
from spacy.tokens import Doc

from src.settings import Config
from os.path import join

from src.bioengine.nlp_processor.extensions.svo import get_noun_verb_chunks
from abc import ABC, abstractmethod


class SpacyI(ABC):
    """
    A factory for creating Spacy instances.
    """
    @staticmethod
    @abstractmethod
    def factory(config: dict):
        """
`       A factory method that initializes a new Spacy instance.
        :param config: the pipeline configuration
        :return: A Spacy model
        """
        pass


class MedicalSpacyFactory(SpacyI):
    """
    A factory class tasked with loading and creating Spacy instances.
    """
    @staticmethod
    def factory(enable_ner: bool = True, enable_benepar=True, config: dict = None):
        """
        A factory method that creates a spacy instance with medical pipelines and custom medical extensions.
        :return: A spacy model optimized for biological tasks
        """
        if config is None:
            config = Config().get_property('spacy')
        disable = config['pipeline']['disable'] if 'disable' in config['pipeline'] else []
        nlp = spacy.load('en_core_sci_md')
        # for stop_word in MedicalSpacyFactory._load_stop():
        #     lexeme = nlp.vocab[stop_word]
        #     lexeme.is_stop = True
        # # nlp.vocab.vectors.from_glove("/home/drago/thesis/BioSentVec_PubMed_MIMICIII-bigram_d700.bin")
        # if enable_ner:
        #     nlp.add_pipe(BiologicalNamedEntity(nlp))
        # if enable_benepar:
        #     nlp.add_pipe(BeneparComponent("benepar_en2"))
        Doc.set_extension('noun_verb_chunks', getter=get_noun_verb_chunks, force=True)
        return nlp

    @staticmethod
    def _load_stop(resource_dir: str = None, stop_word_file: str = None) -> list:
        """
        A static method that loads stopwords from a given file
        :param resource_dir: The resource folder
        :param stop_word_file: The name of the stop word file
        :return: a list of stopwords
        """
        if resource_dir is None:
            resource_dir = Config().get_property('resource_dir')
        if stop_word_file is None:
            stop_word_file = Config().get_property('medical_stop_word_file_name')
        path = join(resource_dir, stop_word_file)
        with open(path) as stop_word_file:
            stop_words = stop_word_file.read().splitlines()
        return stop_words
