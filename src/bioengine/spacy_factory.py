import spacy
from benepar.spacy_plugin import BeneparComponent

from settings import Config
from src.bioengine.preprocessor.extensions import BecasNamedEntity
from os.path import join


class MedicalSpacyFactory:
    """
    A factory class tasked with loading and creating Spacy instances.
    """
    @staticmethod
    def factory():
        """
        A factory method that creates a spacy instance with medical pipelines and custom medical extensions.
        :return:
        """
        spacy.prefer_gpu()
        nlp = spacy.load('en_core_web_sm')
        for stop_word in MedicalSpacyFactory._load_stop():
            lexeme = nlp.vocab[stop_word]
            lexeme.is_stop = True
        nlp.add_pipe(BecasNamedEntity(nlp))
        nlp.add_pipe(BeneparComponent("benepar_en2"))
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
