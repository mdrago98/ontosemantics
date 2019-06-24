from pymetamap import MetaMap
import subprocess
from os import path

from src.settings import Config
from src.bioengine.nlp_processor.taggers.tagger import Tagger


class MetaMapTagger(Tagger):
    """
    A class that abstracts the MEtaMap tagger
    """
    def __init__(self, sentences: list, config: dict = None, start_server=True):
        """
        A constructor that initializes a MetaMap tagger.
        :param sentences: a list of sentences from which to extract the concepts
        :param config: a dictionary of config parameters for the tagger
        :param start_server: a flag that starts the server automatically.
        """
        self.sentences = sentences
        if config is None:
            config = Config().get_property('ner')
        self.tagger_root = config['metamap']
        self.tagger_location = path.join(self.tagger_root, 'bin', 'metamap18')
        # if start_server:
        #     self.server_running = self.start_server()
        # else:
        #     self.server_running = start_server
        self.tagger = MetaMap.get_instance(self.tagger_location)

    def start_server(self) -> bool:
        """
        A function that starts the necessary processes for MetaMap to work.
        :return: True IFF the processes are running
        """
        pos_server, word_disambig_server = self.get_executables()
        running = True
        try:
            subprocess.run([pos_server, 'start'], check=True)
            subprocess.run([word_disambig_server, 'start'])
        except subprocess.CalledProcessError:
            running = False
        return running

    def __enter__(self):
        self.start_server()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Overridden to automatically close down servers to free resources when object is freed
        :param exception_type: the exception
        :param exception_value: the value for the thrown exception
        :param traceback: the traceback
        :return: None
        """
        self.close()

    def close(self) -> bool:
        """
        A helper function to stop the server and free resources
        :return: True IFF server not running
        """
        pos_server, word_disambig_server = self.get_executables()
        not_running = True
        try:
            subprocess.run([pos_server, 'stop'], check=True)
            subprocess.run([word_disambig_server, 'stop'])
        except subprocess.CalledProcessError:
            not_running = False
        return not_running

    def get_executables(self, root_location: str = None) -> tuple:
        """
        A helper function that constructs locations for MetaMap executables
        :param root_location:
        :return:
        """
        if root_location is None:
            root_location = self.tagger_root
        pos_server = path.join(root_location, 'bin', 'skrmedpostctl')
        word_disambig_server = path.join(root_location, 'bin', 'wsdserverctl')
        return pos_server, word_disambig_server

    def tag_sentences(self):
        """
        A function that extracts concepts from a list of sentences.
        :return: a list of concepts
        """
        return self.tagger.extract_concepts(self.sentences)
