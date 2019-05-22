from abc import ABC, abstractmethod


class Tagger(ABC):
    """
    An abstract class that all taggers should implement.
    """
    @abstractmethod
    def tag_sentences(self):
        """
        A method that tags sentences using a tagger returning a list of concept tuples.
        :return: a list of concepts
        """
        pass
