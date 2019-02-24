from abc import ABC, abstractmethod


class Tagger(ABC):
    @abstractmethod
    def tag_sentences(self):
        pass
