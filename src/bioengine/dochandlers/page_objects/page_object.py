from bs4 import BeautifulSoup
from abc import ABC, abstractmethod


class PageObject(ABC):
    """
    An abstract class for a page object. Implementation of PageObject must implement the get_text method
    """
    def clean_tags(self, soup: BeautifulSoup = None, pipeline: dict = None) -> BeautifulSoup:
        """
        A helper function that given a pipeline cleans elements found on the page.
        :param soup: A Beautiful object representing an html file
        :param pipeline: A dictionary representing elements to clean
        :return:
        """
        if soup is None:
            soup = self.soup
        if pipeline is None:
            pipeline = self.pipeline
        decompose = []
        for tag, attributes in pipeline.items():
            if attributes is not None:
                for attribute in attributes.items():
                    for value in attribute[1]:
                        decompose += soup.findAll(tag, {attribute[0]: value})
            else:
                decompose += soup.findAll(tag)
        for item in decompose:
            item.decompose()
        return soup

    @abstractmethod
    def get_text(self):
        """
        An abstract method that returns text paragraph by paragraph.
        :return: a list of paragraphs
        """
        pass
