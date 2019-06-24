import ssl
from urllib.error import HTTPError

from bs4 import BeautifulSoup
from abc import ABC, abstractmethod

from urllib.request import Request, urlopen


from src.settings import Config


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
                for elm, attribute in attributes.items():
                    if attribute is not None:
                        for value in attribute:
                            decompose += soup.findAll(tag, {elm: value})
            else:
                decompose += soup.findAll(tag)
        for item in decompose:
            item.decompose()
        return soup

    @staticmethod
    def open_page(page_name, config: Config = None):
        """
        A helper function that opens a webpage.
        :return: a string containing the raw html
        """
        if config is None:
            config = Config()
        req = Request(page_name, headers=config.get_property('headers'))
        context = ssl._create_unverified_context()
        return urlopen(req, context=context).read()

    def get_page(self) -> BeautifulSoup:
        """
        A helper method for opening a webpage and returns a parsed bs4 object
        :return:
        """
        try:
            result = BeautifulSoup(self.open_page(self.link), 'lxml')
        except HTTPError as error:
            result = None
        return result

    @abstractmethod
    def get_text(self):
        """
        An abstract method that returns text paragraph by paragraph.
        :return: a list of paragraphs
        """
        pass

    def get_article_details(self) -> tuple:
        """
        An abstract method that returns the article details
        :return: a tuple representing the title and a list of authors
        """
        pass

    def get_abstract(self) -> str:
        """
        An abstract method that extracts the abstract from a page object
        :return: a string representing the abstract
        """
        pass
